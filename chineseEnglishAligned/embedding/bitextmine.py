import torch
from torch import Tensor
from modelscope import AutoTokenizer, AutoModel
import pandas as pd
from datasets import load_dataset
import numpy as np
from bitext_mining_utils import file_open, kNN, score_candidates
from collections import defaultdict
from utils.embedding.get_model_embedding import get_embedding

# device 
device = torch.device("cuda"  if torch.cuda.is_available()  else "cpu")

# load model
model_name = ""
model_path = ""
max_length = 2048    # 根据实际情形设置

# load data
# prompt
# task = "Retrieve parallel sentences: "
df = pd.read_excel("bitextmine评估集.xlsx")
queries = df["en"].tolist()
# queries = [get_detailed_instruct(task,i) for i in queries]
corpus = df["ch"].tolist()
# corpus = [get_detailed_instruct(task,i) for i in corpus]

# get embedding
query_embeddings = get_embedding(model_name,model_path,max_length,device,queries=queries,batch_size=10)
corpus_embeddings = get_embedding(model_name,model_path,max_length,device,queries=corpus,batch_size=10)

# 进行双语挖掘
# 设置标签
labels = defaultdict(lambda: defaultdict(bool))
# 这是一个双层嵌套的字典，没有出现过的健可以直接赋值
num_total_parallel = len(queries)
for i in range(len(queries)):
    src_id=i
    trg_id=i
    labels[src_id][trg_id] = True
    labels[trg_id][src_id] = True

# 目标句子和原始句子的标签
source_ids = list(range(len(queries)))
target_ids = list(range(len(queries)))


# 参数：
# We base the scoring on k nearest neighbors for each element
knn_neighbors = 4
# Min score for text pairs. Note, score can be larger than 1
min_threshold = 1
# 这个参数是默认的
# Do we want to use exact search of approximate nearest neighbor search (ANN)
# Exact search: Slower, but we don't miss any parallel sentences
# ANN: Faster, but the recall will be lower
use_ann_search = False
# Number of clusters for ANN. Optimal number depends on dataset size
ann_num_clusters = 32768
# How many cluster to explorer for search. Higher number = better recall, slower
ann_num_cluster_probe = 5

# 进行计算
x = query_embeddings
y = corpus_embeddings
x = x.astype(np.float32)   # 转换源向量
y = y.astype(np.float32)   # 转换目标向量 
x2y_sim, x2y_ind = kNN(x, y, knn_neighbors, use_ann_search, ann_num_clusters, ann_num_cluster_probe)
# Perform kNN in both directions
x2y_sim, x2y_ind = kNN(x, y, knn_neighbors, use_ann_search, ann_num_clusters, ann_num_cluster_probe)
x2y_mean = x2y_sim.mean(axis=1)

y2x_sim, y2x_ind = kNN(y, x, knn_neighbors, use_ann_search, ann_num_clusters, ann_num_cluster_probe)
y2x_mean = y2x_sim.mean(axis=1)

# Compute forward and backward scores
margin = lambda a, b: a / b
fwd_scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin)
bwd_scores = score_candidates(y, x, y2x_ind, y2x_mean, x2y_mean, margin)
fwd_best = x2y_ind[np.arange(x.shape[0]), fwd_scores.argmax(axis=1)]
bwd_best = y2x_ind[np.arange(y.shape[0]), bwd_scores.argmax(axis=1)]

indices = np.stack(
    [np.concatenate([np.arange(x.shape[0]), bwd_best]), np.concatenate([fwd_best, np.arange(y.shape[0])])], axis=1
)
scores = np.concatenate([fwd_scores.max(axis=1), bwd_scores.max(axis=1)])
seen_src, seen_trg = set(), set()

# Extract list of parallel sentences
bitext_list = []
for i in np.argsort(-scores):
    src_ind, trg_ind = indices[i]
    src_ind = int(src_ind)
    trg_ind = int(trg_ind)

    if scores[i] < min_threshold:
        break

    if src_ind not in seen_src and trg_ind not in seen_trg:
        seen_src.add(src_ind)
        seen_trg.add(trg_ind)
        bitext_list.append([scores[i], source_ids[src_ind], target_ids[trg_ind]])


# Measure Performance by computing the threshold
# that leads to the best F1 score performance
bitext_list = sorted(bitext_list, key=lambda x: x[0], reverse=True)

n_extract = n_correct = 0
threshold = 0
best_f1 = best_recall = best_precision = 0
average_precision = 0

for idx in range(len(bitext_list)):
    score, id1, id2 = bitext_list[idx]
    n_extract += 1
    if labels[id1][id2] or labels[id2][id1]:
        n_correct += 1
        precision = n_correct / n_extract
        recall = n_correct / num_total_parallel
        f1 = 2 * precision * recall / (precision + recall)
        average_precision += precision
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            threshold = (bitext_list[idx][0] + bitext_list[min(idx + 1, len(bitext_list) - 1)][0]) / 2

print("Best Threshold:", threshold)
print("Recall:", best_recall)
print("Precision:", best_precision)
print("F1:", best_f1)


