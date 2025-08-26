import torch
from torch import Tensor
from modelscope import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import paired_cosine_distances
from utils.embedding.get_model_embedding import get_embedding

# device 
device = torch.device("cuda"  if torch.cuda.is_available()  else "cpu")

# load model
model_name = ""
model_path = ""
max_length = 2048    # 根据实际情形设置

# task = "Retrieve text that are semantically similar to the given text."
# data
df = pd.read_excel("加入负例paircls评估集.xlsx")
s1 = df["question"].tolist()
s2 = df["description"].tolist()
# s1 = [get_detailed_instruct(task,i) for i in s1]
# s2 = [get_detailed_instruct(task,i) for i in s2]

# get embedding
s1_embeddings = get_embedding(model_name,model_path,max_length,device,queries=s1,batch_size=10)
s2_embeddings = get_embedding(model_name,model_path,max_length,device,queries=s2,batch_size=10)



# 获得相似度
scores = paired_cosine_distances(s1_embeddings, s2_embeddings)

# 获得标签
labels = df["label"].tolist()

def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool):
    assert len(scores) == len(labels)

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    rows = list(zip(scores, labels))

    rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

    best_f1 = best_precision = best_recall = 0
    threshold = 0
    nextract = 0
    ncorrect = 0
    total_num_duplicates = sum(labels)

    for i in range(len(rows) - 1):
        score, label = rows[i]
        nextract += 1

        if label == 1:
            ncorrect += 1

        if ncorrect > 0:
            precision = ncorrect / nextract
            recall = ncorrect / total_num_duplicates
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                threshold = (rows[i][0] + rows[i + 1][0]) / 2

    return best_f1, best_precision, best_recall, threshold


best_f1, best_precision, best_recall, threshold = find_best_f1_and_threshold(scores,labels,False)




print("best_f1, best_precision, best_recall, threshold:",best_f1, best_precision, best_recall, threshold)
