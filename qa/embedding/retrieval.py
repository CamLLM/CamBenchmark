import torch
from torch import Tensor
from modelscope import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from tqdm import trange
import heapq
from datasets import load_dataset
from utils.embedding.get_model_embedding import get_embedding,get_detailed_instruct

def process_query(example, task):
    example["text"] = get_detailed_instruct(task, example["text"])
    return example 

def score_function(a,b):
    return a @ b.T

def compute_ndcg(k,pred_score,true_score):
    dcg = sum((2**rel - 1)/np.log2(i+2) for i,rel in enumerate(pred_score[:k]))
    idcg = sum((2**rel - 1)/np.log2(i+2) for i,rel in enumerate(true_score[:k]))
    return dcg/idcg

# device 
device = torch.device("cuda"  if torch.cuda.is_available()  else "cpu")

# load model
model_name = ""
model_path = ""
max_length = 2048    # 根据实际情形设置

# prompt
# task = "Retrieve relevant aviation maintenance passages that address the query: "

# 数据
corpus = load_dataset("json",data_files= "retrieval/corpus.jsonl")
corpus = corpus['train']
queries = load_dataset("json",data_files= "retrieval/queries.jsonl")
queries = queries['train']
# 在queries中加入task
# queries = queries.map(lambda  x: process_query(x, task))
relevant_docs_data = load_dataset("csv",data_files="retrieval/qrels/test.csv")

# Convert the datasets to dictionaries
corpus = dict(zip(corpus["_id"], corpus["text"]))  # Our corpus (cid => document)
queries = dict(zip(queries["_id"], queries["text"]))  # Our queries (qid => question)
relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
for qid, corpus_ids in zip(relevant_docs_data["train"]["query-id"], relevant_docs_data["train"]["corpus-id"]):
    qid = str(qid)
    corpus_ids = str(corpus_ids)
    if qid not in relevant_docs:
        relevant_docs[qid] = set()
    relevant_docs[qid].add(corpus_ids)

# 分开编号和值
queries_ids = []
for qid in queries:
    if qid in relevant_docs and len(relevant_docs[qid]) > 0:
        queries_ids.append(qid)
queries = [queries[qid] for qid in queries_ids]
corpus_ids = list(corpus.keys())
corpus = [corpus[cid] for cid in corpus_ids]

# get embedding
query_embeddings = get_embedding(model_name,model_path,max_length,device,queries=queries,batch_size=10)
corpus_embeddings = get_embedding(model_name,model_path,max_length,device,queries=corpus,batch_size=10)

# # 储存向量
# # 保存 
# # torch.save(query_embeddings, "query_embeddings_bge.pt" )
# # torch.save(corpus_embeddings,  'corpus_embeddings_bge.pt')   # 或 .pth 后缀 
# # 加载向量
# query_embeddings = torch.load('query_embeddings_bge.pt',  map_location=device)
# corpus_embeddings = torch.load("corpus_embeddings_bge.pt",map_location = device)


# 计算相似度
queries_result_list = {}
name = "dot_score"
queries_result_list[name] = [[] for _ in range(len(query_embeddings))]
corpus_chunk_size = 10
# 分块进行计算
for corpus_start_idx in trange(0, len(corpus), corpus_chunk_size):
    corpus_end_idx = min(corpus_start_idx + corpus_chunk_size, len(corpus))
    sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]
    
    pair_scores = score_function(query_embeddings, sub_corpus_embeddings)
    # Get top-k values
    ndcg_at_k = [1,3,5,10]
    max_k = max(ndcg_at_k)
    pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False)
    pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
    pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

    for query_itr in range(len(query_embeddings)):
        for sub_corpus_id, score in zip(
            pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]
        ):
            corpus_id = corpus_ids[corpus_start_idx + sub_corpus_id]
            if len(queries_result_list[name][query_itr]) < max_k:
                heapq.heappush(queries_result_list[name][query_itr], (score, corpus_id))
            else:
                heapq.heappushpop(queries_result_list[name][query_itr], (score, corpus_id))

#从元组-字典
for query_itr in range(len(queries_result_list[name])):
    for doc_itr in range(len(queries_result_list[name][query_itr])):
        score, corpus_id = queries_result_list[name][query_itr][doc_itr]
        queries_result_list[name][query_itr][doc_itr] = {"corpus_id": corpus_id, "score": score}

# 使用字典推导式储存结果：
result_top10 = {
    f"top{i+1}":[
        corpus[int(sorted(q,key = lambda x:-x["score"])[i]["corpus_id"])]
        for q in queries_result_list[name]
    ]
    for i in range(10)
}
result_top10["问题列"] = queries
result_top10["答案列"] = corpus[:202]
result_top10 = pd.DataFrame(result_top10)

# 加载结果数据集
all_questions = pd.read_excel("retrieval_dataset.xlsx")

# 计算
ndcg_k = [1,3,5,10]
ndcg_list = [[] for _ in ndcg_k]
ndcg = dict(zip(ndcg_k,ndcg_list))
for i in range(len(all_questions)):
    pred_doc = result_top10.loc[i,[f"top{j+1}" for j in range(10)]]
    true_doc = eval(all_questions.at[i,"text_list"])
    true_score = eval(all_questions.at[i,"score_list"])
    pred_score = []
    for idx,k in enumerate(pred_doc):
        try:
            pred_score.append(true_score[true_doc.index(k)])
        except:
            pred_score.append(-1)

    ndcg[1].append(compute_ndcg(1,pred_score,true_score))
    ndcg[3].append(compute_ndcg(3,pred_score,true_score))
    ndcg[5].append(compute_ndcg(5,pred_score,true_score))
    ndcg[10].append(compute_ndcg(10,pred_score,true_score))

ndcg[1] = np.mean(ndcg[1])
ndcg[3] = np.mean(ndcg[3])
ndcg[5] = np.mean(ndcg[5])    
ndcg[10] = np.mean(ndcg[10])


print(ndcg)
 