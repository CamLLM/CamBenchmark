import torch
import torch.nn.functional as F
from torch import Tensor
from modelscope import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm import trange
import os
import json

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

# data
df = pd.read_excel("retrieval_dataset.xlsx")
df = df.reset_index(drop=True) 
ndcg_1 = []
ndcg_3 = []
ndcg_5 = []
ndcg_10 = []
for i in range(len(df)):
    query = df.at[i,"query"]
    corpus = eval(df.at[i,"text_list"])[:10]
    corpus = eval(corpus)
    query_embedding = get_embedding(model_name,model_path,max_length,device,queries=query,batch_size=10)
    corpus_embedding = get_embedding(model_name,model_path,max_length,device,queries=corpus,batch_size=10)
    #相似度
    similarity = query_embedding@corpus_embedding.T
    similarity = similarity.tolist()[0]
    #取顺序（降序）
    pred_rank = np.argsort(similarity)[::-1].tolist()
    #相关性
    true_score = eval(df.at[i,"text_list"])[:10]
    true_score = eval(true_score)
    #模型相关性
    pred_score = [true_score[i] for i in pred_rank]
    #计算ndcg值
    ndcg_1.append(compute_ndcg(1,pred_score,true_score))
    ndcg_3.append(compute_ndcg(3,pred_score,true_score))
    ndcg_5.append(compute_ndcg(5,pred_score,true_score))
    ndcg_10.append(compute_ndcg(10,pred_score,true_score))
    

print(np.mean(ndcg_1),np.mean(ndcg_3),np.mean(ndcg_5),np.mean(ndcg_10))