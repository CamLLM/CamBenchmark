import torch
from torch import Tensor
from modelscope import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from tqdm import trange
from datasets import load_dataset
from utils.embedding.get_model_embedding import get_embedding


def score_function(a,b):
    return a @ b.T

# device 
device = torch.device("cuda"  if torch.cuda.is_available()  else "cpu")

# load model
model_name = ""
model_path = ""
max_length = 2048    # 根据实际情形设置

# 数据
df = pd.read_excel("fault_tree_test.xlsx")
result = []
metric = 0
for i in trange(len(df)):
    query = df.iloc[i]["query"]
    answer = eval(df.iloc[i]["answer"])
    corpus = eval(df.iloc[i]["候选集"])
    
    # 向量
    query_embeddings = get_embedding(model_name,model_path,max_length,device,queries=query,batch_size=10)
    corpus_embeddings = get_embedding(model_name,model_path,max_length,device,queries=corpus,batch_size=10)
    
    score = score_function(query_embeddings,corpus_embeddings)
    score = score.cpu().tolist()[0]
    len_result = df.iloc[i]["题目长度"]
    result_index = np.argsort(score)[::-1][:len_result]
    correct = 0
    result_corpus = []
    for k in result_index:
        if corpus[k] in answer:
            correct = correct+1
        result_corpus.append(corpus[k])
    if correct>0:
        metric +=1

    result.append({
        "prompt":df.iloc[i]["query"],
        "answer":answer,
        "pred":result_corpus,
        "len_HX":df.iloc[i]["候选集的长度"],
        "len_answer":len_result,
        "len_correct_pred":correct,
    })


print(metric)
print(len(df))
    
# 储存结果
# pd.DataFrame(result).to_excel(f"tree_{model_name}.xlsx")
