import torch
from torch import Tensor
from modelscope import AutoTokenizer, AutoModel
import pandas as pd
from utils.embedding.get_model_embedding import get_embedding

# load model
model_name = ""
model_path = ""
max_length = 2048    # 根据实际情形设置
device  = torch.device("cuda"  if torch.cuda.is_available()  else "cpu")

# task = "Retrieve text based on user query"

# load data
df = pd.read_excel("一级系统.xlsx")
s1 = df["描述"].tolist()
# s1 = [get_detailed_instruct(task,i) for i in s1]
gold = df["一级系统"].tolist()
labels = df.iloc[0,2:-1].tolist()


# 获取labels的向量
label_embeddings = get_embedding(model_name,model_path,max_length,device,queries=labels,batch_size=10)
# 获得s1向量
s1_embeddings = get_embedding(model_name,model_path,max_length,device,queries=labels,batch_size=10)

# 进行分类
correct = 0
num = len(gold)
for i in range(0,len(s1_embeddings)):
    similarity = s1_embeddings[i] @ label_embeddings.T
    similarity = similarity.tolist()
    pred = similarity.index(max(similarity))
    if gold[i] == labels[pred]:
        correct+=1
        

print(correct)
print(num)
print(correct/num)