from modelscope import AutoTokenizer, AutoModel
import torch
import pandas as pd
from tqdm import tqdm
from utils.embedding.get_model_embedding import get_embedding

# device 
device = torch.device("cuda"  if torch.cuda.is_available()  else "cpu")

# load model
model_name = ""
model_path = ""
max_length = 2048    # 根据实际情形设置

# load data
df = pd.read_excel("air_choice.xlsx")
data = []
for i in range(len(df)):
    data.append({
        "uid":df.iloc[i]["uid"],
        "题目":df.iloc[i]["试题题目"],
        "选项A":df.iloc[i]["选项A"],
        "选项B":df.iloc[i]["选项B"],
        "选项C":df.iloc[i]["选项C"],
        "选项D":df.iloc[i]["选项D"],
        "答案":df.iloc[i]["答案"]
    })

correct = 0
for i in tqdm(range(len(data))):
    queries = [str(data[i]["题目"])]
    documents=[]
    if data[i]["选项A"] != "":
        documents.append(str(data[i]["选项A"]))
    if data[i]["选项B"] != "":
        documents.append(str(data[i]["选项B"]))
    if data[i]["选项C"] != "":
        documents.append(str(data[i]["选项C"]))
    if data[i]["选项D"] != "":
        documents.append(str(data[i]["选项D"]))
    query_embeddings = get_embedding(model_name,model_path,max_length,device,queries=queries,batch_size=1)
    corpus_embeddings = get_embedding(model_name,model_path,max_length,device,queries=documents,batch_size=1)
    similarity = (query_embeddings @ corpus_embeddings.T)
    similarity = similarity.tolist()[0]
    predict_answer = ["A", "B", "C", "D"][similarity.index(max(similarity))]
    if predict_answer == data[i]["答案"]:
        correct += 1
print(correct)
print(len(data))

