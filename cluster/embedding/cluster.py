from sklearn.cluster import KMeans
import torch
from torch import Tensor
from modelscope import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.metrics  import v_measure_score
from utils.embedding.get_model_embedding import get_embedding

# device 
device = torch.device("cuda"  if torch.cuda.is_available()  else "cpu")

# load model
model_name = ""
model_path = ""
max_length = 2048    # 根据实际情形设置

# load data
task = "Identify categories in user passages."
df = pd.read_excel("聚类评估集.xlsx")
df["标签"] = df["类别"].factorize()[0] 
corpus = df["文本"].tolist()
# corpus = [get_detailed_instruct(task,i) for i in corpus]
label = df["标签"].tolist()

# get embedding
corpus_embeddings = get_embedding(model_name,model_path,max_length,device,queries=corpus,batch_size=10)

# Kmeans聚类
num_clusters = 10
clustering_model = KMeans(n_clusters=num_clusters,n_init=10)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_


clustered_sentences = [[] for i in range(num_clusters)]

for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])
    
# 计算v-measure
v_score = v_measure_score(label, cluster_assignment)
print(f"V-measure Score: {v_score:.4f}")