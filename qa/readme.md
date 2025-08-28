# 民航维修知识问答(Civil aviation maintenance QA)

## Embedding
### 任务构建
检索任务：我们将查询集文本和语料库文本转换为向量，基于查询向量的相似性检索语料库，测试模型的检索能力。    
重排任务：我们使用语料库中的top10进行重新排序任务，以测试模型的精排能力，评估指标也是ndcg@10。
### 数据构建
对于每个问答对，我们使用BGE-large-v1.5、gte-Qwen2-1.5B-instruct、gte-Qwen2-7B-instruct、Qwen3-Embedding-4B和Qwen3-Embedding-8B从语料库中基于问答数据集中的问题标题检索出最相似的前10个文本。按照顺序从五个模型的结果中去除重复项，所得文本的数量在24到46之间。然后，我们使用LLM来协助排序，因为其在长文本排序任务上的能力略有限制。因此，我们将去重后的文本分为两部分：TOP10和TOP10到TOPN（N由去重后的文本数量决定），这两部分分别使用QWEN-235B-A22B进行排序，人工评估者根据顺序对结果进行评分，这些文本及其对应的分数形成用于检索任务的语料库。此外，TOP10文本及其分数形成用于重排文本任务的重排集。
![image](https://github.com/CamBenchmark/cambenchmark/blob/e0b2148cd4d51a00367e6f04970111858994dabb/images/retrieval_data.png)
### 评估
两个任务的评估指标都为ndcg@10。
### Qwen3-8B和Qwen3-4B结果分析

#### 1.在Retrieval阶段，Qwen3-8B比Qwen3-4B低4%，但在Reranker-text阶段，Qwen3-8B比Qwen3-4B高8%**
![image](https://github.com/CamBenchmark/cambenchmark/blob/9772f515592d3253da91c44cd37ebfa04a9844de/images/retrieavl_ana2.png)

#### 2.Qwen3-8B比Qwen3-4B更激进**
![image](https://github.com/CamBenchmark/cambenchmark/blob/e0b2148cd4d51a00367e6f04970111858994dabb/images/retrieval_ana1.png)
## LLM
pass
