# 民航排故树推理问答(Troubleshooting tree-structured QA)

## Embedding

### 任务构建
对于embeddings，该任务为检索任务，旨在考察模型能否召回可能解决飞机故障的相应措施的能力。其中飞机故障为节点，解决措施为其子节点。
### 数据构建
原始数据格式为<Query-Gold_list>,由于Query可能和其他的Gold_list内容部分重合，因此我们进行两步去重，以此得到候选集：  
1.将所有的Gold_list的节点进行去重，得到节点集  
2.对于每一条Query，其都要和节点集进行去重，得到最终的候选节点集。  
对于每一条Query，模型在候选节点集中召回和Gold_list相同数量的节点个数。
### 评估
如果检索到的节点有一条在Gold_list，则标记为1；否则，标记为0。最终评估使用准确率进行。

### 结果分析
![image]images/Fault-tree-analy.png
图文[结果分析](embedding/analysis.py)

## LLM
pass
