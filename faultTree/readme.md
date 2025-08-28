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
为了更稳健地评估模型的结果，从三个维度进行评估，优先级从（1）到（3）：  
（1）完全错误的问题越少越好；  
（2）在完全错误的问题中，简单的越少越好；  
（3）在不完全错误的问题中，难度越高的越好。  
#### 难度评级 
#### 情形一：完全错误（所有节点全错）
- **事件定义**：模型在所有节点上均未能答对。
- **概率公式**：
$$
P_{\text{all wrong}} = \frac{\displaystyle \binom{H - A}{A}}{\displaystyle \binom{H}{A}}
$$
- **难度解释**：
$$
P_{\text{all wrong}} \uparrow \quad\Rightarrow\quad \text{难度} \uparrow 
$$
 
---
 
#### 情形二：不完全错误（至少答对一个节点）
- **事件定义**：模型在 \(C\) 个节点上答对，至少 \(C \ge 1\)。
- **概率公式**：
$$
P_{\text{at least one correct}} = \frac{\displaystyle \binom{H - A}{A - C} \binom{A}{C}}{\displaystyle \binom{H}{A}}
$$
- **难度解释**：
$$
P_{\text{at least one correct}} \uparrow \quad\Rightarrow\quad \text{难度} \downarrow 
$$
 
---
 
#### 符号说明 
| 符号 | 含义 |
|---|---|
| \(H\) | 题库总题数 |
| \(A\) | 被考察的节点数 |
| \(C\) | 模型实际答对的节点数 |



在第一维度上，gte-large-zh、gte-1.5B、gte-7B、Qwen3-4B和Qwen3-8B表现最佳，分别正确回答了14、13、15、15、16个问题。五个模型之间的差异小于3，表明水平一致，因此进行了第二维度的评估。  
在第二维度上，Qwen3-8B = Qwen3-4B > gte-7B > gte-large-zh > gte-1.5B  
在第三维度上，答对条数一致比较才有意义，由于答对条数为1的数目占大头，因此对模型答对条数为1进行分析，Qwen3-8B > Qwen3-4B > gte-1.5B= gte-large-zh > gte-7B  
总体而言，Qwen3-8B > Qwen3-4B > gte-7B > gte-large-zh > gte-1.5B-instruct。  


<p align="center"> <img src="images/Fault-tree-analy.png" style="width: 85%;" id="title-icon">       </p>


## LLM
pass
