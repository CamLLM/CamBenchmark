# 民航文本系统章节定位(Aircraft text chapter location)

## Embedding
### 任务构建
对于嵌入，该任务衡量模型在民航维修中对同类系统的文本聚集能力。
### 数据构建
我们选取了ATA中的21章至30章，将每一章的内容进行语义分割，定义的chunksize=300，将分块文本和其章节名称组织成（文本，章节标题）的数据格式，成为聚类数据集。该数据一共分为10类，用于Aircraft text chapter localization任务。
### 评估
我们使用k-means算法（k=10）进行文本聚类，并使用V-measure进行评测。

## LLM
pass
