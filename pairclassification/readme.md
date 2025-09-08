# 故障描述与 FIM 手册排故条目匹配(Fault description and FIM manual match)

## Embedding

### 任务构建

该任务旨在衡量模型在细颗粒度语料上的对齐能力，我们选用了句子配对数据集。里面包含了 FIM 手册中的故障问题-故障描述对和故障问题-故障对应条目对，以此考查模型将故障问题匹配到相应的故障描述和故障条目的能力。

### 数据构建

我们从 FIM 手册中选取故障问题和对该故障问题的详细描述，形成文本对。又从 FIM 手册的第 49 章中选取故障问题和该故障问题对应的条目名称（条目是指记录飞机故障的详细报告，用于维护分析和安全改进。），形成文本对。而后将两个文本对进行总和，形成正样本文本对。对于负样本文本对，我们使用 BGE-large-v1.5，gte-Qwen2-1.5B-instruct,gte-Qwen2-7B-instruct，Qwen3-Embedding-4B,Qwen3-Embedding-8B 平分正样本集，对于每一个故障问题，使用模型选取除对应的描述或条目名称之外与其相似度前 3 的描述或条目。最后，我们将正负样本构造成句子配对数据集，正负样本比为 1：3，用于 Fault description and FIM manual Match 任务。
![image](https://github.com/CamBenchmark/cambenchmark/blob/0727e4fcf0f3a175bb6e745ea3f3f1fb96b304db/images/pcls_datasets.png)

### 评估

使用 F1 进行评估

### 模型效率

鉴于句子配对任务易复现且数据量合适，我们使用句子配对任务对模型的效率进行了测评，Conan-embedding-v1 在性能和速度之间保持了平衡。
![image](https://github.com/CamBenchmark/cambenchmark/blob/4a62de338325a32546fe2aa1d087c602a42a428f/images/pcls_model_effi.png)

## LLM

### 任务构建

对于 LLM，该任务旨在评估模型在故障描述与 FIM 手册排故条目匹配中的判断能力。给定一个条目/故障描述和一个文本描述，要求模型判断两者是否匹配，这是一个二分类任务。该任务考察模型对民航维修领域专业术语和故障描述的理解能力，以及细粒度文本语义匹配的判断能力。

### 数据构建

使用与 Embedding 任务相同的数据集。

### 评估方法

采用二分类评估：1 表示匹配，0 表示不匹配
最终指标为正确数/总样本数$\times 100$
