# CAMB---民用航空维修评估基准

<h4 align="center">
    <p>
        <b>简体中文</b> |
        <a href="https://github.com/CamBenchmark/cambenchmark/blob/master/README_EN.md">English</a> 
    <p>
</h4>

<p align="center" style="display: flex; flex-direction: row; justify-content: center; align-items: center">
📄 <a href="" target="_blank" style="margin-right: 15px; margin-left: 10px">论文</a> • 
🏆 <a href="" target="_blank"  style="margin-left: 10px">评测结果</a> •
🤗 <a href="" target="_blank" style="margin-left: 10px">数据集</a> 
</p>


## 简介

民航维修领域，行业标准严苛，知识密集，典型的富含知识和推理的业务场景。我们结合民航维修领域的业务和对大模型的理解，建设并开源了一套民航维修领域工业级的大模型评测基准(Civil Aviation Maintenance Benchmark)，既可以评测向量嵌入模型(Embedding)，也可以评测大语言模型(LLM)，同时也在一定程度上弥补了目前大多仅在数学和代码领域研究大模型推理的评测短板。

<p align="center"> <img src="images/camb.png" style="width: 85%;" id="title-icon">       </p>

## 评测任务

CAMB评测基准涵盖民航维修场景中的 7 个任务，涉及到 8 个评估数据集：
* 民航术语双语对齐(Alignment bilingual terminology)
    * Embedding，构建为双语挖掘(BitextMining)任务
    * LLM，构建为中英翻译(Translation)任务
* 民航故障系统定位(Aircraft fault system location)
    * Embedding，利用“民航飞行器一级系统”向量重排(Rerank)构建为分类(Classification)任务
    * LLM，构建为分类(Classification)任务
* 民航文本系统章节定位(Aircraft text chapter location)
    * Embedding，利用“章节系统向量重排(Rerank)”构建为聚类(Cluster)任务
    * LLM，构建为文本分类(Classification)任务
* 故障描述与FIM手册排故条目匹配(Fault description and FIM manual match)
    * Embedding，利用句对向量，构建为匹配(PairClass)任务
    * LLM，构建为匹配(FIM Manual Match)任务
* 民航维修执业资格及上岗考试(Civil aviation maintenance Multiple choice)
    * Embedding，利用“选项向量重排(Rerank)”构建为重排(Rerank-choice)任务
    * LLM，构建为选择题(Multiple-Choice)任务
* 民航维修知识问答(Civil aviation maintenance QA)
    * Embedding，分别构建为文本检索(Retrieval)和文本重排(Rerank-text)任务
    * LLM，构建为问答(maintenance QA)任务
* 民航排故树推理问答(Troubleshooting tree-structured QA)
    * Embedding，利用“候选故障原因向量重排”构建为树节点重排(TroubleTree)任务
    * LLM，构建为树结构溯因推理(Reasoning on Tree)任务

## 评测结果

> **Note：**
> 目前评测时间截止到2025年8月22日

以下表格分别显示了目前 Embedding 和 LLM 在 CAMB 的性能表现。

<details>
<summary>Embedding</summary>
<p align="center"> <img src="images/embedding_result.png" style="width: 85%;" id="title-icon">       </p>
</details>

<details>
<summary>LLM</summary>
<p align="center"> <img src="images/llm_result.png" style="width: 85%;" id="title-icon">       </p>
<p align="center"> <img src="images/whether_think_choice_result.png" style="width: 85%;" id="title-icon">       </p>

</details>


## 评估数据集
除了此项目外，您也可以通过[Hugging Face]()或者[ModelScope]()获取我们的数据。

#### 数据集说明

* [chineseEnglishAligned](chineseEnglishAligned)
    * [bitextmine.xlsx](chineseEnglishAligned/bitextmine.xlsx)，[更多信息](chineseEnglishAligned/readme.md)

        <details>
        <summary>数据展示</summary>
        </details>

* [classification](classification)
    * [classification.xlsx](classification/classification.xlsx)，[更多信息](classification/readme.md)

        <details>
        <summary>数据展示</summary>
        </details>

* [cluster](cluster)
    * [cluster.xlsx](cluster/cluster.xlsx)，[更多信息](cluster/readme.md)

        <details>
        <summary>数据展示</summary>
        </details>

* [pairclassification](pairclassification)
    * [paircls.xlsx](pairclassification/paircls.xlsx)，[更多信息](pairclassification/readme.md)

        <details>
        <summary>数据展示</summary>
        </details>

* [multipleChoice](multipleChoice)
    * [air_choice.xlsx](multipleChoice/air_choice.xlsx)，[更多信息](multipleChoice/readme.md)
        <details>
        <summary>数据展示</summary>
        </details>

* [qa](qa)
    * Embedding
        * [queries.jsonl](qa/queries.jsonl), [corpus.jsonl](qa/corpus.jsonl)，用于[文本检索(Retrieval)任务]()，[数据展示及更多信息](qa/readme.md)
        * [rerank_qa.xlsx](qa/rerank_qa.xlsx)，用于[文本重排(Rerank-text)任务]()，[数据展示及更多信息](qa/readme.md)
    * LLM
        * [qa.xlsx](qa/qa.xlsx)，用于[问答(maintenance QA)任务]()，[数据展示及更多信息](qa/readme.md)   
       
* [faultTree](faultTree)
    * [trouble_tree.jsonl](faultTree/trouble_tree.jsonl)，[更多信息](faultTree/readme.md)
        <details>
        <summary>数据展示</summary>
        
            {
                "prompt": "## 历史推理路径\n- 1-1: 近期NG飞机经常在空中出现无线电高度指示不正常故障，地面却测试无线电高度表正常。并且由于无线电高度表故障，导致许多使用无线电高度信号的系统发生故障。以下是无线电高度表异常的相关的故障报告： 1.左右座无线电高度表指示偏差过大； 2.进近时候双通道自动驾驶不能衔接或者断开； 3.飞行指引杆不正常消失； 4.起飞后、进近阶段、复飞阶段不正常的构型警告； 5.音响高度报告功能失效； 6.速度刹车不在预位灯在自动速度刹车预位时不正常点亮； 自动油门慢车状态在FMA上不正常显示。 | 知识: 原理分析 无线电高度表收发机产生发射电波信号经过传输电缆发送到发射天线，发射天线把发射电波信号发送出去，遇到地面后反射；接收天线接收经地面反射的电波信号，并把接收到的电波信号反馈回无线电高度表收发机；无线电高度表收发机利用发送信号和接收信号的时间差来计算出飞机所在的无线电高度，并把解算出来的高度数据用总线形式发送给DEU和其他飞机各个系统，分别用来在PFD上显示无线电高度和用于其它各个系统的功能计算。无线电高度系统不正常会导致PFD上显示的无线电高度异常，而且还会导致其他利用无线电高度信号计算的系统发生故障。.造成无线电高度表故障的故障部件： 1.发射天线、接收天线（M1737/M1739/M1738/M1740） 2.线路故障：天线到收发机之间的同轴电缆、同轴电缆接头； 3.收发机（M1735/1736） 4.上述的故障部件的组合 ", 
                
                "gold_list": ["在故障再现的时候，必须首先按照FIM34-33 TASK 801 LRRA BITE PROCEDURE 进行无线电高度系统 BITE，并记录故障灯或故障代码，切忌不要先复位跳开关，把故障信息消除。然后根据故障代码按照FIM手册排故"]
            }
            
        </details>

#### 提示词(prompt)
pass

以下是添加直接回答提示后的数据示例：

```
 
```

对于思路链提示，我们将提示从“请直接给出正确答案的选项”修改为“逐步分析并选出正确答案”。

#### 评估脚本
pass

#### LLM-as-judger与人工评估一致性验证

## 贡献
pass

## 引用

```

```
## 许可证

CAMB数据集采用
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).




