# CAMB---民用航空维修评估

<h4 align="center">
    <p>
        <b>简体中文</b> |
        <a href="https://github.com/CamBenchmark/cambenchmark/blob/master/README_EN.md">English</a> 
    <p>
</h4>

<p align="center" style="display: flex; flex-direction: row; justify-content: center; align-items: center">
📄 <a href="" target="_blank" style="margin-right: 15px; margin-left: 10px">论文</a> • 
🏆 <a href="" target="_blank"  style="margin-left: 10px">排行榜</a> •
🤗 <a href="" target="_blank" style="margin-left: 10px">数据集</a> 
</p>


## 简介

pass

<p align="center"> <img src="images/camb.png" style="width: 85%;" id="title-icon">       </p>


## 排行榜

> **Note：**
> pass

以下表格分别显示了目前 Embedding 和 LLM 在民航维修领域中的性能表现。

<details>
<summary>Embedding</summary>
| Name                     | Size  | Mean<br>(Task) | Bitext<br>Mining | Classi-<br>fication | Cluster | Pair<br>Class | Retrieval | Reranker-<br>choice | Reranker-<br>text | Trouble-<br>Tree |
|--------------------------|-------|---------------:|-----------------:|-------------------:|--------:|--------------:|----------:|-------------------:|------------------:|-----------------:|
| Conan-embedding-v1       | 326 M |          55.14 |            49.46 |                 29 |   76.55 |         69.13 |     55.06 |              28.51 |             78.28 |               18 |
| gte-large-zh             | 326 M |          49.67 |            33.30 |                 25 |   73.68 |         65.33 |     46.76 |              29.24 |             74.39 |               28 |
| m3e-large                | 340 M |          43.37 |            22.08 |                 23 |   72.90 |         58.52 |     34.11 |              27.86 |             65.10 |               18 |
| BGE-large-zh-v1.5        | 671 M |          55.14 |            44.82 |                 31 |   78.65 |         61.11 |     63.27 |              30.19 |             77.58 |               22 |
| gte-Qwen2-1.5B-instruct  | 1.5 B |          59.60 |            68.07 |                 18 |   79.67 |         68.07 |     71.18 |              30.59 |             82.29 |               26 |
| gte-Qwen2-7B-instruct    | 7 B   |          63.42 |            80.25 |                 24 |   83.73 |         68.28 |     71.26 |              32.38 |             84.75 |               30 |
| Qwen3-Embedding-4B       | 4 B   |          62.95 |            81.37 |                 22 |   78.26 |         69.17 |     71.52 |              29.76 |             88.85 |               30 |
| Qwen3-Embedding-8B       | 8 B   |          66.27 |            80.79 |                 33 |   87.34 |         69.69 |     67.62 |              30.46 |             95.32 |               32 |
</details>

<details>
<summary>LLM</summary>

</details>


## 数据
开发和测试数据集。您也可以通过[Hugging Face]()获取我们的数据。

#### 快速使用

pass

#### 数据格式
pass。示例：

```

```

#### 提示
pass

以下是添加直接回答提示后的数据示例：

```
 
```

对于思路链提示，我们将提示从“请直接给出正确答案的选项”修改为“逐步分析并选出正确答案”。

#### 评估
pass

## 贡献
pass

## 引用

```

```
## 许可证

CAMB数据集采用
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).
