# aircraft_maintain_benchmark

aircraft_maintain_benchmark
如果只需要跑单个任务，只需要在config.yaml中data下面配置单个任务，其他配置保持不变。
用于构建一个多任务评估流水线，主要包含单选题评测、条目抽取、故障处理方案评分三大功能模块。以下是三个任务及config.yaml配置的详细说明：

multiple_choice:

传入类型: Excel 文件（固定）

路径: ../各机型考试题库(非常重要！) (version 2).xls（在file_path中配置）

工作表: A320AV（选择所需要评的工作表名称）

结构: 使用"试题题目"列作为问题，ABCD选项列存储选项

输出评估标准： 单选题的准确率

extracted_entries:

传入类型: JSON Lines 格式（固定）

../evaluation/0_gold_label.json（在file_path中配置）， 存储抽取任务的黄金标准数据
fim_21_documents.json中包含所有的条目，模型需要在所有条目中抽取10条，然后评估10条抽取条目中是否包含"golds_label"，如包含则为命中。

输出评估标准：抽取条目命中率

scoring:

传入类型: JSON 格式（固定）

../evaluation/冬季北方（提取后两项）.json（在file_path中配置），存储故障处理黄金标准数据（主要是故障现象和对应处理措施）

输出评估标准：技术和分析两个维度利用gpt（在evaluation_api中配置）进行打分

模型配置 (model)
需要在name中配置local_model或api_model，type中配置为local或api
本地模型:

路径: 指向本地模型文件

并行配置: tensor_parallel_size=2

API模型:

示例配置了GPT-4接口，其中api_key为可选参数，如不需要只要将整行注释掉

评估API (evaluation_api)
独立评估接口:
示例配置了GPT-4接口，其中api_key为可选参数，如不需要只要将整行注释掉

生成参数 (generation_params)
最大token数: 5000

温度系数: 0.001（低随机性）

思考模式: Qwen3可选择是否think模式

输出配置 (output)结果保存路径: ../../pcj-backup/workdesk/pipeline（在output_dir中配置）

最终输出：Model_Performance_Metrics_Comparison图片（包含各模型准确率、命中率和打分指标对比）和评估过程中生成的准确率、命中率和打分的数值和具体答案的json文件

# ======================= 第二版（修改为单条运行读写，中断后可恢复） =======================
运行脚本时可使用 --config 参数指定自定义路径
# 指定当前目录下的配置文件
python aircraft_pipeline_with_recovery.py --config my_config.yaml

# 指定绝对路径
python aircraft_pipeline_with_recovery.py --config /path/to/custom_config.yaml

文件配置改为列表，示例如下：
  multiple_choice:
    - type: excel
      file_path: "../516.xls"
      sheet_name: "Sheet1"
    - type: excel
      file_path: "../516.xls"
      sheet_name: "Sheet2"

如果运行时中断（例如在跑Sheet2时中断，只要重新传入Sheet2即可，不需要配置Sheet1重跑）。
output下新增  eval_dir: ../../pcj-backup/workdesk/pipeline2/eval 配置生成指标和最终结果的路径，需要和output_dir配置为不同路径
