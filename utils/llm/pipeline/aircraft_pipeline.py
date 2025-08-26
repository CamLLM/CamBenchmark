"""航空维修数据生成内容质量评估流水线"""
import os
import psutil
import gc
import re
import requests
import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import yaml
import json
import jsonlines
from typing import List, Dict, Any, Union, Optional
from ast import literal_eval
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# ======================= 数据读取模块 =======================
class DataReader:
    """抽象数据读取类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def read_data(self) -> pd.DataFrame:
        raise NotImplementedError

# 数据读取
class ExcelDataReader(DataReader):
    def read_data(self) -> pd.DataFrame:
        return pd.read_excel(
            self.config["file_path"],
            sheet_name=self.config.get("sheet_name")
        )

class CSVDataReader(DataReader):
    def read_data(self) -> pd.DataFrame:
        return pd.read_csv(
            self.config["file_path"],
            encoding=self.config.get("encoding", "utf-8")
        )

class JSONDataReader(DataReader):
    def read_data(self) -> pd.DataFrame:
        return pd.read_json(self.config["file_path"])

class JSONLinesDataReader(DataReader):
    def read_data(self) -> pd.DataFrame:
        with jsonlines.open(self.config["file_path"]) as reader:
            return pd.DataFrame(list(reader))

# ======================= 提示生成模块 =======================
class PromptGenerator:
    def __init__(self, config: Dict[str, Any], data_df: pd.DataFrame):
        self.config = config
        self.data_df = data_df

    def generate_prompts(self) -> List[str]:
        prompt_type = self.config["type"]
        if prompt_type == "multiple_choice":
            return self._generate_mc_prompts()
        elif prompt_type == "extracted_entries":
            return self._generate_entry_prompts()
        elif prompt_type == "scoring":
            return self._generate_scoring_prompts()
        else:
            raise ValueError(f"未知提示类型: {self.prompt_type}")

    def _generate_mc_prompts(self) -> List[str]:
        # config = self.config["multiple_choice"]
        config = self.config
        return [
            f"问题：{row[config['question_column']]}\n"
            f"A. {row[config['choices_columns'][0]]}\n"
            f"B. {row[config['choices_columns'][1]]}\n"
            f"C. {row[config['choices_columns'][2]]}\n"
            f"D. {row[config['choices_columns'][3]]}\n"
            "请直接回答正确选项的字母（A/B/C/D），答案必须是单个字母，不要字母以外的任何内容。"
            for _, row in self.data_df.iterrows()
        ]

    def _generate_entry_prompts(self) -> List[str]:
        # config = self.config["extract_entries"]
        config = self.config
        entry_reader = create_data_reader(config["entries_source"])
        entries = entry_reader.read_data()[config["entry_column"]].tolist()

        return [
            f"故障描述：{row[config['query_column']]}\n"
            f"备选条目：\n" + "\n".join(entries) + "\n"
            '''请选择最相关的10个标签（JSON列表格式必须严格按照以下格式）" + "```json\n[\n    "Right PACK TRIP OFF Light Illuminated and Will Not Reset",\n    "Cabin Temperature Controller BITE Procedure",\n    "DUCT OVERHEAT Light On, Cannot Reset, CONT CABIN",\n    "DUCT OVERHEAT Light On, Cannot Reset, PASS CABIN",\n    "Pack Air Flow is Too Low or Too High",\n    "Pack Outlet Temperature Too Hot or Too Cold",\n    "Pack Temperature Control",\n    "Right Pack Temperature Sensor (Primary)",\n    "Right Pack Temperature Sensor (Backup)",\n    "Right Pack Ram Air Actuator"\n]\n```"'''
            for _, row in self.data_df.iterrows()
        ]

    def _generate_scoring_prompts(self) -> List[str]:
        # config = self.config["scoring"]
        config = self.config
        with open(config['knowledge_base'], "r", encoding='utf-8') as f:
            knowledge = json.load(f)

        return [
            f"""# 任务概述
            您作为工程方案专家，需根据故障描述生成解决方案
            
            # 输入基准
            **故障描述**：<<{row[config['query_column']] or 'N/A'}>>
            **技术规范**：<<{knowledge[0].get('技术文件', 'N/A')}>>
            
            # 工作流程
            生成要求：
            1. 按照**故障描述**生成方案。
            2. 可以参考技术规范。
            
            !! 重要注意事项 !!
            最后几行生成方案必须严格遵照以下例子的格式，不能有任何不一样的格式内容：
            ##solution##
            {{"solution": ["冻雨天气下，如飞机发生前轮转弯卡阻，转弯失效，优先检查前轮转弯控制钢索、导向盘、加法机构及相关部件的结冰情况，如有结冰用加温车、加热管或热风枪等外场加热设备对相关结冰区域进行加热除冰。", "加热除冰完成后，执行前轮转弯系统测试，如故障依旧，按FIM排故。"]}}"""
            for _, row in self.data_df.iterrows()
]

# ======================= 模型生成模块 =======================
class BaseGenerator:
    """生成器基类"""
    def generate_text(self, prompts: List[str], params: Dict) -> List[str]:
        raise NotImplementedError
        
# ======================= 本地模型生成模块 =======================
class LocalLLMGenerator(BaseGenerator):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left"  
        )
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=config.get("tensor_parallel_size", 1),
            trust_remote_code=True
        )

    def generate_text(self, prompts: List[str], params: Dict) -> List[str]:
        generation_config = {
            "max_tokens": params.get("max_tokens", 1000),
            "temperature": params.get("temperature", 0.0),
            "top_p": params.get("top_p", 1.0),
            "stop_token_ids": [self.tokenizer.eos_token_id],
            "seed": 42
        }

        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=params.get("enable_thinking", False)
            ) for prompt in prompts
        ]

        sampling_params = SamplingParams(**generation_config)
        results = []
        #逐条请求获取答案
        for fp in formatted_prompts:
            try:
                outputs = self.llm.generate([fp], sampling_params)
                if outputs:
                    generated_text = outputs[0].outputs[0].text
                    # print(f"generated_text{generated_text}")
                    results.append(generated_text)
                    # print(f"results_list{results}")
                else:
                    results.append("")
            except Exception as e:
                # Log error and append placeholder to maintain list consistency
                print(f"Error generating text for prompt: {e}")
                results.append("") 
        return results
        # outputs = self.llm.generate(formatted_prompts, sampling_params)
        # return [output.outputs[0].text for output in outputs]
# ======================= API服务生成模块 =======================
class APIGenerator(BaseGenerator):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_url = config["api_url"]
        self.api_key = config.get("api_key", None)
        self.model_name = config["model_name"]

    def generate_text(self, prompts: List[str], params: Dict) -> List[str]:

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        } if self.api_key else {'Content-Type': 'application/json'}

        responses = []
        for prompt in prompts:
            payload = json.dumps({
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": params.get("temperature", 0.0),
                "max_tokens": params.get("max_tokens", 1000),
                "top_p": params.get("top_p", 1.0)
            })

            try:
                response = requests.post(
                    self.api_url, 
                    headers=headers,
                    data=payload,
                    timeout=30
                )
                response.raise_for_status()
                content = response.json()['choices'][0]['message']['content']
                responses.append(content)
            except Exception as e:
                print(f"API请求失败: {str(e)}")
                responses.append("")
        return responses
        
# ======================= 结果处理模块 =======================
class AnswerExtractor:
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.patterns = {
        }
        self.think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
        
        if task_name == "multiple_choice":
            self.patterns["option"] = re.compile(r'(?i)([A-D])')

        elif task_name == "extracted_entries":
            self.patterns["entries"] = re.compile(
                r'```json\n(?P<json>[\s\S]+?)\n```',  
                re.IGNORECASE
            )

        elif task_name == "scoring":
            self.patterns["score_answer"] = re.compile(
                r'(?:```json\n|##solution##\s*)(?P<json>(?:\{.*\}|\[.*\]))(?=\s*```|\s*$)',
                re.IGNORECASE | re.DOTALL
            )

    def extract(self, text: str) -> Union[str, dict, None]:
        text = self.think_pattern.sub('', text).strip()
        print(f"text before extraction{text}")
        if not isinstance(text, str) or not text.strip():
            return None

        elif self.task_name == "multiple_choice":
            if "option" in self.patterns:
                option_matches = self.patterns["option"].findall(text)
                if option_matches:
                    return option_matches[-1].upper()
            return None

        elif self.task_name == "extracted_entries":
            if "entries" in self.patterns:
                match = self.patterns["entries"].search(text)
                if match:
                    json_str = match.group("json").strip()
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
            return None

        elif self.task_name == "scoring":
            if "score_answer" in self.patterns:
                match = self.patterns["score_answer"].search(text)
                if match:
                    json_str = match.group("json").strip()
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
            return None
            
        return None   

class Evaluator:
    def __init__(self, api_config: Dict[str, Any]):
        self.api_config = api_config  
        
    def evaluate(self, task_type: str, generated: List, ground_truth: Optional[List] = None) -> Dict:
        def filter_none_pairs(g_list, t_list):
            """返回有效数据对和过滤统计"""
            valid_pairs = []
            filtered = 0
            for g, t in zip(g_list, t_list):
                if g is None or t is None:
                    filtered += 1
                    continue
                valid_pairs.append((g, t))
            return valid_pairs, filtered

        if task_type == "multiple_choice":
            correct = sum(g.upper() in t.upper() for g, t in zip(generated, ground_truth))
            return {"accuracy": correct/len(generated)}
            
        elif task_type == "extracted_entries":
            valid_pairs, filtered = filter_none_pairs(generated, ground_truth)
            if not valid_pairs:
                return {"hit_rate": 0, "filtered": filtered}
            
            results = []
            for g, t in valid_pairs:
                
                if isinstance(g, str):
                    try:
                        entries = json.loads(g)
                    except:
                        entries = []
                elif isinstance(g, list):
                    entries = g
                else:
                    entries = []
                
                hit = t.strip() in entries[:10]
                results.append({
                    "golds_label": t,
                    "model_selected": entries[:10],
                    "hit": hit
                })
            
            hit_rate = pd.DataFrame(results)["hit"].mean()
            return {"results": results,
                    "hit_rate" : hit_rate
                   }
            
        elif task_type == "scoring":
            analysis_scores = []
            technical_scores = []
            filtered = 0
            
            valid_pairs, none_filtered = filter_none_pairs(generated, ground_truth)
            filtered += none_filtered
            
            for generated_text, gt_text in valid_pairs:
                prompt = f"""### 分析维度评估
                逐项比对黄金标准材料中的要素：
                1. 定位生成方案中对应每个分析要点的段落
                2. 按以下标准判定：
                   - 覆盖完备性（完全覆盖/部分覆盖/未覆盖）
                   - 措施正确性（完全正确/部分正确/错误）
                   - 细节颗粒度（参数级/操作级/概念级）
                
                评分矩阵：
                | 覆盖度 | 正确性 | 颗粒度 | 得分 |
                |--------|--------|--------|------|
                | 未覆盖 | -      | -      | 0    |
                | 部分覆盖 | 错误 | -      | 30   |
                | 部分覆盖 | 部分正确 | 概念级 | 50   |
                | 完全覆盖 | 正确 | 操作级 | 80   |
                | 完全覆盖 | 正确 | 参数级 | 100  |
                
                ### 技术维度评估
                验证技术要素实现：
                1. 识别方案中技术组件的应用
                2. 对照技术规范评估：
                   - 技术栈完整性（完整/部分/缺失）
                   - 参数准确性（精确/合理/错误）
                   - 逻辑连贯性（完全一致/部分一致/矛盾）
                
                评分矩阵：
                | 完整性 | 准确性 | 逻辑性 | 得分 |
                |--------|--------|--------|------|
                | 缺失   | -      | -      | 0    |
                | 部分   | 错误   | 矛盾   | 30   |
                | 部分   | 合理   | 部分   | 60   |
                | 完整   | 精确   | 完全   | 100  |
                
                # 输出规范
                【生成方案】
                {generated_text["solution"]}
                
                【黄金参考材料】
                {gt_text}
                
                请严格按照评分矩阵和输出格式要求生成评估报告，最后三行必须包含##Final Scores##和JSON分数。
                
                !! 重要注意事项 !!
                1. 禁止修改JSON键名(Analysis/Technical Score)
                2. 最后三行必须严格保持以下结构：
                ##Final Scores##
                {{"Analysis Score": X, "Technical Score": Y}}
                3. 其中，分数X,Y必须是大于0，小于100的数字"""
    
                try:
                    model_name = self.api_config.get("model_name", "gpt-4")
                    url = self.api_config.get("api_url")
                    api_key = self.api_config.get("api_key", "")
                    response = get_360api_response(prompt, model_name, url, api_key)
                    print(f"response{response}")
                    content = response['choices'][0]['message']['content']
    
                    score_match = re.search(
                        r'##Final Scores##\s*({.*?})\s*$',
                        content,
                        re.DOTALL
                    )
    
                    if not score_match:
                        filtered += 1
                        continue
                        
                    scores = json.loads(score_match.group(1))
                    a_score = scores.get("Analysis Score", 0)
                    t_score = scores.get("Technical Score", 0)

                    if not (0 <= a_score <= 100) or not (0 <= t_score <= 100):
                        filtered += 1
                        continue
                        
                    if abs(a_score - t_score) > 30:
                        filtered += 1
                        continue

                    analysis_scores.append(a_score)
                    technical_scores.append(t_score)
    
                except Exception as e:
                    print(f"API调用失败: {str(e)}")
                    analysis_scores.append(0)
                    technical_scores.append(0)
                # print(analysis_scores)
            avg_analysis = sum(analysis_scores) / len(analysis_scores) if analysis_scores else 0
            avg_technical = sum(technical_scores) / len(technical_scores) if technical_scores else 0
            # print(f'analysis score{avg_analysis}')
            
            return {
                "Analysis Score Average": round(avg_analysis, 2),
                "Technical Score Average": round(avg_technical, 2)
            }

# ======================= 输出处理 =======================
class TaskOutputWriter:
    @staticmethod
    def save_single_task(task_data: dict, model_name: str, output_dir: str):
        """即时写入单个任务数据到对应文件"""
        task_type = task_data["task_type"]
        timestamp = datetime.now().isoformat(timespec='seconds')  

        output_data = {
                "metadata": {
                    "model": model_name,
                    "data_source": task_data["data_stats"]["source"],
                    "timestamp": timestamp
                },
                "config_params": task_data["config_params"],
                "metrics": TaskOutputWriter._convert_to_json_safe(task_data["metrics"]),
                "original_responses": TaskOutputWriter._convert_to_json_safe(task_data["original_responses"]),
                "extracted_answers": TaskOutputWriter._convert_to_json_safe(task_data["extracted_answers"]),
                "ground_truth": TaskOutputWriter._convert_to_json_safe(task_data["ground_truth"])
            }
        
        filename = f"{output_dir}/{os.path.basename(model_name)}_{task_type}_{timestamp}.jsonl"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
        except Exception as e:
            print(e)
            
    @staticmethod
    def _convert_to_json_safe(data):
        if isinstance(data, (str, int, float, bool, type(None))):
            return data
        elif isinstance(data, pd.Series):
            return {"__series__": data.to_dict()}
        elif isinstance(data, pd.DataFrame):
            return {"__dataframe__": data.to_dict(orient="records")}
        elif isinstance(data, dict):
            return {k: TaskOutputWriter._convert_to_json_safe(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [TaskOutputWriter._convert_to_json_safe(item) for item in data]
        elif isinstance(data, datetime):
            return data.isoformat()
        else:
            try:
                return str(data)
            except:
                return "[Unserializable Object]"                    
# ======================= 工具函数 =======================
def get_360api_response(query: str, model_name: str, api_url: str, key: str = "", max_retry_num: int = 3):
    payload = json.dumps({
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 1000,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
        "user": "andy"
    })

    if not key:
        headers = {
            'Content-Type': 'application/json'
        }
    else:
        headers = {
            'Authorization': key,
            'Content-Type': 'application/json'
        }

    for attempt in range(max_retry_num):
        try:
            res = requests.request("POST", api_url, headers=headers, data=payload)
            res.raise_for_status()
            res = json.loads(res.content)
            # print(f'res{res}')
            return res
            
        except Exception as e:
            print(e)
            if attempt == max_retry_num:
                return {"content": "", "query": query, "error": "max retry"}

def create_data_reader(config: Dict) -> DataReader:
    readers = {
        "excel": ExcelDataReader,
        "csv": CSVDataReader,
        "json": JSONDataReader,
        "jsonl": JSONLinesDataReader
    }
    return readers[config.get("type", "excel")](config)

def load_config(path: str) -> Dict:
    with open(path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / 1024 / 1024, 1)

def plot_grouped_barchart(model_names, metric_labels, data_matrix, 
                         figsize=(14,7), colors=None):

    n_models = len(model_names)
    n_metrics = len(metric_labels)
    bar_width = 0.8 / n_metrics  
    
    plt.figure(figsize=figsize)
    index = np.arange(n_models)

    if not colors:
        colors = sns.color_palette("husl", n_metrics)

    for i in range(n_metrics):
        values = [row[i] for row in data_matrix]
        plt.bar(index + i*bar_width, values, bar_width,
                label=metric_labels[i],
                color=colors[i])

    plt.xticks(index + bar_width*(n_metrics-1)/2, model_names, rotation=0, ha='right')
    plt.ylabel("Score", fontsize=12)
    plt.title("Model Performance Metrics Comparison", pad=20)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
    plt.savefig(f'Model_Performance_Metrics_Comparison_{timestamp}.png')

def prepare_bar_data(metrics_data):
    model_names = [os.path.basename(item["model"]) for item in metrics_data]
    bar_labels = ["Accuracy", "Hit Rate", "Analysis", "Technical"]
    
    # 初始化二维数组：模型数量 x 4个指标
    metrics_matrix = []
    for data in metrics_data:
        model_metrics = data.get("metrics", {})

        mc_dict = model_metrics.get("multiple_choice", {"accuracy": 0})
        mc = mc_dict.get("accuracy", 0)  

        ee = model_metrics.get("extracted_entries", {}).get("hit_rate", 0)

        scoring_dict = model_metrics.get("scoring", {})
        sc_ana = scoring_dict.get("analysis", 0)
        sc_tech = scoring_dict.get("technical", 0)
        metrics_matrix.append([mc, ee, sc_ana, sc_tech])
    
    return model_names, bar_labels, metrics_matrix 
# ======================= 主流程 =======================
def main(config_path: str = "config.yaml"):
    config = load_config(config_path)
    results = []
    metrics_data = []
    # 按任务类型组织配置
    task_configs = {
        "multiple_choice": {
            "data": config["data"].get("multiple_choice",""),
            "prompt": config["prompt"].get("multiple_choice",""),
            "eval_params": config["evaluation"].get("multiple_choice","")
        },
        "extracted_entries": {
            "data": config["data"].get("extracted_entries", ""),
            "prompt": config["prompt"].get("extracted_entries", ""),
            "eval_params": config["evaluation"].get("extracted_entries", "")
        },
        "scoring": {
            "data": config["data"].get("scoring", ""),
            "prompt": config["prompt"].get("scoring", ""),
            "eval_params": config["evaluation"].get("scoring", "")
        }
    }

    evaluator = Evaluator(config.get("evaluation_api", {}))
    for model_config in config["model"]:
        try:
            generator_type = model_config.get("type", "local")
        
            if generator_type == "local":
                llm = LocalLLMGenerator(model_config)
            elif generator_type == "api":
                llm = APIGenerator(model_config)
            else:
                raise ValueError(f"不支持的生成器类型: {generator_type}")
            # llm = LLMGenerator(model_config)
            model_results = {"model_name": llm.model_name, "tasks": [], "metrics":{"multiple_choice": {
        },
        "extracted_entries": {
        },
        "scoring": {
        }}}
    
            for task_name, task_cfg in task_configs.items():
                try:
                    if not task_cfg["data"]:
                        continue
                    data_reader = create_data_reader(task_cfg["data"])
                    data = data_reader.read_data()
                    # 生成任务专属Prompt
                    prompt_gen = PromptGenerator(
                        task_cfg["prompt"],
                        data
                    )
                    
                    # print(prompt_gen)
                    
                    prompts = prompt_gen.generate_prompts()
                    # 模型生成响应
                    #逐条
                    responses = llm.generate_text(
                        prompts=prompts,
                        params=config.get("generation_params", {})
                    )
                    
                    answer_column = task_cfg.get("eval_params")["answer_column"]
                    ground_truth = data.get(answer_column)
                    # 提取结构化结果
                    generated_answers = [AnswerExtractor(task_name).extract(r) for r in responses]
                    
                    # 执行评估
                    #逐条
                    metrics = evaluator.evaluate(
                        task_name,
                        generated_answers,
                        ground_truth
                    )

                    task_entry = {
                    "input": data,
                    "task_type": task_name,
                    "data_stats": {
                    "source": task_cfg["data"]["file_path"]
                    },
                    "config_params": config.get("generation_params", {}),
                    "metrics": metrics,
                    "original_responses": responses,
                    "extracted_answers": generated_answers,
                    "ground_truth": ground_truth
                    }
                    # print(task_entry)

                    if task_name == "multiple_choice":
                        model_results["metrics"][task_name] = {
                            "accuracy": metrics.get("accuracy", 0) * 100
                        }
                    elif task_name == "extracted_entries":
                        model_results["metrics"][task_name] = {
                            "hit_rate": metrics.get("hit_rate", 0) * 100
                        }
                    elif task_name == "scoring":
                        model_results["metrics"][task_name] = {
                            "analysis": metrics.get("Analysis Score Average", 0),
                            "technical": metrics.get("Technical Score Average", 0)
                        }
                        
                    model_results["tasks"].append(task_entry)
                    TaskOutputWriter.save_single_task(task_entry, model_results["model_name"], config["output"]["output_dir"])
                except Exception as task_error:
                    print(f"[{task_name}] ERROR: {str(task_error)}")
                    continue
                
            metrics_data.append({
            "model": model_results["model_name"],
            "metrics": model_results["metrics"]
        })
                
            results.append(model_results)
            del llm
            gc.collect()
            # print(results)
        except Exception as model_error:
            print(f"Model {model_config['name']} Failed: {str(model_error)}")
            continue
    print(metrics_data)
    model_names, labels, data = prepare_bar_data(metrics_data)
    plot_grouped_barchart(model_names, labels, data)

if __name__ == "__main__":
    main()
