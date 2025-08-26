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
import hashlib
import argparse
from collections import defaultdict
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
        self.task_stats = {}
 
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
                
    @staticmethod
    def save_single_record(task_data: dict, model_name: str, output_dir: str, index: int, data_source: dict):
        """单条记录追加写入"""
        if data_source.get("type") == "excel":
            sheet_part = data_source.get("sheet_name", "unknown_sheet")
        else:
            file_path = data_source.get("file_path", "unknown_file")
            sheet_part = os.path.splitext(os.path.basename(file_path))[0]  # 去除扩展名
        
        filename = f"{output_dir}/{os.path.basename(model_name)}_{task_data['task_type']}_{sheet_part}.jsonl"   
        task_data["index"] = index
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps(task_data, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"记录写入失败: {str(e)}") 
            
# ======================= 进度追踪 =======================
class ProgressTracker:
    @staticmethod
    def get_progress(model_name: str, task_type: str, data_source: dict) -> tuple[int, list, list]:
        """返回(进度索引, 已生成的答案列表, 对应的真实答案列表)"""
        unique_id = hashlib.md5(json.dumps(data_source, sort_keys=True).encode()).hexdigest()[:8]
        progress_file = f"progress/{model_name}_{task_type}_{unique_id}.progress"
        
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    data = json.load(f)
                    return data["index"], data.get("generated", []), data.get("truths", [])
            except:
                return 0, [], []
        return 0, [], []

    @staticmethod
    def update_progress(model_name: str, task_type: str, data_source: dict, 
                      index: int, generated: list, truths: list):
        """更新进度时同时保存评估数据"""
        unique_id = hashlib.md5(json.dumps(data_source, sort_keys=True).encode()).hexdigest()[:8]
        progress_file = f"progress/{model_name}_{task_type}_{unique_id}.progress"
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        
        # 使用临时文件确保写入原子性
        temp_file = f"{progress_file}.tmp"
        with open(temp_file, 'w') as f:
            json.dump({
                "index": index + 1,  # 存储下一个要处理的索引
                "generated": generated,
                "truths": truths
            }, f)
        os.replace(temp_file, progress_file)

    @staticmethod
    def clean_progress(model_name: str, task_type: str, data_source: dict):
        unique_id = hashlib.md5(json.dumps(data_source, sort_keys=True).encode()).hexdigest()[:8]
        progress_file = f"progress/{model_name}_{task_type}_{unique_id}.progress"
        if os.path.exists(progress_file):
            os.remove(progress_file)
            
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

def prepare_bar_data(metrics_data):
    """准备绘图数据"""
    # 按模型和任务聚合指标
    model_stats = defaultdict(lambda: {
        "multiple_choice": [],
        "extracted_entries": [],
        "scoring_analysis": [],
        "scoring_technical": []
    })
    
    for record in metrics_data:
        model = record["model"]
        task = record["task"]
        metrics = record["metrics"]
        
        if task == "multiple_choice":
            model_stats[model]["multiple_choice"].append(metrics.get("accuracy", 0))
        elif task == "extracted_entries":
            model_stats[model]["extracted_entries"].append(metrics.get("hit_rate", 0))
        elif task == "scoring":
            model_stats[model]["scoring_analysis"].append(metrics.get("Analysis Score Average", 0))
            model_stats[model]["scoring_technical"].append(metrics.get("Technical Score Average", 0))
    
    # 计算各模型平均指标
    model_names = []
    metrics_matrix = []
    bar_labels = [
        "Accuracy", 
        "Hit Rate", 
        "Analysis", 
        "Technical"
    ]
    
    for model_name, stats in model_stats.items():
        mc_avg = np.mean(stats["multiple_choice"]) * 100 if stats["multiple_choice"] else 0
        ee_avg = np.mean(stats["extracted_entries"]) * 100 if stats["extracted_entries"] else 0
        sc_ana = np.mean(stats["scoring_analysis"]) if stats["scoring_analysis"] else 0
        sc_tech = np.mean(stats["scoring_technical"]) if stats["scoring_technical"] else 0
        
        model_names.append(model_name)
        metrics_matrix.append([mc_avg, ee_avg, sc_ana, sc_tech])
    
    return model_names, bar_labels, metrics_matrix

def plot_grouped_barchart(model_names, metric_labels, data_matrix):
    """绘制分组柱状图"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid", font_scale=1.1)
    
    # 设置柱状图参数
    bar_width = 0.2
    index = np.arange(len(model_names))
    colors = sns.color_palette("husl", len(metric_labels))
    
    # 绘制每组柱状图
    for i, (label, color) in enumerate(zip(metric_labels, colors)):
        values = [row[i] for row in data_matrix]
        plt.bar(index + i*bar_width, values, bar_width, 
                label=label, color=color, alpha=0.8)
    
    # 图表装饰
    plt.title("Model Metrics Comparison", pad=20, fontsize=16)
    plt.xlabel("model", fontsize=12)
    plt.xticks(index + bar_width*1.5, model_names, rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 添加数值标签
    for i, model in enumerate(data_matrix):
        for j, value in enumerate(model):
            plt.text(index[i] + j*bar_width, value+1, f'{value:.1f}',
                     ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # 保存并显示
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"model_performance_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", 
                      type=str, 
                      default="config_with_recovery.yaml",
                      help="Path to configuration file")
    return parser.parse_args()

# ======================= 主流程 =======================
def main(config_path: str = "config_with_recovery.yaml"):
    config = load_config(config_path)
    results = []
    metrics_data = []
    
    # 按任务类型组织配置
    task_configs = {
        "multiple_choice": {
            "data": config["data"].get("multiple_choice", []),
            "prompt": config["prompt"].get("multiple_choice", {}),
            "eval_params": config["evaluation"].get("multiple_choice", {})
        },
        "extracted_entries": {
            "data": config["data"].get("extracted_entries", []),
            "prompt": config["prompt"].get("extracted_entries", {}),
            "eval_params": config["evaluation"].get("extracted_entries", {})
        },
        "scoring": {
            "data": config["data"].get("scoring", []),
            "prompt": config["prompt"].get("scoring", {}),
            "eval_params": config["evaluation"].get("scoring", {})
        }
    }

    evaluator = Evaluator(config.get("evaluation_api", {}))
    
    for model_config in config["model"]:
        try:
            # 初始化模型信息
            generator_type = model_config.get("type", "local")
            model_name = os.path.basename(model_config.get("model_name", "unknown_model"))
            
            # 创建生成器
            if generator_type == "local":
                generator = LocalLLMGenerator(model_config)
            elif generator_type == "api":
                generator = APIGenerator(model_config)
            else:
                raise ValueError(f"不支持的生成器类型: {generator_type}")

            # 初始化模型结果结构
            model_results = {
                "model_name": model_name,
                "tasks": [],
                "metrics": {
                    "multiple_choice": {},
                    "extracted_entries": {},
                    "scoring": {}
                }
            }

            # 遍历所有任务类型
            for task_name, task_cfg in task_configs.items():
                if not task_cfg["data"]:
                    continue

                try:
                    # 遍历每个数据源
                    for data_source in task_cfg["data"]:
                        try:
                            # 读取数据
                            data_reader = create_data_reader(data_source)
                            full_data = data_reader.read_data()

                            # 初始化本数据源的评估数据
                            generated_answers = []
                            ground_truths = []
        
                            # 获取进度及已有评估数据
                            start_idx, generated_answers, ground_truths = ProgressTracker.get_progress(
                                model_name, 
                                task_name, 
                                data_source
                            )

                            # 逐条处理数据
                            for idx in range(start_idx, len(full_data)):
                                row = full_data.iloc[idx]
                                try:
                                    # 生成提示
                                    prompt = PromptGenerator(
                                        task_cfg["prompt"], 
                                        pd.DataFrame([row])
                                    ).generate_prompts()[0]
                                    
                                    # 生成响应
                                    response = generator.generate_text(
                                        [prompt], 
                                        config.get("generation_params", {})
                                    )[0]
                                    
                                    # 提取答案
                                    extracted = AnswerExtractor(task_name).extract(response)
                                    ground_truth = row.get(task_cfg["eval_params"]["answer_column"])
                                    
                                    # 构建记录
                                    record = {
                                        "input": row.to_dict(),
                                        "response": response,
                                        "extracted": extracted,
                                        "ground_truth": ground_truth,
                                        "task_type": task_name,
                                        "data_stats": {"source": data_source["file_path"]},
                                        "config_params": config.get("generation_params", {})
                                    }

                                    # 保存记录
                                    TaskOutputWriter.save_single_record(
                                        record, 
                                        model_name,
                                        config["output"]["output_dir"],
                                        idx,
                                        data_source
                                    )
                                    
                                    # 更新评估数据
                                    new_generated = generated_answers + [extracted]
                                    new_truths = ground_truths + [ground_truth]
                                    
                                    # 更新进度（每处理1条保存1次）
                                    ProgressTracker.update_progress(
                                        model_name,
                                        task_name,
                                        data_source,
                                        idx,
                                        new_generated,
                                        new_truths
                                    )

                                    generated_answers = new_generated
                                    ground_truths = new_truths

                                except Exception as e:
                                    print(f"[{model_name}] {task_name} 第{idx}条处理失败: {str(e)}")
                                    continue
                            print(f"g:{generated_answers}")
                            print(f"t:{ground_truths}")
                            # 处理完成后立即评估
                            if generated_answers:
                                metrics = evaluator.evaluate(task_name, generated_answers, ground_truths)
                                print(f"metrics{metrics}")
                                
                                # 保存最终评估结果（原有逻辑）
                                eval_record = {
                                    "model": model_name,
                                    "task": task_name,
                                    "data_source": data_source["file_path"],
                                    "metrics": metrics,
                                    "sample_size": len(generated_answers),
                                    "timestamp": datetime.now().isoformat()
                                }
                                metrics_data.append(eval_record)

                                if data_source.get("type") == "excel":
                                    sheet_part = data_source.get("sheet_name", "unknown_sheet")
                                else:
                                    file_path = data_source.get("file_path", "unknown_file")
                                    sheet_part = os.path.splitext(os.path.basename(file_path))[0]
                                    
                                eval_filename = f"{config['output']['eval_dir']}/{os.path.basename(model_name)}_{task_name}_{sheet_part}.jsonl"
                                with open(eval_filename, 'a', encoding='utf-8') as f:
                                    f.write(json.dumps(eval_record, ensure_ascii=False))
                                    
                                # 清理进度文件
                                ProgressTracker.clean_progress(model_name, task_name, data_source)

                        except Exception as task_error:
                            print(f"[{task_name}] ERROR: {str(task_error)}")
                            continue

                except Exception as task_error:
                    print(f"[{task_name}] ERROR: {str(task_error)}")
                    continue

            # 释放资源
            del generator
            gc.collect()

        except Exception as model_error:
            print(f"Model {model_name} Failed: {str(model_error)}")
            continue

    # 可视化结果
    model_names, labels, data = prepare_bar_data(metrics_data)
    plot_grouped_barchart(model_names, labels, data)

if __name__ == "__main__":
    args = parse_args()
    main(args.config)
