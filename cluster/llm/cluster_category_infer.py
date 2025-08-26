import os

os.environ['CUDA_VISIBLE_DEVICES'] = ""
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
import json
import re
from datetime import datetime

# 模型地址
model_name = ""
excel_file_path = "cluster.xlsx"
sheet_name = "Sheet1"
result_dir = "result_cluster_category"
eval_dir = "result_cluster_category/eval"

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, tensor_parallel_size=8, gpu_memory_utilization=0.95)
sampling_params=SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=32768, presence_penalty=1.5)

df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
result_output_filename = f"{result_dir}/{os.path.basename(model_name)}_cluster_category_{sheet_name}.jsonl"
generated_answers = []
ground_truths = []
primary_extracted_list = ['空调系统', '操纵系统', '电源系统', '防冰和防雨','防火系统','控制系统','燃油系统','液压系统','通讯系统','设备装饰']

def filter_none_pairs(g_list, t_list):
    """返回有效数据对和过滤统计"""
    filtered_pairs = []
    none_count = 0
    for g, t in zip(g_list, t_list):
        if g is None or t is None:
            none_count += 1
            continue
        filtered_pairs.append((g, t))
    return filtered_pairs, none_count

for _, row in df.iterrows():
    prompt = (
        "请根据以下描述定位相应的章节，必须从所给的备选章节列表中选择1个，不要回答其他内容。"
        f"描述：{row['文本']}\n"
        f"备选章节列表{primary_extracted_list}"
    )
    print(prompt)
    messages = [
        {"role": "system", "content": "你是一个非常擅长民航维修的专家"},
        {"role": "user", "content": prompt}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        # enable_thinking=False,
        enable_thinking=True,
    )

    output = llm.generate([formatted_prompt], sampling_params)
    # output = llm.chat(messages, sampling_params)
    response = output[0].outputs[0].text.strip()
    print(response)
    think_pattern = re.compile(r'.*?</think>', re.DOTALL)
    extracted_primary = think_pattern.sub('', response).strip()
    print(extracted_primary)

    json_record = {
    "input": row.to_dict(),
    "response": response,
    "extracted_primary": extracted_primary,
    "primary_gold": row['类别'],
    "sampling_params": {"temperature": sampling_params.temperature,
                        "max_tokens": sampling_params.max_tokens,
                        "repetition_penalty": sampling_params.repetition_penalty,
                        "seed": sampling_params.seed},
    "model": os.path.basename(model_name)
}
    with open(result_output_filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(json_record, ensure_ascii=False) + "\n")
    generated_answers.append(extracted_primary)
    ground_truths.append(row['类别'])

valid_pairs, filtered_count = filter_none_pairs(generated_answers, ground_truths)
correct = sum(t in g for g, t in valid_pairs)
metrics_primary = correct / len(valid_pairs)
eval_record = {
    "model": model_name,
    "task": "system_categories",
    "metrics_primary": metrics_primary,
    "primary_extracted_list": generated_answers,
    "primary_gold_list": ground_truths,
    "none_count": filtered_count,
    "timestamp": datetime.now().isoformat()
}
eval_filename = f"{eval_dir}/{os.path.basename(model_name)}_clustering_category_{sheet_name}_metrics.jsonl"
with open(eval_filename, 'a', encoding='utf-8') as f:
    f.write(json.dumps(eval_record, ensure_ascii=False))