import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
import json
import re
from datetime import datetime

# 模型地址
model_name = ""

excel_file_path = "classification.xlsx"
sheet_name = ""
result_dir = "../../../result/result_system_category"
eval_dir = "../../../result/result_system_category/eval"

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, tensor_parallel_size=2)
sampling_params = SamplingParams(temperature=0, max_tokens=8192, repetition_penalty=1.0, seed=42)

df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
generated_answers = []
ground_truths = []
primary_extracted_list = []
secondary_extracted_list = []
primary_gold_list = []
secondary_gold_list = []

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

with open(system_category_path, "r") as f:
    all_categories = json.load(f)

primary_list = list(all_categories.keys())

for _, row in df.iterrows():
    prompt = (
        "请根据以下机型和描述定位故障所在的一级系统，必须从所给的备选一级系统列表中选择1个，不要回答其他内容。"
        f"描述：{row['描述']}\n"
        f"备选一级系统列表{primary_list}"
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
        enable_thinking=False,
        # enable_thinking=True,
    )

    output = llm.generate([formatted_prompt], sampling_params)
    # output = llm.chat(messages, sampling_params)
    response = output[0].outputs[0].text.strip()
    print(response)
    think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
    extracted_primary = think_pattern.sub('', response).strip()
    print(extracted_primary)

    json_record = {
        "input": row.to_dict(),
        "response": response,
        "extracted_primary": extracted_primary,
        "primary_gold": row['一级系统'],
        "sampling_params": {"temperature": sampling_params.temperature,
                            "max_tokens": sampling_params.max_tokens,
                            "repetition_penalty": sampling_params.repetition_penalty,
                            "seed": sampling_params.seed},
        "model": os.path.basename(model_name)
    }
    with open(result_output_filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(json_record, ensure_ascii=False) + "\n")
    primary_gold_list.append(row['一级系统'])
    primary_extracted_list.append(extracted_primary)

valid_pairs, filtered_count = filter_none_pairs(primary_extracted_list, primary_gold_list)
correct = sum(t in g for g, t in valid_pairs)
metrics_primary = correct / len(valid_pairs)

eval_record = {
    "model": model_name,
    "task": "system_categories",
    "metrics_primary": metrics_primary,
    "primary_extracted_list": primary_extracted_list,
    "primary_gold_list": primary_gold_list,
    "none_count": filtered_count,
    "timestamp": datetime.now().isoformat()
}
eval_filename = f"{eval_dir}/{os.path.basename(model_name)}_system_category_{sheet_name}_metrics.jsonl"
with open(eval_filename, 'a', encoding='utf-8') as f:
    f.write(json.dumps(eval_record, ensure_ascii=False))

