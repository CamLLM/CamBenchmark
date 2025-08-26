import os 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
import json
import re
from datetime import datetime

# 模型地址
model_name = ""
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, tensor_parallel_size=8, gpu_memory_utilization=0.95)
df = pd.read_excel("paircls.xlsx")
for index, row in df.iterrows():
    question = row['question']
    description = row['description']
    gold = row['label']
    prompt = (
        f"条目/故障描述：{question}\n"
        f"文本：{description}\n"
        "请判断文本后跟内容是否与条目/故障描述匹配，匹配回答1，不匹配回答0。请认真思考并作答，答案只需要输出0或1"
    )
    print(prompt)

    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768, repetition_penalty=1.0)
    messages =  [
            {"role": "system", "content": "你是一个非常擅长民航维修的专家"},
            {"role": "user", "content": prompt}
        ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    output = llm.generate([formatted_prompt], sampling_params)
    response = output[0].outputs[0].text.strip()
    print(response)
    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    think_match = think_pattern.sub('', response)
    match = re.search(r'\b(0|1)\b', think_match)
    extracted = match.group(1)
    print(f"extracted{extracted}")
    # extracted_answer = extracted.group(0) if extracted else ''
    json_record = {
        "input": row.to_dict(),
        "response": response,
        "extracted": extracted,
        "ground_truth": gold,
        "sampling_params": {
            "temperature": sampling_params.temperature,
            "max_tokens": sampling_params.max_tokens,
            "repetition_penalty": sampling_params.repetition_penalty,
            "seed": sampling_params.seed
        },
        "model": os.path.basename(model_name)
    }
    with open(f"result/paircls_think_{os.path.basename(model_name)}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(json_record, ensure_ascii=False) + "\n")

