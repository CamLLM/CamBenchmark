import os

os.environ['CUDA_VISIBLE_DEVICES'] = ""
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
import json
import re
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba

# 模型地址
model_name = ""
excel_file_path = "bitextmine.xlsx"
sheet_name = "Sheet1"
result_dir = "result"
eval_dir = "result/eval"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, gpu_memory_utilization=0.95, tensor_parallel_size=8)
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)
total_bleu = 0.0
correct_number = 0

df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
extracted_list = []
gold_list = []
for index, row in df.iterrows():
    prompt = (
        "请将以下英文翻译成中文，保持原意，输出翻译后的中文，不要添加任何内容。\n"
        f"英文：{row['en']}\n"
    )
    print(prompt)
    messages = [
        {"role": "system", "content": "你是一个非常擅长民航领域翻译的专家"},
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
    response = output[0].outputs[0].text.strip()
    print(response)
    think_pattern = re.compile(r'.*?</think>', re.DOTALL)
    extracted = think_pattern.sub('', response).strip()
    gold = row['ch']
    extracted_list.append(extracted)
    gold_list.append(row['ch'])


    smoothie = SmoothingFunction().method4
    extracted_tokens = list(jieba.cut(extracted))
    gold_tokens = list(jieba.cut(gold))
    bleu_score = sentence_bleu([gold_tokens], extracted_tokens, smoothing_function=smoothie)
    total_bleu += bleu_score
    if bleu_score > 0.05:
        correct_number += 1
        
    json_record = {
        "input": row.to_dict(),
        "response": response,
        "extracted": extracted,
        "gold": row['ch'],
        "sampling_params": {"temperature": sampling_params.temperature,
                            "max_tokens": sampling_params.max_tokens,
                            "repetition_penalty": sampling_params.repetition_penalty,
                            "seed": sampling_params.seed},
        "model": os.path.basename(model_name),
        "is_correct": bleu_score > 0.05
        }
    result_output_filename = f"{result_dir}/{os.path.basename(model_name)}_translate_{sheet_name}_think.jsonl"
    with open(result_output_filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(json_record, ensure_ascii=False) + "\n")
    
accuracy = correct_number / len(extracted_list)
print(f"Ave BLEU Score: {total_bleu / len(extracted_list)}")
print(f"Accuracy: {accuracy * 100:.2f}%")
metrics_result = {
    "accuracy": accuracy,
    "ave_bleu_score":  total_bleu / len(extracted_list)
}
eval_output_filename = f"{eval_dir}/{os.path.basename(model_name)}_translate_{sheet_name}_metrics.jsonl"
with open (eval_output_filename, "a", encoding="utf-8") as f1:
    f1.write(json.dumps(metrics_result, ensure_ascii=False) + "\n")
    