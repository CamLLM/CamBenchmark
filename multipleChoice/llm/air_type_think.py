import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
import json
from datetime import datetime
from utils.extract_answer import AnswerExtractor
from utils.evaluator import Evaluator

# 模型地址
model_name = ""
excel_file_path = "air_choice.xlsx"
result_dir = ""
sheet_names = ["Sheet1"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, tensor_parallel_size=2)
# sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=32768, presence_penalty=1.5)
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)
for sheet_name in sheet_names:
    eval_output_filename = f"{result_dir}/{os.path.basename(model_name)}_multiple_choice_{sheet_name}.jsonl"
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    generated_answers = []
    ground_truths = []
    for index, row in df.iterrows():
        if row['problem'] == "二选一":
            prompt = (
                f"下面是一道关于民航维修（机型:{row['flag']}）的选择题：\n"
                f"题目：{row['试题题目']}\n"
                f"A. {row['选项A']}\n"
                f"B. {row['选项B']}\n"
                f"请认真思考并作答，最终只需输出正确答案对应的选项即可。"
            )
        elif row['problem'] == "三选一":
            prompt = (
                f"下面是一道关于民航维修（机型:{row['flag']}）的选择题：\n"
                f"题目：{row['试题题目']}\n"
                f"A. {row['选项A']}\n"
                f"B. {row['选项B']}\n"
                f"C. {row['选项C']}\n"
                f"请认真思考并作答，最终只需输出正确答案对应的选项即可。"
            )
        else:
            prompt = (
                f"下面是一道关于民航维修（机型:{row['flag']}）的选择题：\n"
                f"题目：{row['试题题目']}\n"
                f"A. {row['选项A']}\n"
                f"B. {row['选项B']}\n"
                f"C. {row['选项C']}\n"
                f"D. {row['选项D']}\n"
                f"请认真思考并作答，最终只需输出正确答案对应的选项即可。"
            )
        print(prompt)
        messages = [
            {"role": "system", "content": "你是一个非常擅长民航维修的专家"},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        output = llm.generate([formatted_prompt], sampling_params)
        response = output[0].outputs[0].text.strip()
        print(response)
        extracted = AnswerExtractor("multiple_choice").extract(response)
        # df.at[index, 'response'] = response
        # df.at[index, 'extracted'] = extracted
        # with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        #     df.to_excel(writer, sheet_name=sheet_name, index=False)
        json_record = {
            "input": row.to_dict(),
            row['uid']:{
            "response": response,
            "extracted": extracted
            },
            "ground_truth": row['答案'],
            "sampling_params": {"temperature": sampling_params.temperature,
                                "max_tokens": sampling_params.max_tokens,
                                "repetition_penalty": sampling_params.repetition_penalty,
                                "seed": sampling_params.seed},
            "model": os.path.basename(model_name),
            "enable_thinking": True,
            "flag": row['flag']
        }
        ground_truths.append(row['答案'])
        generated_answers.append(extracted)
        print(json_record)
        with open(eval_output_filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_record, ensure_ascii=False) + "\n")

    # evaluator = Evaluator(api_config={})
    # metrics = evaluator.evaluate("multiple_choice", generated_answers, ground_truths)
    # eval_record = {
    #     "model": model_name,
    #     "task": "multiple_choice",
    #     "metrics": metrics,
    #     "generated_answers": generated_answers,
    #     "ground_truths": ground_truths,
    #     "timestamp": datetime.now().isoformat()
    # }
    # eval_filename = f"{eval_dir}/{os.path.basename(model_name)}_multiple_choice_{sheet_name}_metrics.jsonl"
    # with open(eval_filename, 'a', encoding='utf-8') as f1:
    #     f1.write(json.dumps(eval_record, ensure_ascii=False))

print("Processing complete.")
