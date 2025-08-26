import json
import re
import os
import requests
# 访问gpt-4o服务地址
url = ""
input_file = 'temperature_09/Qwen3-235B-Instruct_fault_tree_.jsonl'
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        extracted = item['extracted']
        gold_list_str = '\n'.join([f" - {gold_item}" for gold_item in item['gold_list']])
        prompt = (
        f"### 任务说明 ###\n"
        f"请根据问题执行以下两个任务：\n"
        f"1. **内容覆盖判断**：评估'答案1'是否**基本涵盖**了'答案2'列表中的核心内容。\n"
        f"   - 如果是，输出 1\n"
        f"   - 否则，输出 0\n"
        f"2. **具体覆盖点识别**：找出'答案1'中涵盖'答案2'的**具体条目序号**。\n"
        f"   - 如果没有，则输出 '无'。\n\n"
        # f"### 问题 ###\n"
        # f"{item['prompt']}\n"
        f"### 待评估答案 ###\n"
        f"答案1：{extracted}\n"
        f"答案2（参考答案列表）：{gold_list_str}\n\n"
        f"### 输出格式要求 ###\n"
        f"输出必须严格按以下格式：\n"
        f"第一行：'1或0\n"
        f"第二行：''答案1'中涵盖'答案2'的具体条目序号，用,隔开（例如 '2,3,5'）或 `无`\n"
        f"注意：输出中不要包含任何额外解释或空格，仅输出两行文本。"
        )
        payload = json.dumps({
        "model": "gpt-4o",
        "messages" : [
            {"role": "system", "content": "你是一个非常擅长民航维修的专家"},
            {"role": "user", "content": prompt}
            # {"role": "user", "content": "你好"}
        ],
        "stream": False,
        "max_completion_tokens": 512
        })
        headers = {
          'Authorization': '',
          'Content-Type': 'application/json'
        }
        while True:
            try:
                response = requests.request("POST", url, headers=headers, data=payload).text
                response = json.loads(response)
                print(f"response{response}\n")
                print(f"type{type(response)}")
                # response = json.loads(response)
                res = response['choices'][0]['message']['content']
                print(res)
                break
            except Exception as e:
                print(e)
        # think_pattern = re.compile(r'<judgment>.*?</judgment>', re.DOTALL)
        # extracted = think_pattern.sub('', res).strip()
        json_record = {
            "prompt": item['prompt'],
            "response": res,
            "extracted": extracted,
            "ground_truth": gold_list_str,
            "model": "gpt-4o"
        }
        with open(f"eval/temperature09/qwen3-235B-thinking-2507-no_fault_tree.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(json_record, ensure_ascii=False) + "\n")
            
        
