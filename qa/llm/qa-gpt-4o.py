import json
import re
import os
import requests
# 访问gpt-4o服务地址
url = ""
# 待评估模型预测生成的结果文件
input_file = 'result/Qwen3-235B-A22B-Thinking-2507_qa_Sheet1.jsonl'
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        extracted = item['extracted']
        prompt = (
        f"### 任务说明 ###\n"
        f"请根据问题执行以下任务：\n"
        f"检查答案1与答案2的关系：\n"
        f"   - 若答案1中基本包含答案2的核心内容，输出2\n"
        f"   - 若答案1包含答案2的部分信息，输出1"
        f"   - 若答案1未包含答案2中的任何信息， 输出 0\n"
        f"答案1：{extracted}\n"
        f"答案2：{item['ground_truth']}\n"
        f"注意：输出中不要包含任何额外解释或空格，仅输出数字。"
        )
        payload = json.dumps({
        "model": "gpt-4o",
        "messages" : [
            {"role": "system", "content": "你是一个非常擅长民航维修的专家"},
            {"role": "user", "content": prompt}
            # {"role": "user", "content": "你好"}
        ],
        "stream": False,
        "max_completion_tokens": 2048

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
        think_pattern = re.compile(r'<judgment>.*?</judgment>', re.DOTALL)
        extracted = think_pattern.sub('', res).strip()
        json_record = {
            "input": item['input'],
            "response": res,
            "extracted": extracted,
            "ground_truth": item['ground_truth'],
            "model": "gpt-4o"
        }
        with open(f"result/eval/qwen3-235b-think-2507.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(json_record, ensure_ascii=False) + "\n")
            
        
