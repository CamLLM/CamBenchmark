import re
import json
from typing import Union
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