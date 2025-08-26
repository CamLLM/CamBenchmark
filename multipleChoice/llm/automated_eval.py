import jieba
from difflib import SequenceMatcher
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.util import ngrams
import json

class AnswerEvaluator:
    def __init__(self):
        # 设置各层级的阈值参数
        self.inclusion_threshold = 0.7
        self.lcs_thresholds = (0.2, 0.35)
        self.bleu_thresholds = (0.1, 0.2)
        self.token_overlap_thresholds = (0.1, 0.2)

    def evaluate(self, reference, candidate):
        """
        漏斗式评估模型生成答案与标准答案的相似性
        返回评分: 0(不匹配), 1(部分匹配), 2(匹配)
        """
        # 第一层: 标准答案是否包含在模型答案中
        inclusion_score = self._check_inclusion(reference, candidate)

        if inclusion_score == 2:
            return 2

        # 第二层: 最长公共子串方法
        lcs_score = self._longest_common_substring(reference, candidate)

        if lcs_score == 2:
            return 2
        elif lcs_score == 1 and inclusion_score == 0:
            return 1

        # 第三层: BLEU/Rouge方法
        bleu_score = self._calculate_bleu(reference, candidate)

        if bleu_score == 2:
            return 2
        elif bleu_score == 1 and lcs_score == 0 and inclusion_score == 0:
            return 1

        # 第四层: 分词重叠方法
        overlap_score = self._token_overlap(reference, candidate)

        if overlap_score == 2:
            return 2
        elif overlap_score == 1 and bleu_score == 0 and lcs_score == 0 and inclusion_score == 0:
            return 1

        # 如果所有层级都未达到匹配标准
        return -1

    def _check_inclusion(self, gold, response):
        """
        第一层评估: 检查标准答案是否包含在模型答案中
        """
        # 简单包含检查
        if gold in response:
            return 2

        gold_len = len(gold)
        if gold_len == 0:
            return 0

        #SequenceMatcher计算相似度
        max_similarity = 0
        # 滑动窗口检查候选答案中的所有可能子串
        for i in range(len(response) - gold_len + 1):
            substring = response[i:i + gold_len]
            similarity = SequenceMatcher(None, gold, substring).ratio()
            if similarity > max_similarity:
                max_similarity = similarity
                if max_similarity >= self.inclusion_threshold:
                    return 2
        
        print(f"Max Inclusion Similarity: {max_similarity}")
        return 0

    def _longest_common_substring(self, gold, response):
        """
        第二层评估: 最长公共子串方法
        """
        seq_matcher = SequenceMatcher(None, gold, response)
        match = seq_matcher.find_longest_match(0, len(gold), 0, len(response))
        lcs_length = match.size

        if lcs_length == 0:
            return 0

        # 计算最长公共子串相对于标准答案长度的比例
        ratio = lcs_length / len(gold)
        print(f"LCS Length: {lcs_length}, Ratio: {ratio}")
        if ratio >= self.lcs_thresholds[1]:
            return 2
        elif ratio >= self.lcs_thresholds[0]:
            return 1
        else:
            return 0

    def _calculate_bleu(self, gold, response):
        """
        第三层评估: BLEU
        """
        # 分词
        gold_tokens = list(jieba.cut(gold))
        res_tokens = list(jieba.cut(response))

        # 计算BLEU-4分数
        smoothie = SmoothingFunction().method4
        try:
            bleu_score = sentence_bleu([gold_tokens], res_tokens,
                                       weights=(0.25, 0.25, 0.25, 0.25),
                                       smoothing_function=smoothie)
        except:
            bleu_score = 0

        # 根据阈值返回评分
        print(f"BLEU Score: {bleu_score}")
        if bleu_score >= self.bleu_thresholds[1]:
            return 2
        elif bleu_score >= self.bleu_thresholds[0]:
            return 1
        else:
            return 0

    def _token_overlap(self, reference, candidate):
        """
        第四层评估: 分词重叠方法
        """
        # 中文分词
        ref_tokens = set(jieba.cut(reference))
        cand_tokens = set(jieba.cut(candidate))

        if not ref_tokens or not cand_tokens:
            return 0

        # 计算交集和并集
        intersection = ref_tokens & cand_tokens
        union = ref_tokens | cand_tokens

        overlap_ratio = len(intersection) / len(union)
        print(f"Token Overlap Ratio: {overlap_ratio}")
        if overlap_ratio >= self.token_overlap_thresholds[1]:
            return 2
        elif overlap_ratio >= self.token_overlap_thresholds[0]:
            return 1
        else:
            print(f"Token Overlap Ratio: {overlap_ratio}")
            return 0

# 使用示例
if __name__ == "__main__":
    evaluator = AnswerEvaluator()

    with open("../../result/fault_tree_test/test_32b.jsonl", "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]
        test_cases = [("".join(item['gold_list']), item['extracted']) for item in test_data]

    # 测试用例
    # test_cases = [
    #     ("北京是中国的首都", "中国的首都是北京"),  
    #     ("北京是中国的首都", "北京是中国的政治中心"), 
    #     ("北京是中国的首都", "上海是中国最大的城市"),  
    #     ("Python是一种编程语言", "Python是一种高级编程语言"), 
    #     ("人工智能是未来趋势", "AI将主导未来发展"),  
    #     ("""测量两部ADF接收机后的插座上的供电插钉电压""", """两部ADF系统皆故障，而面板上的频率显示窗显示正常，这说明什么？这说明故障点可能不在控制面板本身，因为频率显示正常，说明控制面板的显示部分和基本供电是正常的。但两部ADF接收机都不工作，这可能意味着问题出在供电或信号路径上。\n\n接下来，既然已经测量了从ADF控制面板到两部ADF接收机的电源线路，结果正常，那么下一步应该考虑其他可能影响两部ADF接收机的共用部分。例如，可能的问题点包括：\n\n1. **电源输入问题**：虽然线路正常，但可能115V交流的第二套转换汇流条本身没有供电，或者相关的跳开关（如C1383）没有闭合或损坏。需要检查跳开关的状态和供电情况。\n\n2. **控制面板的输出问题**：虽然频率显示正常，但控制面板可能没有正确地将控制信号或电源输出到接收机。需要进一步检查控制面板的输出信号和电源是否正常。\n\n3. **接收机的公共故障**：如果两部接收机同时出现相同的问题，可能是由于它们共用的某个部件或环境问题，例如接收机的公共电源、接地问题，或者接收机的安装位置导致的信号干扰。\n\n4. **天线系统问题**：虽然天线可能正常，但天线的连接或切换开关可能存在问题，导致两部接收机都无法接收到信号。需要检查天线连接和切换机制。\n\n5. **软件或固件问题**：虽然不太常见，但接收机的软件或固件可能存在故障，导致无法正常工作。可以尝试重置或更新软件。\n\n排故方法：\n\n1. **检查跳开关C1383的状态**：确认跳开关是否闭合，是否有烧毁或接触不良的情况。如果跳开关有问题，更换或修复。\n\n2. **测量第二套转换汇流条的电压**：确认115V交流电源是否正常供应到ADF接收机。如果电压异常，需要检查电源系统。\n\n3. **检查控制面板的输出信号**：使用示波器或万用表测量控制面板输出到接收机的信号和电源是否正常。如果信号异常，可能需要更换控制面板或修复相关电路。\n\n4. **检查天线连接和切换开关**：检查天线的连接是否牢固，切换开关是否正常工作。如果发现问题，进行修复或更换。\n\n5. **尝试重置或更新接收机的软件**：如果怀疑是软件问题，可以尝试重置接收机或更新其固件。\n\n通过以上步骤，可以逐步排查故障点，最终确定并解决两部ADF系统故障的问题。\n\n\n\n\n根据你提供的历史推理路径和当前故障状态（**两部ADF系统皆故障，而面板上的频率显示窗显示正常**），我们可以进一步分析并提出下一步最有可能的几个原因及排故方法如下：\n\n---\n\n## ✅ 当前状态总结\n\n- **故障现象**：两部ADF系统均不工作，但控制面板频率显示正常。\n- **已排除的可能原因**：\n  - ADF接收机本身故障（已对调测试，故障依旧）。\n  - ADF控制面板显示部分故障（频率显示正常）。\n  - ADF控制面板到接收机的电源线路故障（已测量正常）。\n- **可能的共用故障点**：两部ADF系统共用的电源、控制信号、天线系统或环境干扰。\n\n---\n\n## 🔍 下一步最有可能的几个原因分析\n\n### 1. **第二套转换汇流条（115V AC）供电异常**\n- **分析**：虽然从控制面板到接收机的线路正常，但第二套转换汇流条本身可能没有供电，或跳开关C1383未闭合/损坏。\n- **排故方法**：\n  - 使用万用表测量第二套转换汇流条的电压是否为115V AC。\n  - 检查跳开关C1383是否闭合，是否有烧毁、接触不良或跳闸。\n  - 若跳开关正常但电压异常，需进一步检查电源系统（如TRU、静变流器等）。\n\n---\n\n### 2. **ADF控制面板的控制信号输出异常**\n- **分析**：虽然频率显示正常，但控制面板可能未正确输出控制信号（如频率调谐信号、工作模式信号）到接收机。\n- **排故方法**：\n  - 使用示波器或信号发生器检查控制面板输出到接收机的控制信号是否正常。\n  - 检查控制面板内部电路板是否有烧毁、虚焊或电容鼓包。\n  - 若控制信号异常，可能需要更换控制面板或修复电路板。\n\n---\n\n### 3. **天线系统或天线切换开关故障**\n- **分析**：两部ADF系统共用天线系统，若天线切换开关故障或天线连接不良，可能导致两部接收机均无法接收到信号。\n- **排故方法**：\n  - 检查ADF天线连接是否牢固，是否有断线或接触不良。\n  - 检查天线切换开关（如位于P6板或控制面板内部）是否正常工作。\n  - 使用信号发生器模拟AM信号，测试天线是否能正常接收信号。\n\n---\n\n### 4. **接收机公共电源或接地问题**\n- **分析**：虽然电源线路正常，但接收机可能因接地不良或公共电源故障而无法工作。\n- **排故方法**：\n  - 检查接收机的接地线是否连接良好。\n  - 检查接收机的公共电源（如28V DC）是否正常。\n  - 检查接收机的安装位置是否有电磁干扰（EMI）。\n\n---\n\n### 5. **软件/固件问题（可能性较低，但不可忽视）**\n- **分析**：若接收机硬件均正常，但系统仍无法工作，可能是软件或固件异常。\n- **排故方法**：\n  - 尝试对接收机进行软件重置或固件更新。\n  - 检查是否有相关的软件版本兼容性问题。\n  - 若有可用的软件诊断工具，可尝试读取接收机的故障代码。\n\n---\n\n## ✅ 建议的下一步排故流程\n\n| 步骤 | 检查内容 | 工具/方法 | 预期结果 |\n|------|----------|-----------|----------|\n| 1 | 第二套转换汇流条电压 | 万用表 | 确认115V AC供电正常 |\n| 2 | 跳开关C1383状态 | 目视+万用表 | 确认闭合且无烧毁 |\n| 3 | 控制面板输出信号 | 示波器/信号发生器 | 确认控制信号正常 |\n| 4 | ADF天线连接与切换开关 | 目视+信号测试 | 确认天线系统正常 |\n| 5 | 接收机接地与公共电源 | 万用表 | 确认接地良好、电源正常 |\n| 6 | 接收机软件/固件 | 软件工具 | 确认无软件异常 |\n\n---\n\n## 📌 总结\n\n当前故障表现为**两部ADF系统均不工作，但控制面板显示正常**，说明故障点不在接收机本身或控制面板显示部分，而更可能出在**共用的电源、控制信号或天线系统**。建议优先检查**第二套转换汇流条供电**和**控制面板输出信号**，再逐步排查天线系统和接收机公共部分。\n\n如需进一步帮助，可提供接收机型号、控制面板型号或具体测试数据，以便更精准定位故障。""")
    # ]

    for ref, cand in test_cases:
        score = evaluator.evaluate(ref, cand)
        print(f"标准答案: '{ref}'")
        print(f"模型答案: '{cand}'")
        print(f"评估结果: {score} ({['', '部分匹配', '匹配', '不匹配'][score]})")
        print("-" * 50)
        json_record = {
            "gold": ref,
            "response": cand,
            "result": ['', '部分匹配', '匹配', '不匹配'][score],
            "score": score
        }
        with open("../../result/fault_tree_test/evaluation_results_32b.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(json_record, ensure_ascii=False) + "\n")