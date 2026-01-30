import time

from rt_segmentation import (RTLLMOffsetBased,
                             RTLLMForcedDecoderBased,
                             RTLLMSentBased,
                             RTRuleRegex,
                             RTNewLine,
                             RTPRMBase,
                             bp,
                             sdb_login,
                             load_prompt,
                             load_example_trace, RTLLMSurprisal, RTLLMEntropy, RTLLMTopKShift, RTLLMFlatnessBreak,
                             export_gold_set)


if __name__ == "__main__":
    # export_gold_set()
    res = RTPRMBase.model_inference(problem='Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$.',
                              sentences=["Okay, let's try to solve this problem step by step.",
                                         "The question is asking for the sum of all integer bases b > 9 where the number 17 in base b divides 97 in base b.",
                                         "First, I need to understand what 17_b and 97_b represent in decimal.",
                                         "For any base b, a two-digit number like xy_b translates to x*b + y in decimal.",
                                         "So, 17_b would be 1*b + 7, which is b + 7. Similarly, 97_b is 9*b + 7.",
                                         "The problem states that 17_b divides 97_b.",
                                         "That means when we convert both to decimal, (b + 7) must divide (9b + 7) without leaving a remainder."])
    print(res)

