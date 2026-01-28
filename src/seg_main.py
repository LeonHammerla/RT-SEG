import time

from rt_segmentation import (RTLLMOffsetBased,
                             RTLLMForcedDecoderBased,
                             RTLLMSentBased,
                             RTRuleRegex,
                             RTNewLine,
                             bp,
                             sdb_login,
                             load_prompt,
                             load_example_trace, RTLLMSurprisal, RTLLMEntropy, RTLLMTopKShift, RTLLMFlatnessBreak)


def test():
    s = time.time()
    res = RTLLMSentBased.segment_with_sentence_chunks(trace=load_example_trace("trc1"),
                                                      chunk_size=20,
                                                      prompt="",
                                                      system_prompt=load_prompt("system_prompt_sentbased"),
                                                      model_name="Qwen/Qwen2.5-7B-Instruct"
                                                      # model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"
                                                      )
    e = time.time()
    print(res)
    print(e - s)
    for idx, r in enumerate(res):
        # print(trc[r[0]:r[1]])
        # print("-"*10)
        pass


def test2():
    offsets = RTLLMForcedDecoderBased._segment(trace=load_example_trace("trc1"),
                                               system_prompt=load_prompt("system_prompt_forceddecoder"),
                                               model_name="Qwen/Qwen2.5-7B-Instruct")
    print(offsets)
    for ofs in offsets:
        print(50 * "=")
        print(load_example_trace("trc1")[ofs[0]:ofs[1]])


def test3():
    offsets = RTLLMSurprisal._segment(trace=load_example_trace("trc1"),
                                               system_prompt=load_prompt("system_prompt_surprisal"),
                                               model_name="Qwen/Qwen2.5-7B-Instruct")
    print(offsets)
    for ofs in offsets:
        print(50 * "=")
        print(load_example_trace("trc1")[ofs[0]:ofs[1]])

def test4():
    offsets = RTLLMEntropy._segment(trace=load_example_trace("trc1"),
                                               system_prompt=load_prompt("system_prompt_surprisal"),
                                               model_name="Qwen/Qwen2.5-7B-Instruct")
    print(offsets)
    for ofs in offsets:
        print(50 * "=")
        print(load_example_trace("trc1")[ofs[0]:ofs[1]])


def test5():
    offsets = RTLLMTopKShift._segment(trace=load_example_trace("trc1"),
                                               system_prompt=load_prompt("system_prompt_surprisal"),
                                               model_name="Qwen/Qwen2.5-7B-Instruct")
    print(offsets)
    for ofs in offsets:
        print(50 * "=")
        print(load_example_trace("trc1")[ofs[0]:ofs[1]])


def test6():
    offsets = RTLLMFlatnessBreak._segment(trace=load_example_trace("trc1"),
                                               system_prompt=load_prompt("system_prompt_surprisal"),
                                               model_name="Qwen/Qwen2.5-7B-Instruct")
    print(offsets)
    for ofs in offsets:
        print(50 * "=")
        print(load_example_trace("trc1")[ofs[0]:ofs[1]])

if __name__ == "__main__":
    # RTLLMBased.segment()
    test4()

