import time

from rt_segmentation import (RTLLMBased,
                             RTRuleBased,
                             RTNewLine,
                             bp,
                             sdb_login,
                             load_prompt,
                             load_example_trace)


if __name__ == "__main__":

    """res = RTNewLine.paragraph_ranges_regex(trace)
    print(res)
    for idx, r in enumerate(res):
        print(trace[r[0]:r[1]])

    res = RTRuleBased.sentence_spans(trace)
    print(res)
    for idx, r in enumerate(res):
        print(trace[r[0]:r[1]])"""
    # mxtral: 123
    s = time.time()
    """res = RTLLMBased.segment_with_chunk_retry(trace=trc,
                                              chunk_size=700,
                                              prompt=RTLLMBased.PROMPT,
                                              system_prompt=RTLLMBased.SYSTEM_PROMPT,
                                              model_name="Qwen/Qwen2.5-7B-Instruct-1M",
                                              margin=100,
                                              max_retries_per_chunk=10)"""
    res = RTLLMBased.segment_with_sentence_chunks(trace=load_example_trace("trc1"),
                                                  chunk_size=20,
                                                  prompt="",
                                                  system_prompt=load_prompt("system_prompt_sentbased"),
                                                  model_name="Qwen/Qwen2.5-7B-Instruct")
    e = time.time()
    print(res)
    print(e-s)
    for idx, r in enumerate(res):
        #print(trc[r[0]:r[1]])
        #print("-"*10)
        pass

