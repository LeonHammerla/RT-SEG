import pytest
import time

from rt_segmentation import (RTLLMOffsetBased,
                             RTLLMForcedDecoderBased,
                             RTLLMSentBased,
                             RTRuleRegex,
                             RTNewLine,
                             bp,
                             sdb_login,
                             load_prompt,
                             load_example_trace, RTLLMSurprisal, RTLLMEntropy, RTLLMTopKShift, RTLLMFlatnessBreak,
                             export_gold_set)



def test_RTLLMSentBased():
    res = RTLLMSentBased._segment(trace=load_example_trace("trc1"),
                                  chunk_size=20,
                                  prompt="",
                                  system_prompt=load_prompt("system_prompt_sentbased"),
                                  model_name="Qwen/Qwen2.5-7B-Instruct"
                                  # model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"
                                  )
    assert isinstance(res, list)
    assert isinstance(res[0], tuple) or isinstance(res[0], list)
    assert isinstance(res[0][0], int) and isinstance(res[0][1], int)


def test_RTLLMForcedDecoderBased():
    res = RTLLMForcedDecoderBased._segment(trace=load_example_trace("trc1"),
                                               system_prompt=load_prompt("system_prompt_forceddecoder"),
                                               model_name="Qwen/Qwen2.5-7B-Instruct")
    assert isinstance(res, list)
    assert isinstance(res[0], tuple) or isinstance(res[0], list)
    assert isinstance(res[0][0], int) and isinstance(res[0][1], int)


def test_RTLLMSurprisal():
    res = RTLLMSurprisal._segment(trace=load_example_trace("trc1"),
                                               system_prompt=load_prompt("system_prompt_surprisal"),
                                               model_name="Qwen/Qwen2.5-7B-Instruct")
    assert isinstance(res, list)
    assert isinstance(res[0], tuple) or isinstance(res[0], list)
    assert isinstance(res[0][0], int) and isinstance(res[0][1], int)

def test_RTLLMEntropy():
    res = RTLLMEntropy._segment(trace=load_example_trace("trc1"),
                                               system_prompt=load_prompt("system_prompt_surprisal"),
                                               model_name="Qwen/Qwen2.5-7B-Instruct")
    assert isinstance(res, list)
    assert isinstance(res[0], tuple) or isinstance(res[0], list)
    assert isinstance(res[0][0], int) and isinstance(res[0][1], int)


def test_RTLLMTopKShift():
    res = RTLLMTopKShift._segment(trace=load_example_trace("trc1"),
                                               system_prompt=load_prompt("system_prompt_surprisal"),
                                               model_name="Qwen/Qwen2.5-7B-Instruct")
    assert isinstance(res, list)
    assert isinstance(res[0], tuple) or isinstance(res[0], list)
    assert isinstance(res[0][0], int) and isinstance(res[0][1], int)


def test_RTLLMFlatnessBreak():
    res = RTLLMFlatnessBreak._segment(trace=load_example_trace("trc1"),
                                               system_prompt=load_prompt("system_prompt_surprisal"),
                                               model_name="Qwen/Qwen2.5-7B-Instruct")
    assert isinstance(res, list)
    assert isinstance(res[0], tuple) or isinstance(res[0], list)
    assert isinstance(res[0][0], int) and isinstance(res[0][1], int)


if __name__ == "__main__":
    pytest.main([
        "-v",
        "-s",
        "--log-cli-level=INFO",
        __file__
    ])