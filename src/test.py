import pytest
import time

from rt_segmentation import (RTLLMOffsetBased,
                             RTLLMForcedDecoderBased,
                             RTLLMSegUnitBased,
                             RTBERTopicSegmentation,
                             RTEmbeddingBasedSemanticShift,
                             RTZeroShotSeqClassification,
                             RTEntailmentBasedSegmentation,
                             RTRuleRegex,
                             RTNewLine,
                             bp,
                             sdb_login,
                             load_prompt,
                             load_example_trace, RTLLMSurprisal, RTLLMEntropy, RTLLMTopKShift, RTLLMFlatnessBreak,
                             export_gold_set,
                             RTSeg, OffsetFusionGraph,RTLLMReasoningFlow, RTLLMThoughtAnchor)


def test_RTLLMSentBased():
    res = RTLLMSegUnitBased._segment(trace=load_example_trace("trc1"),
                                     seg_base_unit="clause",
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

def test_RTZeroShotSeqClassification():
    offsets, labels = RTZeroShotSeqClassification._segment(trace=load_example_trace("trc1"),
                                                           seg_base_unit="clause",
                                                           model_name="facebook/bart-large-mnli")

    for ofs, label in zip(offsets, labels):
        print(50 * "=")
        print(load_example_trace("trc1")[ofs[0]:ofs[1]])
        print(label)

    assert isinstance(offsets, list)
    assert isinstance(labels, list)
    assert isinstance(offsets[0], tuple) or isinstance(offsets[0], list)
    assert isinstance(offsets[0][0], int) and isinstance(offsets[0][1], int)
    assert isinstance(labels[0], str)


@pytest.mark.parametrize("use_trace", ["trc1", "trc2"])
def test_RTBERTopicSegmentation(use_trace):
    trace_data = load_example_trace(use_trace)
    offsets, labels = RTBERTopicSegmentation._segment(trace=trace_data,
                                                      seg_base_unit="clause",
                                                      system_prompt=load_prompt("system_prompt_topic_label"),
                                                      model_name="Qwen/Qwen2.5-1.5B-Instruct",
                                                      all_custom_labels=True)

    for ofs, label in zip(offsets, labels):
        print(50 * "=")
        print(trace_data[ofs[0]:ofs[1]])
        print(label)

    # Assertions
    assert isinstance(offsets, list)
    assert isinstance(labels, list)
    assert isinstance(offsets[0], (tuple, list))
    assert isinstance(offsets[0][0], int)
    assert isinstance(offsets[0][1], int)
    assert isinstance(labels[0], str)

def test_RTEmbeddingBasedSemanticShift():
    offsets, labels = RTEmbeddingBasedSemanticShift._segment(trace=load_example_trace("trc1"),
                                                             seg_base_unit="clause",
                                                             min_threshold=0.4,
                                                             tolerance=0.20
                                                             )
    for ofs, label in zip(offsets, labels):
        print(50 * "=")
        print(load_example_trace("trc1")[ofs[0]:ofs[1]])
        print(label)
    assert isinstance(offsets, list)
    assert isinstance(labels, list)
    assert isinstance(offsets[0], tuple) or isinstance(offsets[0], list)
    assert isinstance(offsets[0][0], int) and isinstance(offsets[0][1], int)
    assert isinstance(labels[0], str)

def test_RTEntailmentBasedSegmentation():
    offsets, labels = RTEntailmentBasedSegmentation._segment(trace=load_example_trace("trc1"),
                                                             seg_base_unit="clause",)
    for ofs, label in zip(offsets, labels):
        print(50 * "=")
        print(load_example_trace("trc1")[ofs[0]:ofs[1]])
        print(label)
    assert isinstance(offsets, list)
    assert isinstance(labels, list)
    assert isinstance(offsets[0], tuple) or isinstance(offsets[0], list)
    assert isinstance(offsets[0][0], int) and isinstance(offsets[0][1], int)
    assert isinstance(labels[0], str)

def test_FactorySegmentation():
    rt_seg = RTSeg(engines=[RTRuleRegex, RTBERTopicSegmentation],
                   aligner=OffsetFusionGraph)
    offsets, labels = rt_seg(trace=load_example_trace("trc1"), seg_base_unit="clause")
    print(offsets)
    print(labels)
    assert isinstance(offsets, list)
    assert isinstance(labels, list)
    assert isinstance(offsets[0], tuple) or isinstance(offsets[0], list)
    assert isinstance(offsets[0][0], int) and isinstance(offsets[0][1], int)
    assert isinstance(labels[0], str)


def test_RTLLMReasoningFlow():
    offsets, labels = RTLLMReasoningFlow._segment(trace=load_example_trace("trc2"),
                                                             seg_base_unit="sent",system_prompt=load_prompt("system_prompt_reasoning_flow"),
                                                             user_prompt=load_prompt("user_prompt_reasoning_flow"),
                                                             model_name="Qwen/Qwen3-4B-Instruct-2507")
    for ofs, label in zip(offsets, labels):
        print(50 * "=")
        print(load_example_trace("trc2")[ofs[0]:ofs[1]])
        print(label)
    assert isinstance(offsets, list)
    assert isinstance(labels, list)
    assert isinstance(offsets[0], tuple) or isinstance(offsets[0], list)
    assert isinstance(offsets[0][0], int) and isinstance(offsets[0][1], int)
    assert isinstance(labels[0], str)

def test_RTLLMThoughtAnchor():
    offsets, labels = RTLLMThoughtAnchor._segment(trace=load_example_trace("trc1"),
                                                             seg_base_unit="sent",system_prompt=load_prompt("system_prompt_thought_anchor"),
                                                             user_prompt=load_prompt("user_prompt_thought_anchor"),
                                                             model_name="Qwen/Qwen3-4B-Instruct-2507")
    for ofs, label in zip(offsets, labels):
        print(50 * "=")
        print(load_example_trace("trc1")[ofs[0]:ofs[1]])
        print(label)
    assert isinstance(offsets, list)
    assert isinstance(labels, list)
    assert isinstance(offsets[0], tuple) or isinstance(offsets[0], list)
    assert isinstance(offsets[0][0], int) and isinstance(offsets[0][1], int)
    assert isinstance(labels[0], str)

if __name__ == "__main__":
    pytest.main([
        "-v",
        "-s",
        "--log-cli-level=INFO",
        __file__
    ])