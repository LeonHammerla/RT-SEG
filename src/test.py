import json

import pandas as pd
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
                             RTSeg, OffsetFusionGraph, RTLLMReasoningFlow, RTLLMArgument, RTLLMThoughtAnchor)

from eval_utils import (export_gold_set,
                        export_rf_data_gold_set,
                        import_annotated_data,
                        evaluate_aggregate_segmentations,
                        aggregated_results_to_json,
                        evaluate_segmentations)



def test_RTLLMSentBased():
    offsets, labels = RTLLMSegUnitBased._segment(trace=load_example_trace("trc1"),
                                     seg_base_unit="clause",
                                      chunk_size=20,
                                      prompt="",
                                      system_prompt=load_prompt("system_prompt_sentbased"),
                                      model_name="Qwen/Qwen2.5-7B-Instruct"
                                      # model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"
                                      )
    assert isinstance(offsets, list)
    assert isinstance(offsets[0], tuple) or isinstance(offsets[0], list)
    assert isinstance(offsets[0][0], int) and isinstance(offsets[0][1], int)


def test_RTLLMForcedDecoderBased():
    offsets, labels = RTLLMForcedDecoderBased._segment(trace=load_example_trace("trc1"),
                                               system_prompt=load_prompt("system_prompt_forceddecoder"),
                                               model_name="Qwen/Qwen2.5-7B-Instruct")
    assert isinstance(offsets, list)
    assert isinstance(offsets[0], tuple) or isinstance(offsets[0], list)
    assert isinstance(offsets[0][0], int) and isinstance(offsets[0][1], int)


def test_RTLLMSurprisal():
    offsets, labels = RTLLMSurprisal._segment(trace=load_example_trace("trc1"),
                                               system_prompt=load_prompt("system_prompt_surprisal"),
                                               model_name="Qwen/Qwen2.5-7B-Instruct")
    assert isinstance(offsets, list)
    assert isinstance(offsets[0], tuple) or isinstance(offsets[0], list)
    assert isinstance(offsets[0][0], int) and isinstance(offsets[0][1], int)

def test_RTLLMEntropy():
    offsets, labels = RTLLMEntropy._segment(trace=load_example_trace("trc1"),
                                               system_prompt=load_prompt("system_prompt_surprisal"),
                                               model_name="Qwen/Qwen2.5-7B-Instruct")
    assert isinstance(offsets, list)
    assert isinstance(offsets[0], tuple) or isinstance(offsets[0], list)
    assert isinstance(offsets[0][0], int) and isinstance(offsets[0][1], int)


def test_RTLLMTopKShift():
    offsets, labels = RTLLMTopKShift._segment(trace=load_example_trace("trc1"),
                                               system_prompt=load_prompt("system_prompt_surprisal"),
                                               model_name="Qwen/Qwen2.5-7B-Instruct")
    assert isinstance(offsets, list)
    assert isinstance(offsets[0], tuple) or isinstance(offsets[0], list)
    assert isinstance(offsets[0][0], int) and isinstance(offsets[0][1], int)


def test_RTLLMFlatnessBreak():
    offsets, labels = RTLLMFlatnessBreak._segment(trace=load_example_trace("trc1"),
                                               system_prompt=load_prompt("system_prompt_surprisal"),
                                               model_name="Qwen/Qwen2.5-7B-Instruct")
    assert isinstance(offsets, list)
    assert isinstance(offsets[0], tuple) or isinstance(offsets[0], list)
    assert isinstance(offsets[0][0], int) and isinstance(offsets[0][1], int)

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
                   aligner=OffsetFusionGraph,
                   seg_base_unit="clause")
    offsets, labels = rt_seg(trace=load_example_trace("trc1"))
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


def test_RTLLMArgument():
    offsets, labels = RTLLMArgument._segment(trace=load_example_trace("trc1"),
                                                  seg_base_unit="sent",
                                                  system_prompt=load_prompt("system_prompt_argument"),
                                                  user_prompt=load_prompt("user_prompt_argument"),
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

def test_RTLLMThoughtAnchor():
    offsets, labels = RTLLMThoughtAnchor._segment(trace=load_example_trace("trc1"),
                                                  seg_base_unit="sent",
                                                  system_prompt=load_prompt("system_prompt_thought_anchor"),
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


def test_SingleTraceEval():
    trace = "Step 1: Get data. Data is [1, 2]. Step 2: Sum data. Sum is 3. Step 3: Square it. Result is 9."

    segment_data = {
        "Ground_Truth": [(0, 31), (31, 59), (59, 84)],
        "Regex_Splitter": [(0, 31), (31, 84)],  # Missed Step 3
        "LLM_Fine": [(0, 16), (16, 31), (31, 46), (46, 59), (59, 84)],  # Over-segmented
        "Streber": [(0, 31), (31, 59), (59, 84)],
    }

    # Run single-trace evaluation
    tables = evaluate_segmentations(
        trace=trace,
        segmentations=segment_data,
        gold_key="Ground_Truth",
        sigma=5.0,
        window=3,
        slack=10,
    )

    print("=== Single Trace Evaluation ===")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        for name, df in tables.items():
            print(f"\n--- {name} ---")
            print(df)

    assert isinstance(tables, dict)


def test_AggregatedEval():
    trace = "Step 1: Get data. Data is [1, 2]. Step 2: Sum data. Sum is 3. Step 3: Square it. Result is 9."

    segment_data = {
        "Ground_Truth": [(0, 31), (31, 59), (59, 84)],
        "Regex_Splitter": [(0, 31), (31, 84)],  # Missed Step 3
        "LLM_Fine": [(0, 16), (16, 31), (31, 46), (46, 59), (59, 84)],  # Over-segmented
        "Streber": [(0, 31), (31, 59), (59, 84)],
    }
    # Aggregate across multiple traces (here using the same trace 3 times)
    agg_tables = evaluate_aggregate_segmentations(
        traces=[trace, trace, trace],
        segmentations=[segment_data, segment_data, segment_data],
        gold_key="Ground_Truth",
        sigma=5.0,
        window=3,
        slack=10,
    )

    print("\n=== Aggregated Linear Metrics ===")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(agg_tables['linear_metrics'])

    print("\n=== Aggregated Agreement Metrics ===")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(agg_tables['pairwise_agreement_metrics'])

    print("\n=== Aggregated Agreement Metrics ===")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(agg_tables['per_method_agreement_metrics'])

    assert isinstance(json.dumps(aggregated_results_to_json(agg_tables)), str)



if __name__ == "__main__":
    pytest.main([
        "-v",
        "-s",
        "--log-cli-level=INFO",
        # "-k", "test_RTBERTopicSegmentation",
        __file__
    ])