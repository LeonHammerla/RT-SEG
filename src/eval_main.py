import copy
from typing import Literal

import pandas as pd
from surrealdb import Surreal
from tqdm import tqdm

from rt_segmentation import sdb_login, evaluate_aggregate_segmentations, evaluate_segmentations, \
    evaluate_approaches_bounding_similarity, score_approaches_triadic_boundary_similarity_complete_ta, \
    get_single_engine_results_ta_and_rf, plot_single_engine_results_ta_and_rf
from rt_segmentation import (RTLLMOffsetBased,
                             RTLLMForcedDecoderBased,
                             RTLLMSegUnitBased,
                             RTRuleRegex,
                             RTNewLine,
                             RTPRMBase,
                             RTLLMReasoningFlow,
                             RTLLMArgument,
                             RTLLMThoughtAnchor,
                             RTLLMEntropy,
                             RTLLMTopKShift,
                             RTLLMFlatnessBreak,
                             RTLLMSurprisal,
                             RTBERTopicSegmentation,
                             RTZeroShotSeqClassification,
                             RTEntailmentBasedSegmentation,
                             RTEmbeddingBasedSemanticShift,
                             bp, sdb_login, load_prompt, load_example_trace,
                             export_gold_set, export_rf_data_gold_set, upload_rf_data,
                             OffsetFusionGraph,
                             RTSeg,
                             OffsetFusion,
                             RTZeroShotSeqClassificationTA,
                             RTZeroShotSeqClassificationRF,
                             import_annotated_data,
                             score_approaches_triadic_boundary_similarity, plot_score_vs_time_ta, plot_score_vs_time_rf)
from rt_segmentation.seg_eval_utils import extract_all_from_database
from rt_segmentation.seg_plot_utils import boxplot_evolutionary_search, kde_evolutionary_search


def human_agreement():
    login_data = sdb_login()
    with Surreal(login_data["url"]) as db:
        db.signin({"username": login_data["user"], "password": login_data["pwd"]})
        db.use(login_data["ns"], login_data["db"])

        res = db.query("SELECT *, ->has_though_anchor_gold_ha->though_anchor_gold_ha.* as ha, ->has_though_anchor_gold_ve->though_anchor_gold_ve.* as ve from rtrace")

    traces = []
    anno_data = []
    for rtrace in res:
        traces.append(rtrace["rt"])
        anno_data.append({"ve": rtrace["ve"][0]["split"],
                          "ha": rtrace["ha"][0]["split"]})

    """# print(traces[0])
    # print(anno_data[0])

    # Aggregate across multiple traces (here using the same trace 3 times)
    agg_tables = evaluate_aggregate_segmentations(
        traces=traces,
        segmentations=anno_data,
        gold_key="ha",
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

    # Aggregate across multiple traces (here using the same trace 3 times)
    agg_tables = evaluate_aggregate_segmentations(
        traces=traces,
        segmentations=anno_data,
        gold_key="ve",
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
        print(agg_tables['per_method_agreement_metrics'])"""

    evaluate_approaches_bounding_similarity(traces, anno_data)





if __name__ == "__main__":
    mm = [
        [RTLLMOffsetBased],
        [RTLLMForcedDecoderBased],
        [RTLLMSegUnitBased],
        [RTRuleRegex],
        [RTNewLine],
        [RTPRMBase],
        [RTEntailmentBasedSegmentation],
        [RTLLMEntropy],
        [RTLLMTopKShift],
        [RTLLMFlatnessBreak],
        [RTLLMSurprisal],
        [RTBERTopicSegmentation],
        [RTZeroShotSeqClassification],
        [RTLLMReasoningFlow],
        [RTLLMArgument],
        [RTLLMThoughtAnchor],
        [RTEmbeddingBasedSemanticShift],
        [RTZeroShotSeqClassificationRF],
        [RTZeroShotSeqClassificationTA],
        [RTZeroShotSeqClassificationRF, RTZeroShotSeqClassificationTA],
        [RTLLMReasoningFlow, RTLLMThoughtAnchor],
        [RTLLMReasoningFlow, RTLLMThoughtAnchor, RTLLMArgument],
        [RTLLMThoughtAnchor, RTZeroShotSeqClassificationTA],
        [RTLLMReasoningFlow, RTZeroShotSeqClassificationRF],
        [RTNewLine, RTRuleRegex, RTLLMSegUnitBased]
    ]
    aa: OffsetFusion = OffsetFusionGraph
    ss: Literal["clause", "sent"] = "clause"
    # score_approaches_triadic_boundary_similarity(mm, aa, ss)

    # score_approaches_triadic_boundary_similarity_complete()
    # plot_score_vs_time_ta()
    # plot_score_vs_time_rf()
    plot_single_engine_results_ta_and_rf(3)

    # extract_all_from_database()
    # boxplot_evolutionary_search()

    # kde_evolutionary_search()