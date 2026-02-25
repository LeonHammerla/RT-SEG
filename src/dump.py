import time
from functools import partial
from typing import Any, Literal
import multiprocessing as mp

from surrealdb import Surreal, RecordID
from tqdm import tqdm

mp.set_start_method('spawn', force=True)


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
                             OffsetFusionGraph,
                             RTSeg,
                             OffsetFusion,
                             RTZeroShotSeqClassificationTA,
                             RTZeroShotSeqClassificationRF)

from eval_utils import (import_annotated_data,
                         export_gold_set,
                         export_rf_data_gold_set)


def rename():
    aligner: OffsetFusion = OffsetFusionGraph
    seg_base_unit: Literal["clause", "sent"] = "clause"
    models = [
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
    for comb in tqdm(models, desc="renaming"):
        if len(comb) == 1:
            old_id = comb[0].__name__
        else:
            old_id = "_".join([m.__name__ for m in comb])

        rt_seg = RTSeg(
            engines=comb,
            aligner=None if len(comb) == 1 else aligner,
            label_fusion_type="concat",
            seg_base_unit=seg_base_unit
        )
        print(rt_seg.exp_id, old_id)

        login_data = sdb_login()
        with Surreal(login_data["url"]) as db:
            db.signin({"username": login_data["user"], "password": login_data["pwd"]})
            db.use(login_data["ns"], login_data["db"])

            db.query(f"INSERT INTO {rt_seg.exp_id} SELECT * FROM {old_id};")
            db.query(f"DEFINE TABLE {rt_seg.exp_id} SCHEMALESS;")
            db.query(f"DEFINE INDEX idx_id ON {rt_seg.exp_id} FIELDS id;")

            db.query(f"INSERT INTO has_{rt_seg.exp_id} SELECT * FROM has{old_id};")
            db.query(f'UPDATE has_{rt_seg.exp_id} SET in = type::thing("{rt_seg.exp_id}", record::id(in)) WHERE record::tb(in) = "{old_id}";')
            db.query(f"DEFINE TABLE has_{rt_seg.exp_id} SCHEMALESS TYPE RELATION IN rtrace OUT {rt_seg.exp_id};")
            db.query(f"DEFINE INDEX idx_rt_id ON has_{rt_seg.exp_id} FIELDS id;")
            db.query(f"DEFINE INDEX idx_rt_in ON has_{rt_seg.exp_id} FIELDS in;")
            db.query(f"DEFINE INDEX idx_rt_out ON has_{rt_seg.exp_id} FIELDS out;")

            db.query(f"REMOVE TABLE {old_id};")
            db.query(f"REMOVE TABLE has_{old_id};")


def repair():
    models = [
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
    aligner: OffsetFusion = OffsetFusionGraph
    seg_base_unit: Literal["clause", "sent"] = "clause"

    login_data = sdb_login()
    with Surreal(login_data["url"]) as db:
        db.signin({"username": login_data["user"], "password": login_data["pwd"]})
        db.use(login_data["ns"], login_data["db"])

        traces = db.query("SELECT * FROM rtrace;")


        for comb in tqdm(models, desc="repairing"):
            rt_seg = RTSeg(
                engines=comb,
                aligner=None if len(comb) == 1 else aligner,
                label_fusion_type="concat",
                seg_base_unit=seg_base_unit
            )
            for trace in traces:
                split_id = RecordID(rt_seg.exp_id, trace.get("id").id)
                db.insert_relation(f"has_{rt_seg.exp_id}", {"in": trace.get("id"), "out": split_id})