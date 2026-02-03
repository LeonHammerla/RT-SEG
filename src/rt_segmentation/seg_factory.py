from typing import Tuple, List, Literal, Optional, Union

from .seg_utils import bp, sdb_login, load_prompt, load_example_trace

from .seg_base import SegBase

from .rule_split_regex import RTRuleRegex
from .rule_split_newline import RTNewLine
from .llm_split_offset import RTLLMOffsetBased
from .llm_split_sent_chunks import RTLLMSentBased
from .llm_split_forced_decoder import RTLLMForcedDecoderBased
from .llm_split_surprisal import RTLLMSurprisal
from .llm_split_entropy import RTLLMEntropy
from .llm_split_topk import RTLLMTopKShift
from .llm_split_flatness_break import RTLLMFlatnessBreak
from .bertopic_segmentation import RTBERTopicSegmentation
from .zeroshot_seq_classification import RTZeroShotSeqClassification
from .prm_split import RTPRMBase
from .semantic_shift import RTEmbeddingBasedSemanticShift
from .entailment import RTEntailmentBasedSegmentation
from .late_fusion import (OffsetFusionFuzzy,
                          OffsetFusionGraph,
                          OffsetFusionMerge,
                          OffsetFusionVoting,
                          OffsetFusionFlatten,
                          OffsetFusionIntersect,
                          OffsetFusion,
                          LabelFusion)


class RTSeg:
    def __init__(self,
                 engines: Union[List[SegBase], SegBase],
                 aligner: Optional[OffsetFusion] = None,
                 label_fusion_type: Literal["majority", "concat"] = "majority"):
        """
        Initializes a new instance of the class with the specified engine.

        This class provides a mechanism to manage and interact with the 
        provided `SegBase` engine parameter. It initializes the object 
        and stores the engine instance for further use.

        Available segmentation engines:
        - RTRuleRegex: Regular expression based segmentation
        - RTNewLine: Newline based segmentation 
        - RTLLMOffsetBased: LLM-based segmentation using character offsets
        - RTLLMSentBased: LLM-based segmentation using sentence chunks
        - RTLLMForcedDecoderBased: LLM-based segmentation with forced decoding
        - RTLLMSurprisal: LLM-based segmentation using surprisal scores
        - RTLLMEntropy: LLM-based segmentation using entropy
        - RTLLMTopKShift: LLM-based segmentation using top-k token shifts
        - RTLLMFlatnessBreak: LLM-based segmentation using flatness breaks
        - RTBERTopicSegmentation: Topic-based segmentation using BERTopic
        - RTZeroShotSeqClassification: Zero-shot sequence classification
        - RTPRMBase: PRM-based segmentation
        - RTEmbeddingBasedSemanticShift: Semantic shift detection using embeddings
        - RTEntailmentBasedSegmentation: Segmentation based on entailment

        :param engine: The engine instance of type `SegBase` to be
                       utilized by the class. It plays a central 
                       role in all operations of the class.
        """
        if isinstance(engines, SegBase):
            engines = [engines]

        if aligner is None and len(engines) > 1:
            raise ValueError("Multiple engines provided without an alignment strategy.")

        self.engines = engines
        self.aligner = aligner
        self.label_fusion_type = label_fusion_type
        self.default_kwargs = {
            RTRuleRegex: {"model_name": None,
                          "system_prompt": None},
            RTNewLine: {"model_name": None,
                        "system_prompt": None},
            RTLLMOffsetBased: {"model_name": "Qwen/Qwen2.5-7B-Instruct",
                               "system_prompt": load_prompt("system_prompt_offset")},
            RTLLMSentBased: {"model_name": "Qwen/Qwen2.5-7B-Instruct",
                             "system_prompt": load_prompt("system_prompt_sentbased")},
            RTLLMForcedDecoderBased: {"model_name": "Qwen/Qwen2.5-7B-Instruct",
                                      "system_prompt": load_prompt("system_prompt_forceddecoder")},
            RTLLMSurprisal: {"model_name": "Qwen/Qwen2.5-7B-Instruct",
                             "system_prompt": load_prompt("system_prompt_surprisal")},
            RTLLMEntropy: {"model_name": "Qwen/Qwen2.5-7B-Instruct",
                           "system_prompt": load_prompt("system_prompt_surprisal")},
            RTLLMTopKShift: {"model_name": "Qwen/Qwen2.5-7B-Instruct",
                             "system_prompt": load_prompt("system_prompt_surprisal")},
            RTLLMFlatnessBreak: {"model_name": "Qwen/Qwen2.5-7B-Instruct",
                                 "system_prompt": load_prompt("system_prompt_surprisal")},
            RTBERTopicSegmentation: {"model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                                     "system_prompt": load_prompt("system_prompt_topic_label")},
            RTZeroShotSeqClassification: {"model_name": "facebook/bart-large-mnli",
                                          "system_prompt": "",
                                          "labels": ["verification", "pivot", "inference", "framing", "conclusion"]},
            RTPRMBase: {"model_name": "Qwen/Qwen2.5-Math-7B-PRM800K",
                        "system_prompt": ""},
            RTEmbeddingBasedSemanticShift: {"model_name": "all-MiniLM-L6-v2",
                                            "system_prompt": ""},
            RTEntailmentBasedSegmentation: {"model_name": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                                            "system_prompt": ""}
        }
        
    def __call__(self, trace: str, **kwargs) -> Tuple[List[Tuple[int, int]], List[str]]:
        engine_offsets = []
        engine_labels = []
        for engine in self.engines:
            for kwarg in kwargs:
                self.default_kwargs[engine][kwarg] = kwargs[kwarg]
            offsets, labels = engine._segment(trace, **self.default_kwargs[engine])
            engine_offsets.append(offsets)
            engine_labels.append(labels)

        if len(engine_offsets) > 1 and self.aligner is not None:
            fused_offsets = self.aligner.fuse(engine_offsets, **kwargs)
            fused_labels = LabelFusion.fuse(engine_offsets, engine_labels, fused_offsets, self.label_fusion_type)
            return fused_offsets, fused_labels
        else:
            return engine_offsets[0], engine_labels[0]