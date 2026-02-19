import random
import traceback
from typing import List, Tuple, Dict, Literal
import colorsys
from rich.text import Text
from rich.markup import escape
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual import on, events
from textual.widgets import Header, Footer, Static, Button, TextArea, Label, Select, SelectionList, RadioButton, RadioSet

from rt_segmentation import (RTLLMOffsetBased,
                             RTLLMForcedDecoderBased,
                             RTLLMSegUnitBased,
                             RTRuleRegex,
                             RTNewLine,
                             RTBERTopicSegmentation,
                             RTZeroShotSeqClassification,
                             bp,
                             sdb_login,
                             load_prompt,
                             load_example_trace,
                             RTLLMSurprisal,
                             RTLLMEntropy,
                             RTLLMTopKShift,
                             RTLLMFlatnessBreak,
                             export_gold_set,
                             RTEmbeddingBasedSemanticShift,
                             RTPRMBase,
                             RTEntailmentBasedSegmentation,
                             RTLLMReasoningFlow,
                             RTLLMThoughtAnchor,
                             RTLLMArgument,
                             RTZeroShotSeqClassificationTA,
                             RTZeroShotSeqClassificationRF,
                             RTSeg,
                             OffsetFusionFuzzy,
                             OffsetFusionGraph,
                             OffsetFusionMerge,
                             OffsetFusionVoting,
                             OffsetFusionIntersect,
                             OffsetFusion,
                             LabelFusion
                             )


RTLLMOffsetBased.load_model("Qwen/Qwen2.5-7B-Instruct")
RTLLMTopKShift.load_model("Qwen/Qwen2.5-0.5B-Instruct")
RTZeroShotSeqClassification.load_model("facebook/bart-large-mnli")
# RTPRMBase.load_model("Qwen/Qwen2.5-Math-7B-PRM800K")
RTEmbeddingBasedSemanticShift.load_embedding_model("all-MiniLM-L6-v2")
RTEntailmentBasedSegmentation.load_model("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")

