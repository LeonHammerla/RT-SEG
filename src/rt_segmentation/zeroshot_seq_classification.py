import copy
import json
import os
import random
import time
from collections import Counter
from functools import lru_cache, partial
from typing import List, Dict, Tuple, Literal, Any
from datasets import load_dataset
import torch
from datasets import Dataset, concatenate_datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from surrealdb import Surreal, RecordID
from typing import List
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, pipeline
import numpy as np

from .seg_utils import bp, sdb_login, load_prompt, load_example_trace
from .seg_base import SegBase


class RTZeroShotSeqClassification(SegBase):
    @staticmethod
    @lru_cache(maxsize=1)
    def load_tokenizer():
        return PunktSentenceTokenizer()

    @staticmethod
    @lru_cache(maxsize=1)
    def load_model(model_name: Literal[
                    "facebook/bart-large-mnli"]):


        classifier = pipeline("zero-shot-classification",
                              model=model_name,
                              device_map="auto",
                              torch_dtype="auto"
                              )
        return classifier

    @staticmethod
    def chunks(s: str, n: int) -> list[str]:
        if n <= 0:
            raise ValueError("Chunk length must be positive")
        return [s[i:i + n] for i in range(0, len(s), n)]


    @staticmethod
    def _segment(
            trace: str,
            seg_base_unit: Literal["sent", "clause"],
            model_name: Literal[
                "facebook/bart-large-mnli",
                "FacebookAI/roberta-large-mnli",
                "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"],
            labels: List[str] = ["Context", "Planning", "Fact", "Restatement", "Example", "Reflection", "Conclusion"],
            **kwargs
    ) -> Tuple[List[Tuple[int, int]], List[str]]:

        offsets = SegBase.get_base_offsets(trace, seg_base_unit=seg_base_unit)

        classifier = RTZeroShotSeqClassification.load_model(model_name)
        offset_labels = []
        for offset in offsets:
            trace_seq = trace[offset[0]:offset[1]]

            result = classifier(trace_seq, labels, multi_label=False)
            offset_labels.append(result["labels"][0])

        assert len(offset_labels) == len(offsets), "Something went wrong with the model inference."
        if offset_labels == [] and offsets == []:
            return [], []

        final_offsets = []
        final_labels = []
        current_offset = offsets[0][0]
        for idx in range(1, len(offsets)):
            if offset_labels[idx - 1] != offset_labels[idx]:
                final_offsets.append((current_offset, offsets[idx - 1][1]))
                final_labels.append(offset_labels[idx - 1])
                current_offset = offsets[idx][0]

        if final_offsets:
            if final_offsets[-1][1] != offsets[-1][1]:
                final_offsets.append((current_offset, offsets[-1][1]))
                final_labels.append(offset_labels[-1])
        else:
            final_offsets.append((current_offset, offsets[-1][1]))
            final_labels.append(offset_labels[-1])


        return final_offsets, final_labels


class RTZeroShotSeqClassificationRF(RTZeroShotSeqClassification):
    _segment = staticmethod(partial(RTZeroShotSeqClassification._segment, labels=["Context",
                                                                                  "Planning",
                                                                                  "Fact",
                                                                                  "Reasoning",
                                                                                  "Restatement",
                                                                                  "Assumption",
                                                                                  "Example",
                                                                                  "Reflection",
                                                                                  "Conclusion"]))
class RTZeroShotSeqClassificationTA(RTZeroShotSeqClassification):
    _segment = staticmethod(partial(RTZeroShotSeqClassification._segment, labels=["Problem Setup",
                                                                                  "Plan Generation",
                                                                                  "Fact Retrieval",
                                                                                  "Active Computation",
                                                                                  "Uncertainty Management",
                                                                                  "Result Consolidation",
                                                                                  "Self Checking",
                                                                                  "Final Answer Emission"]))
