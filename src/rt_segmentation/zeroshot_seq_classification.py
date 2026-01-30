import copy
import json
import os
import random
import time
from collections import Counter
from functools import lru_cache
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
            model_name: Literal[
                "Qwen/Qwen2.5-7B-Instruct-1M",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "Qwen/Qwen2.5-7B-Instruct"],
            chunk_size: int = 100,
            prompt: str = "",
            system_prompt: str = "",
            max_retry: int = 30,
            labels: List[str] = ["Context", "Planning", "Fact", "Restatement", "Example", "Reflection", "Conclusion"],
            **kwargs
    ) -> Tuple[List[Tuple[int, int]], List[str]]:

        offsets = list(RTZeroShotSeqClassification.load_tokenizer().span_tokenize(trace))

        classifier = RTZeroShotSeqClassification.load_model(model_name)
        final_offsets = []
        final_labels = []
        for offset in offsets:
            trace_seq = trace[offset[0]:offset[1]]

            result = classifier(trace_seq, labels, multi_label=False)
            final_offsets.append(offset)
            final_labels.append(result["labels"][0])



        return final_offsets, final_labels
