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
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from surrealdb import Surreal, RecordID
from typing import List
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, AutoModelForSequenceClassification
import numpy as np
from .seg_utils import bp, sdb_login, load_prompt, load_example_trace
from .seg_base import SegBase


class RTEntailmentBasedSegmentation(SegBase):
    @staticmethod
    @lru_cache(maxsize=1)
    def load_tokenizer():
        return PunktSentenceTokenizer()

    @staticmethod
    @lru_cache(maxsize=1)
    def load_model(model_name: Literal[
        "Qwen/Qwen2.5-7B-Instruct-1M",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Qwen/Qwen2.5-7B-Instruct"]
                   ):

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        return model, tokenizer

    @staticmethod
    def _predict_coherence(context: str, next_sent: str, model, tokenizer):
        inputs = tokenizer(
            context,
            next_sent,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            output = model(**inputs)

        probs = torch.softmax(output.logits[0], -1).tolist()

        p_entailment = probs[0]
        p_neutral = probs[1]
        #p_contradiction = probs[2]

        score = p_entailment + (p_neutral * 0.4)  # - (p_contradiction * 2.0)

        return max(0.0, min(1.0, score))


    @staticmethod
    def _segment(
            trace: str,
            max_retry: int = 30,
            tolerance: float = 0.15,
            min_threshold: float = 0.25,
            chunk_size: int = 100,
            prompt: str = "",
            system_prompt: str = "",
            model_name: Literal[
                "Qwen/Qwen2.5-7B-Instruct-1M",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "Qwen/Qwen2.5-7B-Instruct"] = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            **kwargs
    ) -> tuple[list[Any], list[str]]:

        offsets = list(RTEntailmentBasedSegmentation.load_tokenizer().span_tokenize(trace))
        # trace = nltk.sent_tokenize(trace)
        strace = [trace[tr[0]:tr[1]] for tr in offsets]
        model, tokenizer = RTEntailmentBasedSegmentation.load_model(model_name)
        # Current Segment State
        current_segment = [strace[0]]

        running_avg_score = 0.85

        final_offsets = []

        curr_offset = offsets[0]

        i = 1
        while i < len(strace):
            next_sent = strace[i]
            offset = offsets[i]

            context = " ".join(current_segment[-2:])
            score = RTEntailmentBasedSegmentation._predict_coherence(context, next_sent, model, tokenizer)

            dynamic_threshold = max(min_threshold, running_avg_score - tolerance)
            is_coherent = score >= dynamic_threshold

            if not is_coherent and (i + 1 < len(strace)):
                sentence_after = strace[i + 1]
                score_lookahead = RTEntailmentBasedSegmentation._predict_coherence(context, sentence_after, model, tokenizer)

                if score_lookahead >= dynamic_threshold:
                    is_coherent = True

            if is_coherent:
                current_segment.append(next_sent)
                running_avg_score = (running_avg_score * 0.7) + (score * 0.3)
                curr_offset = (curr_offset[0], offset[1])
            else:
                final_offsets.append(curr_offset)
                current_segment = [next_sent]
                running_avg_score = 0.85
                curr_offset = offset

            i += 1

        if current_segment:
            final_offsets.append(curr_offset)

        return final_offsets, ["UNLABELLED" for _ in final_offsets]
