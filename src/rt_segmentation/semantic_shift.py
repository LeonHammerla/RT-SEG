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
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import numpy as np
from .seg_utils import bp, sdb_login, load_prompt, load_example_trace
from .seg_base import SegBase


class RTEmbeddingBasedSemanticShift(SegBase):
    @staticmethod
    @lru_cache(maxsize=1)
    def load_tokenizer():
        return PunktSentenceTokenizer()

    @staticmethod
    @lru_cache(maxsize=1)
    def load_embedding_model(model_name: Literal[
        "all-MiniLM-L6-v2",
        "Qwen/Qwen3-Embedding-0.6B",]):
        embedding_model = SentenceTransformer(model_name)
        return embedding_model

    @staticmethod
    def _get_centroid(vectors):
        return np.mean(vectors, axis=0)


    @staticmethod
    def _segment(
            trace: str,

            embedding_model_name: Literal[
                "Qwen/Qwen2.5-7B-Instruct-1M",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "Qwen/Qwen2.5-7B-Instruct"] = "all-MiniLM-L6-v2",
            max_retry: int = 30,
            tolerance: float = 0.15,
            min_threshold: float = 0.4,
            chunk_size: int = 100,
            prompt: str = "",
            system_prompt: str = "",
            model_name: Literal[
                "Qwen/Qwen2.5-7B-Instruct-1M",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "Qwen/Qwen2.5-7B-Instruct"] = "Qwen/Qwen2.5-7B-Instruct",
            **kwargs
    ) -> tuple[list[Any], list[str]]:

        offsets = list(RTEmbeddingBasedSemanticShift.load_tokenizer().span_tokenize(trace))
        # trace = nltk.sent_tokenize(trace)
        strace = [trace[tr[0]:tr[1]] for tr in offsets]
        embeddings = RTEmbeddingBasedSemanticShift.load_embedding_model(embedding_model_name).encode(strace)
        # Current Segment State
        current_segment_sents = [strace[0]]
        current_segment_vecs = [embeddings[0]]

        running_avg_sim = 1.0

        final_offsets = []

        curr_offset = offsets[0]

        i = 1
        while i < len(strace):
            sent = strace[i]
            vec = embeddings[i]
            offset = offsets[i]

            centroid = RTEmbeddingBasedSemanticShift._get_centroid(current_segment_vecs)
            sim = util.cos_sim(centroid, vec).item()

            dynamic_threshold = running_avg_sim - tolerance

            effective_threshold = max(dynamic_threshold, min_threshold)
            is_match = sim >= effective_threshold

            if not is_match and (i + 1 < len(strace)):
                next_vec = embeddings[i + 1]
                sim_next = util.cos_sim(next_vec, centroid).item()

                # If the NEXT sentence fits the current segment, the current one
                # might be a "bridge" or "outlier" (e.g. "Note: remember X").
                # We keep it to preserve flow.
                if sim_next >= effective_threshold:
                    is_match = True


            if is_match:
                current_segment_sents.append(sent)
                current_segment_vecs.append(vec)

                running_avg_sim = (running_avg_sim + sim) / 2
                curr_offset = (curr_offset[0], offset[1])
            else:
                final_offsets.append(curr_offset)
                curr_offset = offset

                current_segment_sents = [sent]
                current_segment_vecs = [vec]
                running_avg_sim = 1.0
            i += 1

        if current_segment_sents:
            final_offsets.append(curr_offset)

        return final_offsets, ["UNLABELLED" for _ in final_offsets]