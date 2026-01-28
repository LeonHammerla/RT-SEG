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
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import numpy as np
from .seg_utils import bp, sdb_login, load_prompt, load_example_trace
from .seg_base import SegBase


class RTLLMOffsetBased(SegBase):
    @staticmethod
    @lru_cache(maxsize=1)
    def load_tokenizer():
        return PunktSentenceTokenizer()

    @staticmethod
    @lru_cache(maxsize=1)
    def load_model(model_name: Literal[
                    "Qwen/Qwen2.5-7B-Instruct-1M",
                    "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "Qwen/Qwen2.5-7B-Instruct"]):

        # if model_name == "Qwen/Qwen2.5-7B-Instruct-1M" or model_name == "Qwen/Qwen2.5-7B-Instruct":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    @staticmethod
    def chunks(s: str, n: int) -> list[str]:
        if n <= 0:
            raise ValueError("Chunk length must be positive")
        return [s[i:i + n] for i in range(0, len(s), n)]

    @staticmethod
    def _trace_pass(trace: str,
                 prompt: str,
                 system_prompt: str,
                 model_name: Literal[
                     "Qwen/Qwen2.5-7B-Instruct-1M",
                     "mistralai/Mixtral-8x7B-Instruct-v0.1",
                     "Qwen/Qwen2.5-7B-Instruct"]
                 ):
        model, tokenizer = RTLLMOffsetBased.load_model(model_name)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{prompt}{trace}"}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=8000
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


    @staticmethod
    def _segment(
            trace: str,
            chunk_size: int,
            prompt: str,
            system_prompt: str,
            model_name: Literal[
                "Qwen/Qwen2.5-7B-Instruct-1M",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "Qwen/Qwen2.5-7B-Instruct"],
            margin: int = 200,
            max_retries_per_chunk: int = 10,
            **kwargs
    ) -> List[Tuple[int, int]]:

        all_segments: List[Tuple[int, int]] = []

        i = 0
        while i < max(0, len(trace) - 20):
            # Build chunk with carryover (carryover is CONTEXT, not to be re-segmented)
            base_chunk = trace[i:i + chunk_size]

            response = RTLLMOffsetBased._trace_pass(
                base_chunk, "", system_prompt, model_name
            )
            # --- robust JSON parsing ---
            try:
                print(base_chunk)
                print(response)
                response = response.strip()
                response = response[response.find("["):response.rfind("]") + 1]
                local_segments = json.loads(response)

                if isinstance(local_segments, dict):
                    local_segments = list(local_segments.values())

                if (
                        isinstance(local_segments, list)
                        and len(local_segments) == 2
                        and all(isinstance(x, int) for x in local_segments)
                ):
                    local_segments = [local_segments]

            except Exception:
                continue  # malformed output → stop this chunk

            if not local_segments:
                break

            for seg in local_segments:
                all_segments.append((i + seg[0], i + seg[1]))
            i = all_segments[-1][1]

        if i < len(trace):
            all_segments.append((i, len(trace)))

        return all_segments

