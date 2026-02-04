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
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, pipeline, AutoModelForSequenceClassification
import numpy as np
from .seg_utils import bp, sdb_login, load_prompt, load_example_trace
from .seg_base import SegBase


class RTArgument(SegBase):
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
    def _segment(
            trace: str,
            model_name: Literal[
                "Qwen/Qwen2.5-7B-Instruct-1M",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "Qwen/Qwen2.5-7B-Instruct"],
            context_window: int = 2,
            system_prompt: str = '',
            adu_prompt: str = '',
            **kwargs
    ) -> Tuple[List[Tuple[int, int]], List[str]]:

        offsets = list(RTArgument.load_tokenizer().span_tokenize(trace))
        strace = [trace[o[0]:o[1]] for o in offsets]
        context_windows = []
        total_sentences = len(strace)

        for i in range(total_sentences):
            # Compute window boundaries
            start = max(0, i - context_window)
            end = min(total_sentences, i + context_window + 1)  # +1 because Python slices are exclusive
            # Extract context window
            context = strace[start:end]
            context_windows.append(context)

        prompts = []

        for i, tr in enumerate(strace):
            context = context_windows[i]
            if tr == context[0]:
                prompts.append(adu_prompt.format(PREVIOUS_SEGMENT='', TARGET_SEGMENT=tr, NEXT_SEGMENT=context[-1]))
            elif tr == context[-1]:
                prompts.append(adu_prompt.format(PREVIOUS_SEGMENT=context[0], TARGET_SEGMENT=tr, NEXT_SEGMENT=""))
            else:
                prompts.append(
                    adu_prompt.format(PREVIOUS_SEGMENT=context[0], TARGET_SEGMENT=tr, NEXT_SEGMENT=context[-1]))

        adu_labels = []
        model, tokenizer = RTArgument.load_model(model_name)

        for prompt in tqdm(prompts, desc="Labelling segments"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=128
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            adu_labels.append(response)

        labels = []

        for item in adu_labels:
            match = re.match(r'\*\*\[(.*?)\]\*\*', item)
            if match:
                labels.append(match.group(1))


        current_start = 0
        current_end = 0
        final_offsets = []
        final_labels = []
        current_label = labels[0]
        for i in range(len(labels)-1):
            offset = offsets[i]

            if labels[i] == labels[i + 1]:
                current_end = offset[1]
            else:
                final_offsets.append((current_start, current_end))
                final_labels.append(current_label)
                current_start = offsets[i + 1][0]
                current_end = offsets[i + 1][1]
                current_label = labels[i + 1]

        return final_offsets, final_labels
