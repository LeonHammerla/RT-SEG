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


class RTLLMForcedDecoderBased(SegBase):
    PUNCTUATION = {".", "!", "?", ";", ":", "\n"}
    SOFT_PUNCTUATION = {",", "\t"}

    @staticmethod
    @lru_cache(maxsize=1)
    def load_tokenizer():
        return PunktSentenceTokenizer()

    @staticmethod
    @lru_cache(maxsize=1)
    def load_model(model_name: Literal[
                    "Qwen/Qwen2.5-7B-Instruct-1M",
                    "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "Qwen/Qwen2.5-7B-Instruct"],
                   special_token: str = "<|seg|>"
                   ):

        # if model_name == "Qwen/Qwen2.5-7B-Instruct-1M" or model_name == "Qwen/Qwen2.5-7B-Instruct":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})
        model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer

    @staticmethod
    def truncate_kv_dynamic(cache: DynamicCache, max_tokens: int):
        """
        Truncate Qwen2 DynamicCache in-place.
        Works with transformers 4.57.x
        """
        if cache is None:
            return cache

        current_len = cache.get_seq_length()
        if current_len > max_tokens:
            cache.crop(current_len - max_tokens)

        return cache

    @staticmethod
    def normalize(x):
        x = np.asarray(x)
        return (x - x.mean()) / (x.std() + 1e-6)

    @staticmethod
    def compute_split_score(gap_z, punc, alpha=1.0, beta=0.8):
        """
        Higher score => more likely to split
        """
        return (
                -alpha * gap_z  # low gap => good split
                + beta * punc  # punctuation prior
        )

    @staticmethod
    def choose_threshold(scores, q=97.5):
        return np.percentile(scores, q)

    @staticmethod
    def _trace_pass(trace: str,
                    system_prompt: str,
                    model_name: Literal[
                        "Qwen/Qwen2.5-7B-Instruct-1M",
                        "mistralai/Mixtral-8x7B-Instruct-v0.1",
                        "Qwen/Qwen2.5-7B-Instruct"],
                    max_kv_tokens: int = 512,
                    alpha: float = 1.0,
                    beta: float = 1.5,
                    quantile: float = 90.0,
                    sep_tok: str = "<|seg|>"):
        """
        First pass: compute normalized "gap" scores for separator insertion.
        """
        model, tokenizer = RTLLMForcedDecoderBased.load_model(model_name,
                                                              special_token=sep_tok
                                                              )
        model.eval()
        sep_ids = []
        for s in ("Step", " Step", sep_tok):
            ids = tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                sep_ids.append(ids[0])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": trace},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        trace_ids = tokenizer.encode(trace, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], device=model.device)

        decoded_so_far = ""
        past_key_values = None
        trace_ptr = 0

        gaps = []
        puncs = []

        # initial prompt forward
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            if past_key_values is not None:
                past_key_values = RTLLMForcedDecoderBased.truncate_kv_dynamic(past_key_values, max_kv_tokens)

        while trace_ptr < len(trace_ids):
            logits = out.logits[0, -1]
            log_probs = torch.log_softmax(logits, dim=-1)
            true_id = trace_ids[trace_ptr]

            last_char = decoded_so_far[-1] if decoded_so_far else ""
            if last_char in RTLLMForcedDecoderBased.PUNCTUATION:
                sep_bias = 1.0
            elif last_char in RTLLMForcedDecoderBased.SOFT_PUNCTUATION:
                sep_bias = 0.5
            else:
                sep_bias = 0.0

            true_logp = log_probs[true_id]
            sep_logp = torch.max(log_probs[sep_ids])

            # FLIP: gap > 0 → separator more likely
            gap = sep_logp - true_logp
            gaps.append(gap.item())
            puncs.append(sep_bias)

            # force true token
            next_id = torch.tensor([[true_id]], device=model.device)
            decoded_so_far += tokenizer.decode([true_id], skip_special_tokens=False)
            trace_ptr += 1

            # forward step
            with torch.no_grad():
                out = model(input_ids=next_id, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
                if past_key_values is not None:
                    past_key_values = RTLLMForcedDecoderBased.truncate_kv_dynamic(past_key_values, max_kv_tokens)

        # z-normalize gap
        gap_z = RTLLMForcedDecoderBased.normalize(np.array(gaps))
        scores = alpha * gap_z + beta * np.array(puncs)
        threshold = RTLLMForcedDecoderBased.choose_threshold(scores, q=quantile)
        return threshold, gap_z.mean(), gap_z.std()

    @staticmethod
    def _segment_pass(trace: str,
                      system_prompt: str,
                      model_name: Literal[
                          "Qwen/Qwen2.5-7B-Instruct-1M",
                          "mistralai/Mixtral-8x7B-Instruct-v0.1",
                          "Qwen/Qwen2.5-7B-Instruct"],
                      threshold: float,
                      gap_mean: float,
                      gap_std: float,
                      max_kv_tokens: int = 512,
                      alpha: float = 1.0,
                      beta: float = 1.5,
                      sep_tok: str = "<|seg|>"):
        """
        Second pass: perform segmentation using computed threshold.
        """
        model, tokenizer = RTLLMForcedDecoderBased.load_model(model_name,
                                                              special_token=sep_tok
                                                              )
        model.eval()
        sep_id = tokenizer.get_vocab().get("<|seg|>")
        sep_ids = []
        for s in ("Step", " Step", sep_tok):
            ids = tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                sep_ids.append(ids[0])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": trace},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        trace_ids = tokenizer.encode(trace, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], device=model.device)

        decoded_so_far = ""
        char_cursor = 0
        boundaries = [0]
        past_key_values = None
        trace_ptr = 0
        last_tokid = None
        # initial prompt
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            if past_key_values is not None:
                past_key_values = RTLLMForcedDecoderBased.truncate_kv_dynamic(past_key_values, max_kv_tokens)

        while trace_ptr < len(trace_ids):
            logits = out.logits[0, -1]
            log_probs = torch.log_softmax(logits, dim=-1)
            true_id = trace_ids[trace_ptr]

            last_char = decoded_so_far[-1] if decoded_so_far else ""
            if last_char in RTLLMForcedDecoderBased.PUNCTUATION:
                sep_bias = 1.0
            elif last_char in RTLLMForcedDecoderBased.SOFT_PUNCTUATION:
                sep_bias = 0.2
            else:
                sep_bias = 0.0

            true_logp = log_probs[true_id]
            sep_logp = torch.max(log_probs[sep_ids])

            gap_z = (sep_logp - true_logp - gap_mean) / gap_std
            score = alpha * gap_z + beta * sep_bias

            # print(true_logp, sep_logp, score, threshold)
            if (not true_id in sep_ids) and (score > threshold) and (last_tokid != sep_id):
                # insert boundary
                boundaries.append(char_cursor)
                next_id = torch.tensor([[sep_id]], device=model.device)
                last_tokid = sep_id
            else:
                next_id = torch.tensor([[true_id]], device=model.device)
                token_text = tokenizer.decode([true_id], skip_special_tokens=False)
                decoded_so_far += token_text
                char_cursor += len(token_text)
                trace_ptr += 1
                last_tokid = true_id

            with torch.no_grad():
                out = model(input_ids=next_id, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
                if past_key_values is not None:
                    past_key_values = RTLLMForcedDecoderBased.truncate_kv_dynamic(past_key_values, max_kv_tokens)

        boundaries.append(len(trace))
        print(boundaries)
        segments = [(a, b) for a, b in zip(boundaries[:-1], boundaries[1:]) if a < b]
        return segments

    @staticmethod
    def _segment(trace: str,
                 system_prompt: str,
                 model_name: Literal[
                     "Qwen/Qwen2.5-7B-Instruct-1M",
                     "mistralai/Mixtral-8x7B-Instruct-v0.1",
                     "Qwen/Qwen2.5-7B-Instruct"],
                 max_kv_tokens: int = 512,
                    alpha: float = 1.0,
                    beta: float = 2,
                    quantile: float = 90.0,
                 sep_tok: str = "<|seg|>",
                 **kwargs):
        threshold, gap_mean, gap_std = RTLLMForcedDecoderBased._trace_pass(trace=trace,
                                                                           system_prompt=system_prompt,
                                                                           model_name=model_name,
                                                                           max_kv_tokens=max_kv_tokens,
                                                                           alpha=alpha,
                                                                           beta=beta,
                                                                           quantile=quantile,
                                                                           sep_tok=sep_tok)
        # print(threshold, gap_mean, gap_std)
        return RTLLMForcedDecoderBased._segment_pass(trace=trace,
                                                     system_prompt=system_prompt,
                                                     model_name=model_name,
                                                     threshold=threshold,
                                                     gap_mean=gap_mean,
                                                     gap_std=gap_std,
                                                     max_kv_tokens=max_kv_tokens,
                                                     alpha=alpha,
                                                     beta=beta,
                                                     sep_tok=sep_tok)