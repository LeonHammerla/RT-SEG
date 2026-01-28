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


class RTLLMTopKShift(SegBase):
    PUNCTUATION = {".", "!", "?", "\n"}
    SOFT_PUNCTUATION = {",", "\t", ";", ":"}

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
    def truncate_kv_dynamic(cache: DynamicCache, max_tokens: int):
        if cache is None:
            return cache
        current_len = cache.get_seq_length()
        if current_len > max_tokens:
            cache.crop(current_len - max_tokens)
        return cache

    @staticmethod
    def _trace_pass(trace: str,
                    system_prompt: str,
                    model_name: Literal[
                        "Qwen/Qwen2.5-7B-Instruct-1M",
                        "mistralai/Mixtral-8x7B-Instruct-v0.1",
                        "Qwen/Qwen2.5-7B-Instruct"],
                    max_kv_tokens: int = 512,
                    top_k: int = 20,
                    window: int = 6,
                    quantile: int = 90):
        """
        First pass: compute JS-divergence between consecutive top-k distributions.
        High divergence → likely segment boundary.
        """
        import torch.nn.functional as F

        model, tokenizer = RTLLMTopKShift.load_model(model_name)
        model.eval()
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

        # initial prompt forward
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            if past_key_values is not None:
                past_key_values = RTLLMTopKShift.truncate_kv_dynamic(past_key_values, max_kv_tokens)

        # collect top-k distributions
        topk_probs_list = []

        while trace_ptr < len(trace_ids):
            logits = out.logits[0, -1]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_ids = torch.topk(probs, top_k)
            topk_probs = topk_probs / topk_probs.sum()  # renormalize
            topk_probs_list.append((topk_probs.cpu().to(torch.float32).numpy(), topk_ids.cpu().numpy()))

            # force true token
            true_id = trace_ids[trace_ptr]
            next_id = torch.tensor([[true_id]], device=model.device)
            decoded_so_far += tokenizer.decode([true_id], skip_special_tokens=False)
            trace_ptr += 1

            # forward step
            with torch.no_grad():
                out = model(input_ids=next_id, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
                if past_key_values is not None:
                    past_key_values = RTLLMTopKShift.truncate_kv_dynamic(past_key_values, max_kv_tokens)

        # compute JS-divergence between consecutive top-k distributions
        from scipy.spatial.distance import jensenshannon
        punc_ids = [tokenizer.get_vocab().get(p) for p in RTLLMTopKShift.PUNCTUATION]
        deltas = []

        for idx in range(1, len(topk_probs_list)):
            p_probs, p_ids = topk_probs_list[idx - 1]
            q_probs, q_ids = topk_probs_list[idx]

            # align top-k vocab: create sparse vector over union of tokens
            union_ids = np.union1d(p_ids, q_ids)
            p_vec = np.zeros(len(union_ids))
            q_vec = np.zeros(len(union_ids))

            p_map = {tid: i for i, tid in enumerate(union_ids)}
            q_map = {tid: i for i, tid in enumerate(union_ids)}
            for prob, tid in zip(p_probs, p_ids):
                p_vec[p_map[tid]] = prob
            for prob, tid in zip(q_probs, q_ids):
                q_vec[q_map[tid]] = prob

            js = jensenshannon(p_vec, q_vec)
            if trace_ids[idx - 1] in punc_ids:
                deltas.append(js)

        # normalize and threshold
        deltas = np.array(deltas)
        delta_z = (deltas - deltas.mean()) / (deltas.std() + 1e-6)
        threshold = np.percentile(delta_z, quantile)

        # segment pass
        decoded_so_far = ""
        char_cursor = 0
        boundaries = [0]
        trace_ptr = 0
        idx, jdx = 0, 0
        last_split_sentence_idx = 0
        while trace_ptr < len(trace_ids):
            true_id = trace_ids[trace_ptr]
            token_text = tokenizer.decode([true_id], skip_special_tokens=False)
            decoded_so_far += token_text
            char_cursor += len(token_text)

            if (len(trace_ids) - window > idx >= window) and trace_ids[idx - 1] in punc_ids:
                if delta_z[jdx] > threshold:  # high JS → boundary
                    boundaries.append(char_cursor)
                    last_split_sentence_idx = jdx
                jdx += 1

            trace_ptr += 1
            idx += 1

        boundaries.append(len(trace))
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
                 top_k: int = 20,
                 window: int = 15,
                 quantile: int = 90,
                 **kwargs):
        return RTLLMTopKShift._trace_pass(trace=trace,
                                          system_prompt=system_prompt,
                                          model_name=model_name,
                                          max_kv_tokens=max_kv_tokens,
                                          top_k=top_k,
                                          window=window,
                                          quantile=quantile)
