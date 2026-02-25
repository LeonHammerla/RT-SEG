from functools import lru_cache
from typing import List, Dict, Tuple, Literal, Any
import torch
from nltk.tokenize import PunktSentenceTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import numpy as np

from .seg_base import SegBase


class RTLLMEntropy(SegBase):
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
    def _trace_pass(trace: str,
                    system_prompt: str,
                    model_name: Literal[
                        "Qwen/Qwen2.5-7B-Instruct-1M",
                        "mistralai/Mixtral-8x7B-Instruct-v0.1",
                        "Qwen/Qwen2.5-7B-Instruct"],
                    max_kv_tokens: int = 512,
                    window: int = 6,
                    quantile: int = 1):
        """
        First pass: compute normalized "gap" scores for separator insertion.
        """
        model, tokenizer = RTLLMEntropy.load_model(model_name
                                                              )
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
                past_key_values = RTLLMEntropy.truncate_kv_dynamic(past_key_values, max_kv_tokens)

        scores = []
        while trace_ptr < len(trace_ids):
            logits = out.logits[0, -1]
            probs = torch.softmax(logits, dim=-1)
            true_id = trace_ids[trace_ptr]
            entropy = -torch.sum(probs * torch.log(probs + 1e-12))
            scores.append(entropy.item())
            # force true token
            next_id = torch.tensor([[true_id]], device=model.device)
            decoded_so_far += tokenizer.decode([true_id], skip_special_tokens=False)
            trace_ptr += 1

            # forward step
            with torch.no_grad():
                out = model(input_ids=next_id, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
                if past_key_values is not None:
                    past_key_values = RTLLMEntropy.truncate_kv_dynamic(past_key_values, max_kv_tokens)

        assert len(trace_ids) == len(scores)
        deltas = []
        punc_ids = [tokenizer.get_vocab().get(p) for p in RTLLMEntropy.PUNCTUATION]
        for idx in range(window, len(scores) - window):
            prev = np.mean(scores[idx - window:idx])
            nex = np.mean(scores[idx:idx + window])
            if trace_ids[idx - 1] in punc_ids:
                deltas.append(nex - prev)

        delta_z = (np.array(deltas) - np.mean(deltas)) / (np.std(deltas) + 1e-6)
        threshold = np.percentile(delta_z, q=quantile)
        decoded_so_far = ""
        char_cursor = 0
        boundaries = [0]
        trace_ptr = 0
        idx, jdx = 0, 0
        last_split_sentence_idx = 0
        while trace_ptr < len(trace_ids):
            true_id = trace_ids[trace_ptr]
            # print(true_logp, sep_logp, score, threshold)
            if (len(trace_ids) - window > idx >= window) and trace_ids[idx - 1] in punc_ids:
                # d = jdx - last_split_sentence_idx
                # multiplier = 2 - min(d / 5, 1)
                # dynamic_threshold = threshold * multiplier
                print(delta_z[jdx], threshold, delta_z[jdx] < threshold)
                # print(delta_z[jdx], dynamic_threshold, delta_z[jdx] < dynamic_threshold)
                if delta_z[jdx] < threshold:
                    boundaries.append(char_cursor)
                    last_split_sentence_idx = jdx
                jdx += 1

            token_text = tokenizer.decode([true_id], skip_special_tokens=False)
            decoded_so_far += token_text
            char_cursor += len(token_text)
            trace_ptr += 1
            idx += 1

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
                    window: int = 30,
                 quantile: int = 10,
                 **kwargs):
        offsets = RTLLMEntropy._trace_pass(trace=trace,
                                          system_prompt=system_prompt,
                                          model_name=model_name,
                                          max_kv_tokens=max_kv_tokens,
                                          window=window,
                                          quantile=quantile)
        return offsets, ["UNK" for _ in offsets]

