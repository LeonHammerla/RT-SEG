from functools import lru_cache
from typing import List, Dict, Tuple, Literal, Any
import torch
from nltk.tokenize import PunktSentenceTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import numpy as np

from .seg_base import SegBase


class RTLLMFlatnessBreak(SegBase):
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
        First pass: detect segmentation boundaries using flatness-break in surprisal.
        """
        model, tokenizer = RTLLMFlatnessBreak.load_model(model_name)
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

        # forward prompt
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            if past_key_values is not None:
                past_key_values = RTLLMFlatnessBreak.truncate_kv_dynamic(past_key_values, max_kv_tokens)

        surprisal = []
        while trace_ptr < len(trace_ids):
            logits = out.logits[0, -1]
            log_probs = torch.log_softmax(logits, dim=-1)
            true_id = trace_ids[trace_ptr]
            true_logp = log_probs[true_id]
            surprisal.append(-true_logp.item())

            # force token
            next_id = torch.tensor([[true_id]], device=model.device)
            decoded_so_far += tokenizer.decode([true_id], skip_special_tokens=False)
            trace_ptr += 1

            with torch.no_grad():
                out = model(input_ids=next_id, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
                if past_key_values is not None:
                    past_key_values = RTLLMFlatnessBreak.truncate_kv_dynamic(past_key_values, max_kv_tokens)

        # --- FLATNESS BREAK DETECTION ---
        surprisal = np.array(surprisal)
        punc_ids = [tokenizer.get_vocab().get(p) for p in RTLLMFlatnessBreak.PUNCTUATION]

        # rolling mean and variance
        mean_window = window
        flatness_scores = []
        for i in range(mean_window, len(surprisal) - mean_window):
            # previous window
            prev_window = surprisal[i - mean_window:i]
            next_window = surprisal[i:i + mean_window]

            # compute mean and variance shift
            mean_prev = np.mean(prev_window)
            mean_next = np.mean(next_window)
            var_prev = np.var(prev_window)
            var_next = np.var(next_window)

            # score: large shift in mean relative to prior std
            score = ((mean_next - mean_prev) / (np.sqrt(var_prev + 1e-6)))
            # can also include variance increase if desired:
            # score += (var_next - var_prev) / (var_prev + 1e-6)

            # only consider sentence boundaries
            if trace_ids[i - 1] in punc_ids:
                flatness_scores.append(score)

        flatness_scores = np.array(flatness_scores)
        score_mean = flatness_scores.mean()
        score_std = flatness_scores.std() + 1e-6
        flatness_z = (flatness_scores - score_mean) / score_std
        threshold = np.percentile(flatness_z, q=quantile)

        # --- SECOND PASS: SEGMENTATION ---
        decoded_so_far = ""
        char_cursor = 0
        boundaries = [0]
        trace_ptr = 0
        idx, jdx = 0, 0
        last_split_sentence_idx = 0

        while trace_ptr < len(trace_ids):
            true_id = trace_ids[trace_ptr]
            token_text = tokenizer.decode([true_id], skip_special_tokens=False)

            # check for boundary only at punctuation
            if (len(trace_ids) - mean_window > idx >= mean_window) and trace_ids[idx - 1] in punc_ids:
                if flatness_z[jdx] < threshold:
                    boundaries.append(char_cursor)
                    last_split_sentence_idx = jdx
                jdx += 1

            decoded_so_far += token_text
            char_cursor += len(token_text)
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
                    window: int = 15,
                 quantile: int = 10,
                 **kwargs):
        offsets = RTLLMFlatnessBreak._trace_pass(trace=trace,
                                              system_prompt=system_prompt,
                                              model_name=model_name,
                                              max_kv_tokens=max_kv_tokens,
                                              window=window,
                                              quantile=quantile)
        return offsets, ["UNK" for _ in offsets]
