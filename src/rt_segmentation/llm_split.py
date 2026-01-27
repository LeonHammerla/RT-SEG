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




class RTLLMOffsetBased:
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
    def _segment(trace: str,
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
    def segment_with_chunk_retry(
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
    ) -> List[Tuple[int, int]]:

        all_segments: List[Tuple[int, int]] = []

        i = 0
        while i < max(0, len(trace) - 20):
            # Build chunk with carryover (carryover is CONTEXT, not to be re-segmented)
            base_chunk = trace[i:i + chunk_size]

            response = RTLLMOffsetBased._segment(
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


    @staticmethod
    def segment(wipe: bool = False):
        login_data = sdb_login()
        with Surreal(login_data["url"]) as db:
            db.signin({"username": login_data["user"], "password": login_data["pwd"]})
            db.use(login_data["ns"], login_data["db"])
            if wipe:
                db.query("REMOVE TABLE llm_offset_chunk_split;")
                db.query("DEFINE TABLE llm_offset_chunk_split SCHEMALESS;")
                db.query("DEFINE INDEX idx_llm_offset_chunk_split_id ON llm_offset_chunk_split FIELDS id;")

                db.query("REMOVE TABLE has_llm_offset_chunk_split;")
                db.query("DEFINE TABLE has_llm_offset_chunk_split SCHEMALESS TYPE RELATION IN rtrace OUT llm_offset_chunk_split;")
                db.query("DEFINE INDEX idx_rt_id ON has_llm_offset_chunk_split FIELDS id;")
                db.query("DEFINE INDEX idx_rt_in ON has_llm_offset_chunk_split FIELDS in;")
                db.query("DEFINE INDEX idx_rt_out ON has_llm_offset_chunk_split FIELDS out;")

            results = db.query("SELECT *, ->has_llm_offset_chunk_split->llm_offset_chunk_split.* as seg from rtrace where ->has_llm_offset_chunk_split->llm_offset_chunk_split == [] and string::len(rt) < 20000 ")

            for res in tqdm(results, desc=f"Segmenting traces with LLM"):
                rt = res.get("rt")
                try:
                    offsets = RTLLMOffsetBased.segment_with_chunk_retry(trace=rt,
                                                                      chunk_size=40,
                                                                      prompt="",
                                                                      system_prompt=load_prompt("system_prompt_offset"),
                                                                      model_name="Qwen/Qwen2.5-7B-Instruct")
                except Exception as e:
                    print(e)
                    continue

                split_id = RecordID("llm_sent_chunk_split", res.get("id").id)
                db.upsert(split_id, {"split": offsets})
                db.insert_relation("has_llm_sent_chunk_split", {"in": res.get("id"), "out": split_id})



class RTLLMSentBased:
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
    def _segment(trace: str,
                 prompt: str,
                 system_prompt: str,
                 model_name: Literal[
                     "Qwen/Qwen2.5-7B-Instruct-1M",
                     "mistralai/Mixtral-8x7B-Instruct-v0.1",
                     "Qwen/Qwen2.5-7B-Instruct"]
                 ):
        model, tokenizer = RTLLMSentBased.load_model(model_name)
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
    def segment_with_sentence_chunks(
            trace: str,
            chunk_size: int,
            prompt: str,
            system_prompt: str,
            model_name: Literal[
                "Qwen/Qwen2.5-7B-Instruct-1M",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "Qwen/Qwen2.5-7B-Instruct"],
            max_retry: int = 30
    ) -> List[Tuple[int, int]]:

        all_segments: List[Tuple[int, int]] = []

        offsets = list(RTLLMSentBased.load_tokenizer().span_tokenize(trace))
        # trace = nltk.sent_tokenize(trace)
        strace = [trace[tr[0]:tr[1]] for tr in offsets]
        retry = 0
        i = 0
        while i < max(0, len(strace) - 1):
            # Build chunk with carryover (carryover is CONTEXT, not to be re-segmented)
            base_chunk = strace[i:i + chunk_size]
            base_chunk_input = json.dumps({idx: sent for idx, sent in enumerate(base_chunk)})
            response = RTLLMSentBased._segment(
                base_chunk_input, "", system_prompt, model_name
            )
            # --- robust JSON parsing ---
            try:
                #print(base_chunk)
                #print(response)
                response = response.strip()
                response = response[response.find("["):response.rfind("]") + 1]
                local_segments = json.loads(response)

                if isinstance(local_segments, dict):
                    local_segments = list(local_segments.values())

                if (
                        isinstance(local_segments, list)
                        and len(local_segments) == 1
                        and all(isinstance(x, int) for x in local_segments)
                ):
                    local_segments = [local_segments]

            except Exception as e:
                print(50*"=")
                print(e)
                print(response)
                print(50 * "=")
                if retry < max_retry:
                    retry += 1
                else:
                    raise e
                continue  # malformed output → stop this chunk

            if not local_segments:
                break

            for seg in local_segments:
                all_segments.append([s + i for s in seg])


            check_seg = [s + i for s in local_segments[-1]]
            i = min(check_seg)
            if max(check_seg) >= len(strace) - 2:
                break
            else:
                del all_segments[-1]
                if len(check_seg) == chunk_size:
                    chunk_size += 5


        if max(all_segments[-1]) < len(strace) - 1:
            all_segments.append([sid for sid in range(max(all_segments[-1]), len(strace) - 1)])

        final_offsets = []
        # print(all_segments)
        for seg in all_segments:
            try:
                left_boundary = offsets[seg[0]][0]
            except IndexError:
                left_boundary = len(trace)
            try:
                right_boundary = offsets[seg[-1]][1]
            except IndexError:
                right_boundary = len(trace)

            final_offsets.append((left_boundary, right_boundary))

        corrected_final_offsets = []
        for idx, offset in enumerate(final_offsets[:-1]):
            corrected_final_offsets.append((offset[0], final_offsets[idx + 1][0]))
        corrected_final_offsets.append(final_offsets[-1])
        return corrected_final_offsets

    @staticmethod
    def segment(wipe: bool = False):
        login_data = sdb_login()
        with Surreal(login_data["url"]) as db:
            db.signin({"username": login_data["user"], "password": login_data["pwd"]})
            db.use(login_data["ns"], login_data["db"])
            if wipe:
                db.query("REMOVE TABLE llm_sent_chunk_split;")
                db.query("DEFINE TABLE llm_sent_chunk_split SCHEMALESS;")
                db.query("DEFINE INDEX idx_llm_sent_chunk_split_id ON llm_sent_chunk_split FIELDS id;")

                db.query("REMOVE TABLE has_llm_sent_chunk_split;")
                db.query("DEFINE TABLE has_llm_sent_chunk_split SCHEMALESS TYPE RELATION IN rtrace OUT llm_sent_chunk_split;")
                db.query("DEFINE INDEX idx_rt_id ON has_llm_sent_chunk_split FIELDS id;")
                db.query("DEFINE INDEX idx_rt_in ON has_llm_sent_chunk_split FIELDS in;")
                db.query("DEFINE INDEX idx_rt_out ON has_llm_sent_chunk_split FIELDS out;")

            results = db.query("SELECT *, ->has_llm_sent_chunk_split->llm_sent_chunk_split.* as seg from rtrace where ->has_llm_sent_chunk_split->llm_sent_chunk_split == [] and string::len(rt) < 20000 ")

            for res in tqdm(results, desc=f"Segmenting traces with LLM"):
                rt = res.get("rt")
                try:
                    offsets = RTLLMSentBased.segment_with_sentence_chunks(trace=rt,
                                                                      chunk_size=40,
                                                                      prompt="",
                                                                      system_prompt=load_prompt("system_prompt_sentbased"),
                                                                      model_name="Qwen/Qwen2.5-7B-Instruct")
                except Exception as e:
                    print(e)
                    continue

                split_id = RecordID("llm_sent_chunk_split", res.get("id").id)
                db.upsert(split_id, {"split": offsets})
                db.insert_relation("has_llm_sent_chunk_split", {"in": res.get("id"), "out": split_id})


class RTLLMForcedDecoderBased:
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
                sep_bias = 0.5
            else:
                sep_bias = 0.0

            true_logp = log_probs[true_id]
            sep_logp = torch.max(log_probs[sep_ids])

            gap_z = (sep_logp - true_logp - gap_mean) / gap_std
            score = alpha * gap_z + beta * sep_bias

            print(true_logp, sep_logp, score, threshold)
            if score > threshold:
                # insert boundary
                boundaries.append(char_cursor)
                next_id = torch.tensor([[sep_id]], device=model.device)
            else:
                next_id = torch.tensor([[true_id]], device=model.device)
                token_text = tokenizer.decode([true_id], skip_special_tokens=False)
                decoded_so_far += token_text
                char_cursor += len(token_text)
                trace_ptr += 1

            with torch.no_grad():
                out = model(input_ids=next_id, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
                if past_key_values is not None:
                    past_key_values = RTLLMForcedDecoderBased.truncate_kv_dynamic(past_key_values, max_kv_tokens)

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
                    alpha: float = 1.0,
                    beta: float = 2,
                    quantile: float = 90.0,
                 sep_tok: str = "<|seg|>"):
        threshold, gap_mean, gap_std = RTLLMForcedDecoderBased._trace_pass(trace=trace,
                                                                           system_prompt=system_prompt,
                                                                           model_name=model_name,
                                                                           max_kv_tokens=max_kv_tokens,
                                                                           alpha=alpha,
                                                                           beta=beta,
                                                                           quantile=quantile,
                                                                           sep_tok=sep_tok)
        print(threshold, gap_mean, gap_std)
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