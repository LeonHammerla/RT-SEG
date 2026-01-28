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

