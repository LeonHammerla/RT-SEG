import json
import os
import random
from collections import Counter
from functools import lru_cache
from typing import List, Dict
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
from transformers import AutoModelForCausalLM, AutoTokenizer


@lru_cache(maxsize=1)
def bp():
    return os.path.realpath(os.path.join(os.path.realpath(__file__), "../../.."))


@lru_cache(maxsize=1)
def sdb_login():
    with open(f"{bp()}/data/sdb_login.json", "r") as f:
        config = json.load(f)
    return config


class RTNewLine:
    @staticmethod
    def paragraph_ranges_regex(text: str) -> list[tuple[int, int]]:
        # Find positions after each \n\n or at start
        positions = [m.end() for m in re.finditer(r'\n\n|\A', text)]
        # Pair consecutive positions
        return list(zip(positions, positions[1:] + [len(text)]))

    @staticmethod
    def new_line_split():
        login_data = sdb_login()
        with Surreal(login_data["url"]) as db:
            db.signin({"username": login_data["user"], "password": login_data["pwd"]})
            db.use(login_data["ns"], login_data["db"])
            db.query("REMOVE TABLE newline_split;")
            db.query("DEFINE TABLE newline_split SCHEMALESS;")
            db.query("DEFINE INDEX idx_newline_split_id ON newline_split FIELDS id;")

            db.query("REMOVE TABLE has_newline_split;")
            db.query("DEFINE TABLE has_newline_split SCHEMALESS TYPE RELATION IN rtrace OUT newline_split;")
            db.query("DEFINE INDEX idx_rt_id ON has_newline_split FIELDS id;")
            db.query("DEFINE INDEX idx_rt_in ON has_newline_split FIELDS in;")
            db.query("DEFINE INDEX idx_rt_out ON has_newline_split FIELDS out;")

            results = db.query("SELECT * from rtrace")

            for res in results:
                rt = res.get("rt")
                offsets = RTNewLine.paragraph_ranges_regex(rt)

                split_id = RecordID("newline_split", res.get("id").id)
                db.upsert(split_id, {"split": offsets})
                db.insert_relation("has_newline_split", {"in": res.get("id"), "out": split_id})


class RTRuleBased:
    INFERENCE_MARKERS = [
        r"\btherefore\b", r"\bthus\b", r"\bhence\b", r"\bso\b",
        r"\bfollows that\b", r"\bimplies\b", r"\bwe conclude\b"
    ]

    CONTRAST_MARKERS = [
        r"\bbut\b", r"\bhowever\b", r"\bthough\b", r"\binstead\b",
        r"\bactually\b", r"\bnevertheless\b", r"\bon the other hand\b"
    ]

    REVISION_MARKERS = [
        r"\bthis is wrong\b", r"\bthat was wrong\b", r"\bi was mistaken\b",
        r"\blet me reconsider\b", r"\bwait\b"
    ]

    GOAL_MARKERS = [
        r"\bnow we\b", r"\blet us\b", r"\bnext\b",
        r"\bconsider\b", r"\bwe need to\b", r"\bthe goal\b"
    ]

    FINAL_MARKERS = [
        r"\bthe answer is\b", r"\btherefore the answer\b",
        r"\bin conclusion\b", r"\bfinal answer\b"
    ]

    _sent_tokenizer = PunktSentenceTokenizer()

    # -----------------------------
    # Marker helpers
    # -----------------------------
    @staticmethod
    def has_marker(text: str, markers: List[str]) -> bool:
        text = text.lower()
        return any(re.search(m, text) for m in markers)

    @staticmethod
    def introduces_inference(sentence: str) -> bool:
        return RTRuleBased.has_marker(sentence, RTRuleBased.INFERENCE_MARKERS)

    @staticmethod
    def goal_shift(sentence: str) -> bool:
        return RTRuleBased.has_marker(sentence, RTRuleBased.GOAL_MARKERS)

    @staticmethod
    def is_final(sentence: str) -> bool:
        return RTRuleBased.has_marker(sentence, RTRuleBased.FINAL_MARKERS)

    @staticmethod
    def starts_new_segment(sentence: str) -> bool:
        return (
            RTRuleBased.introduces_inference(sentence)
            or RTRuleBased.has_marker(sentence, RTRuleBased.CONTRAST_MARKERS)
            or RTRuleBased.has_marker(sentence, RTRuleBased.REVISION_MARKERS)
            or RTRuleBased.goal_shift(sentence)
            or RTRuleBased.is_final(sentence)
        )

    # -----------------------------
    # Sentence spans (core change)
    # -----------------------------
    @staticmethod
    def sentence_spans(text: str) -> List[tuple]:
        """
        Returns a list of (start, end) sentence spans into the original text.
        """
        return list(RTRuleBased._sent_tokenizer.span_tokenize(text))

    # -----------------------------
    # Segment offsets
    # -----------------------------
    @staticmethod
    def segment(text: str) -> List[Dict[str, int]]:
        spans = RTRuleBased.sentence_spans(text)

        if not spans:
            return []

        segments = []
        current_start = spans[0][0]

        for i, (sent_start, sent_end) in enumerate(spans):
            sentence_text = text[sent_start:sent_end]

            if i > 0 and RTRuleBased.starts_new_segment(sentence_text):
                # close previous segment exactly at this sentence start
                segments.append((current_start, sent_start))
                current_start = sent_start

        # final segment goes to end of text
        segments.append((current_start, len(text)))

        return segments

    @staticmethod
    def rule_split():
        login_data = sdb_login()
        with Surreal(login_data["url"]) as db:
            db.signin({"username": login_data["user"], "password": login_data["pwd"]})
            db.use(login_data["ns"], login_data["db"])
            db.query("REMOVE TABLE rule_split;")
            db.query("DEFINE TABLE rule_split SCHEMALESS;")
            db.query("DEFINE INDEX idx_rule_split_id ON rule_split FIELDS id;")

            db.query("REMOVE TABLE has_rule_split;")
            db.query("DEFINE TABLE has_rule_split SCHEMALESS TYPE RELATION IN rtrace OUT rule_split;")
            db.query("DEFINE INDEX idx_rt_id ON has_rule_split FIELDS id;")
            db.query("DEFINE INDEX idx_rt_in ON has_rule_split FIELDS in;")
            db.query("DEFINE INDEX idx_rt_out ON has_rule_split FIELDS out;")

            results = db.query("SELECT * from rtrace")

            for res in results:
                rt = res.get("rt")
                offsets = RTRuleBased.segment(rt)

                split_id = RecordID("rule_split", res.get("id").id)
                db.upsert(split_id, {"split": offsets})
                db.insert_relation("has_rule_split", {"in": res.get("id"), "out": split_id})


class RTQwen2_5Based:
    SYSTEM_PROMPT = """You are a text segmentation assistant.

Your task is to segment an input text into contiguous reasoning segments and return only their character offsets.

You must not:
- add, remove, rewrite, or explain any content
- assign labels or categories
- interpret correctness
- change segmentation granularity beyond what is present

You must:
- segment the text wherever the function of the reasoning clearly changes (e.g. description → analysis → conclusion)
- preserve the original order
- ensure that all characters are covered exactly once
- use character offsets relative to the full input string

OFFSET RULES:
- Use 0-based character indexing
- Format: [start_offset, end_offset) (end offset is exclusive)
- Segments must be contiguous and non-overlapping

OUTPUT FORMAT:
Return only a JSON array of offset pairs, nothing else:
{
  [start, end],
  [start, end],
  ...
}"""
    PROMPT = "Segment the following text into reasoning segments according to the system instructions.\n\nTEXT TO SEGMENT (use exact character positions and return the offsets in json format):"
    @staticmethod
    @lru_cache(maxsize=1)
    def load_model():
        model_name = "Qwen/Qwen2.5-7B-Instruct-1M"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    @staticmethod
    def qwen2_5_segment(trace: str, prompt: str, system_prompt: str):
        model, tokenizer = RTQwen2_5Based.load_model()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{prompt}\n\n{trace}"}
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


if __name__ == "__main__":
    # new_line_split()
    # RTRuleBased.rule_split()
    trace = """
We assume A implies B and B implies C.
Therefore C holds.
But this assumption is incorrect.
Let me reconsider the premises.
The answer is C.
"""
    trace = """Okay, let's try to figure out this problem. The question is asking for the sum of all integer bases b > 9 where 17_b is a divisor of 97_b. Hmm, first I need to understand what these numbers mean in different bases.

So, in base b, the number 17_b would be equal to 1*b^1 + 7*b^0, which simplifies to b + 7. Similarly, 97_b is 9*b^1 + 7*b^0, so that's 9b + 7. The problem states that 17_b divides 97_b, meaning that (9b + 7) divided by (b + 7) should be an integer. So, I need to find all bases b > 9 where (9b + 7)/(b + 7) is an integer.

Let me write that division as a fraction: (9b + 7)/(b + 7). Maybe I can simplify this expression. Let's try polynomial division or see if I can rewrite it. Let me see:

If I divide 9b + 7 by b + 7, maybe I can do some algebra. Let's see:

Let me consider 9b + 7 divided by b + 7. Let's write it as 9*(b + 7) - 63 + 7 divided by (b + 7). Wait, that might not be helpful. Alternatively, maybe express the numerator in terms of the denominator.

Let me think. Let me write 9b + 7 as 9*(b + 7) - 63 + 7. Wait, 9*(b + 7) is 9b + 63. So 9b +7 = 9*(b +7) - 56. Therefore, (9b +7)/(b +7) = 9 - 56/(b +7). So the expression simplifies to 9 - 56/(b +7). For this to be an integer, 56/(b +7) must be an integer. Therefore, (b +7) must be a divisor of 56.

So, since b > 9, then b +7 must be a positive divisor of 56, and also since b +7 must be greater than 9 +7 = 16? Wait, no. Wait, b is greater than 9, so b +7 is greater than 16. Therefore, we need to find all positive divisors of 56 that are greater than 16, and then subtract 7 from them to get b.

Wait, but let me check. Let me list all divisors of 56. The divisors of 56 are 1, 2, 4, 7, 8, 14, 28, 56. But since b +7 must be greater than 16 (since b >9), then the possible divisors are 28, 56. Because 14 is 14, which would make b +7=14 → b=7, but b must be greater than 9. So 14 is too small. Similarly, 8 and 7 are too small. So the only divisors of 56 greater than 16 are 28 and 56. Therefore, b +7 =28 or 56. Therefore, b=21 or b=56-7=49. Wait, 56-7 is 49. So the possible bases are 21 and 49. Therefore, the sum is 21 +49=70.

Wait, but let me check if that's correct. Let me verify each case.

First, for b=21: 17_b is 1*21 +7=28. 97_b is 9*21 +7=198. 198 divided by 28 is 7. So 198/28=7. So yes, that works.

For b=49: 17_b is 1*49 +7=56. 97_b is 9*49 +7=441 +7=448. 448 divided by 56 is 8. So that's also an integer. So both cases work.

Are there any other divisors? Let me check again. The divisors of 56 are 1,2,4,7,8,14,28,56. So the ones greater than 16 are 28 and 56. So yes, only those two. Therefore, the answer is 21 +49=70. So the sum is 70.

But wait, let me make sure there are no other divisors. For example, if b +7 is a divisor of 56, but maybe negative divisors? But since b>9, b+7>16, so divisors have to be positive. So no negative divisors. So yes, only 28 and 56. So the answer is 21 +49=70.

Therefore, the final answer is 70. Let me check if that's correct. Yes, seems right."""

    res = RTNewLine.paragraph_ranges_regex(trace)
    print(res)
    for idx, r in enumerate(res):
        print(trace[r[0]:r[1]])

    res = RTRuleBased.sentence_spans(trace)
    print(res)
    for idx, r in enumerate(res):
        print(trace[r[0]:r[1]])

    res = RTQwen2_5Based.qwen2_5_segment(trace, RTQwen2_5Based.PROMPT, RTQwen2_5Based.SYSTEM_PROMPT)
    print(res)
    for idx, r in enumerate(res):
        print(trace[r[0]:r[1]])


