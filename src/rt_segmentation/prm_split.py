from functools import lru_cache
from typing import List, Dict, Tuple, Literal, Any
import torch
from typing import List
from nltk.tokenize import PunktSentenceTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import numpy as np

from .seg_base import SegBase


class RTPRMBase(SegBase):
    PUNCTUATION = {".", "!", "?", "\n"}
    SOFT_PUNCTUATION = {",", "\t", ";", ":"}

    @staticmethod
    @lru_cache(maxsize=1)
    def load_tokenizer():
        return PunktSentenceTokenizer()

    @staticmethod
    @lru_cache(maxsize=1)
    def load_model():
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B-PRM800K",
                                                  trust_remote_code=True)
        model = AutoModel.from_pretrained(
            "Qwen/Qwen2.5-Math-7B-PRM800K",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=False,  # <--- FIX?
        ).eval()

        return model, tokenizer

    @staticmethod
    def make_step_rewards(logits, token_masks):
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels

        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]  # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res

    @staticmethod
    def model_inference(problem: str,
                        sentences: List[str]):
        model, tokenizer = RTPRMBase.load_model()

        data = {
            "system": "Please reason step by step, and put your final answer within \\boxed{}.",
            "query": problem,
            "response": sentences
        }

        messages = [
            {"role": "system", "content": data['system']},
            {"role": "user", "content": data['query']},
            {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
        ]
        conversation_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        input_ids = tokenizer.encode(
            conversation_str,
            return_tensors="pt",
        ).to(model.device)

        outputs = model(input_ids=input_ids)

        step_sep_id = tokenizer.encode("<extra_0>")[0]
        token_masks = (input_ids == step_sep_id)
        step_reward = RTPRMBase.make_step_rewards(outputs[0], token_masks)
        return step_reward

    @staticmethod
    def chunk_list(lst, n):
        return [lst[i:i + n] for i in range(0, len(lst), n)]

    @staticmethod
    def _segment(trace: str,
                 seg_base_unit: Literal["sent", "clause"],
                 problem: str = "The problem is unknown, please reason step by step.",
                 chunk_size: int = 50,
                 window: int = 4,
                 quantile: int = 60,
                 **kwargs):
        all_segments: List[Tuple[int, int]] = []

        offsets = SegBase.get_base_offsets(trace, seg_base_unit=seg_base_unit)

        strace = [trace[tr[0]:tr[1]] for tr in offsets]

        scores = []
        for chunk in RTPRMBase.chunk_list(strace, chunk_size):
            scores.extend(RTPRMBase.model_inference(problem, chunk)[0])
        assert len(scores) == len(offsets), "Something went wrong with the model inference."

        deltas = []

        for idx in range(len(scores)):
            prev_slice = scores[max(0, idx - window):idx]
            next_slice = scores[idx:min(len(scores), idx + window)]

            prev = np.mean(prev_slice) if prev_slice else scores[idx]
            nex = np.mean(next_slice) if next_slice else scores[idx]

            deltas.append(nex - prev)

        delta_mag = abs(np.array(deltas))
        delta_z = (delta_mag - delta_mag.mean()) / (delta_mag.std() + 1e-6)
        threshold = np.percentile(delta_z, q=quantile)

        current_offset = 0
        warmup = 5
        for idx in range(len(offsets) - 1):
            local_threshold = threshold * (0.5 + 0.5 * min(idx / warmup, 1.0))
            if delta_z[idx] > local_threshold and offsets[idx][1] - current_offset > 4:
                all_segments.append((current_offset, offsets[idx][1]))
                current_offset = offsets[idx + 1][1]

        all_segments.append((current_offset, len(trace)))
        return all_segments, ["UNK" for _ in all_segments]
