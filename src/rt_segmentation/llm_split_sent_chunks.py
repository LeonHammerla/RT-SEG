import json
from functools import lru_cache
from typing import List, Dict, Tuple, Literal, Any
from typing import List
from nltk.tokenize import PunktSentenceTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .seg_base import SegBase


class RTLLMSegUnitBased(SegBase):
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
        model, tokenizer = RTLLMSegUnitBased.load_model(model_name)
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
            seg_base_unit: Literal["sent", "clause"],
            chunk_size: int,
            prompt: str,
            system_prompt: str,
            model_name: Literal[
                "Qwen/Qwen2.5-7B-Instruct-1M",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "Qwen/Qwen2.5-7B-Instruct"],
            max_retry: int = 30,
            **kwargs
    ) -> tuple[list[Any], list[str]]:

        all_segments: List[Tuple[int, int]] = []

        offsets = SegBase.get_base_offsets(trace, seg_base_unit=seg_base_unit)

        # trace = nltk.sent_tokenize(trace)
        strace = [trace[tr[0]:tr[1]] for tr in offsets]
        retry = 0
        i = 0
        while i < max(0, len(strace) - 1):
            # Build chunk with carryover (carryover is CONTEXT, not to be re-segmented)
            base_chunk = strace[i:i + chunk_size]
            base_chunk_input = json.dumps({idx: sent for idx, sent in enumerate(base_chunk)})
            response = RTLLMSegUnitBased._trace_pass(
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
        return corrected_final_offsets, ["UNK" for _ in corrected_final_offsets]
