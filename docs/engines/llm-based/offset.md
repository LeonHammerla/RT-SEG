---
layout: default
title: RTLLMOffsetBased
parent: Downstream LLM Engines
nav_order: 2
---

# RTLLMOffsetBased (Direct offset proposal via prompted LLM)

## Idea

`RTLLMOffsetBased` is a downstream LLM engine that asks an instruction-tuned model to output **explicit character offsets** for segmentation boundaries.  
Instead of operating over a base unit (sentences/clauses) and merging labels, this engine treats segmentation as a *direct structured prediction* problem: the model returns a list of `(start, end)` spans (relative to a chunk), which are then stitched into a full-trace segmentation.

This approach is useful when you want:
- segmentation that is not constrained to sentence/clause units, and
- direct compatibility with the library’s canonical output representation (character offsets).

---

## Method (high-level)

1. **Chunk the trace**
   The trace is processed in sliding chunks of size `chunk_size` to stay within model context limits.

2. **Prompt the LLM for offsets**
   For each chunk, the engine calls the LLM with:
   - a `system_prompt`
   - a user message containing `prompt + chunk_text`

   The model is expected to output a JSON list containing spans, e.g.:

   ```json
   [[0, 120], [120, 260], [260, 415]]


Offsets are interpreted as **character indices within the chunk**.

3. **Robust parsing**
   The engine attempts to parse the model output by:

   * stripping the output,
   * extracting the substring between the first `[` and last `]`,
   * loading JSON via `json.loads`.

   If the model returns a dict, values are used as segments. If a single `[start, end]` pair is returned, it is wrapped into a list.

4. **Stitch segments into global offsets**
   Each local `(a, b)` span is converted into a global offset:

   * `(i + a, i + b)` where `i` is the chunk’s starting position in the full trace.

5. **Progress via last end offset**
   After each chunk, the engine advances `i` to the end of the last predicted segment:

   * `i = all_segments[-1][1]`

6. **Finish**
   If any remainder remains, append a final segment `(i, len(trace))`.

Output:

* `offsets`: list of global `(start, end)` character spans
* `labels`: currently `"UNK"` for each segment

---

## Models used

This engine uses instruction-tuned causal LMs via:

* `AutoModelForCausalLM`
* `AutoTokenizer`

Supported model identifiers in the code:

* `Qwen/Qwen2.5-7B-Instruct-1M`
* `Qwen/Qwen2.5-7B-Instruct`
* `mistralai/Mixtral-8x7B-Instruct-v0.1`

Implementation notes:

* Prompts are formatted using `tokenizer.apply_chat_template(...)`.
* Generation uses `max_new_tokens=8000` for long structured outputs.
* Inference uses `device_map="auto"` and `torch_dtype="auto"`.

> For reproducibility, the segmentation behavior is highly prompt-dependent; report both the model name and the prompt templates used in experiments.

---

## Key parameters

* `chunk_size: int`
  Chunk length in characters. Controls granularity and context available to the model.

* `prompt: str`
  Prefix prompt inserted before the chunk text. In the current implementation, this parameter is passed into `_segment` but `_trace_pass` is called with an empty prompt (`""`). If you intend to use `prompt`, ensure it is forwarded correctly (see Notes).

* `system_prompt: str`
  System instruction. Should enforce JSON-only output and specify offset conventions.

* `max_retries_per_chunk: int` (default: `10`)
  Present in the signature but not currently used in the implementation.

* `margin: int` (default: `200`)
  Present in the signature but not currently used in the implementation (often used for overlap/carryover).

---

## Usage

```python
from rt_seg.seg_factory import RTSeg
from rt_seg.llm_offset_based import RTLLMOffsetBased  # adjust import to your package structure

trace = "..."

system_prompt = (
    "You are a segmentation assistant. "
    "Return only JSON: a list of [start, end] character offsets for coherent segments."
)

segmentor = RTSeg(engines=RTLLMOffsetBased)

offsets, labels = segmentor(
    trace,
    chunk_size=4000,
    system_prompt=system_prompt,
    prompt="",  # see note below
    model_name="Qwen/Qwen2.5-7B-Instruct",
)

segments = [trace[s:e] for s, e in offsets]
```