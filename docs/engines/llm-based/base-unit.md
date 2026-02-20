---
layout: default
title: RTLLMSegUnitBased
parent: Downstream LLM Engines
nav_order: 3
---

# RTLLMSegUnitBased (LLM segmentation over sentence/clause indices)

## Idea

`RTLLMSegUnitBased` is a downstream LLM engine that performs segmentation **over precomputed base units** (sentences or clauses), rather than over raw character offsets.

The trace is first decomposed into base units using `SegBase.get_base_offsets(...)`. The LLM then receives a JSON-encoded list/dict of these units for a local chunk and is asked to return **segments as lists of unit indices**. These index-level segments are finally converted back into **character offsets**.

Compared to `RTLLMOffsetBased`, this approach:
- constrains the model to choose boundaries only at base-unit boundaries (often improving robustness), and
- avoids brittle character-level offset prediction.

---

## Method (high-level)

1. **Base segmentation**
   Compute base offsets:
   - `offsets = SegBase.get_base_offsets(trace, seg_base_unit=...)`
   and extract base strings:
   - `strace = [trace[s:e] for (s,e) in offsets]`

2. **Chunk the base-unit list**
   Process base units in chunks of size `chunk_size` (interpreted here as **number of base units**, not characters).

3. **Prompt the LLM with base units**
   Each chunk is encoded as JSON, mapping local indices to text:

   ```json
   {"0": "…", "1": "…", "2": "…"}

The engine calls the LLM with:

* a `system_prompt`
* a user message containing `prompt + base_chunk_input`

The model is expected to return a JSON list of segments expressed as lists of indices, e.g.:

```json
[[0, 1, 2], [3, 4], [5, 6, 7]]
```

4. **Robust parsing and retries**
   The engine attempts to parse the substring between the first `[` and last `]`, then `json.loads(...)`.
   If parsing fails, it retries up to `max_retry`.

5. **Stitch segments into global unit indices**
   Local segment indices are shifted by the current chunk offset `i`:

   * `seg_global = [s + i for s in seg_local]`

6. **Advance to next chunk**
   The engine advances `i` using the last returned segment:

   * `check_seg = [s + i for s in local_segments[-1]]`
   * `i = min(check_seg)`

   To avoid duplicating context near chunk boundaries, it may drop the last predicted segment (`del all_segments[-1]`) and re-run with adjusted chunk sizing.

7. **Convert unit-index segments to character offsets**
   For each global unit-index segment `seg = [j_1, ..., j_k]`:

   * left boundary = `offsets[j_1][0]`
   * right boundary = `offsets[j_k][1]`

   Finally, offsets are “corrected” to ensure non-overlapping spans by snapping each segment’s end to the next segment’s start.

Output:

* `corrected_final_offsets`: list of character spans
* `labels`: currently `"UNK"` for all segments

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
* Generation uses `max_new_tokens=8000`.
* Uses `device_map="auto"` and `torch_dtype="auto"`.

> As with other downstream prompting engines, segmentation is highly prompt- and model-dependent; report both for reproducibility.

---

## Key parameters

* `seg_base_unit: Literal["sent", "clause"]`
  Base unit used for the index space.

* `chunk_size: int`
  Number of base units included in each LLM call. Larger chunk sizes provide more context but increase generation length and JSON complexity.

* `system_prompt: str`
  Should enforce strict JSON output and define the expected index-segmentation format.

* `prompt: str`
  Prefix prepended to the JSON chunk input. In the current implementation `_trace_pass` is called with `prompt=""` (see Notes).

* `max_retry: int` (default: `30`)
  Maximum retries when output parsing fails.

---

## Usage

```python
from rt_seg.seg_factory import RTSeg
from rt_seg.llm_seg_unit_based import RTLLMSegUnitBased  # adjust import to your package structure

trace = "..."

system_prompt = (
    "You are a segmentation assistant. "
    "Input is a JSON dict mapping indices to text spans. "
    "Return only JSON: a list of segments, each segment is a list of integer indices, "
    "covering the chunk in order."
)

segmentor = RTSeg(engines=RTLLMSegUnitBased, seg_base_unit="sent")

offsets, labels = segmentor(
    trace,
    seg_base_unit="sent",
    chunk_size=25,
    system_prompt=system_prompt,
    prompt="",  # see note below
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_retry=30,
)

segments = [trace[s:e] for s, e in offsets]
```