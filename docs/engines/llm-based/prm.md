---
layout: default
title: RTPRMBase
parent: Downstream LLM Engines
nav_order: 4
---

# RTPRMBase (PRM-based segmentation via step reward shifts)

## Idea

`RTPRMBase` segments reasoning traces using a **Process Reward Model (PRM)**: instead of relying on next-token probabilities (surprisal/entropy) or semantic similarity, it uses a model trained to score the *quality* of intermediate reasoning steps.

The core hypothesis is that **boundaries between coherent reasoning steps** coincide with **changes in PRM step-level reward**. When the local reward signal shifts sharply, the engine inserts a segment boundary.

This is particularly well-suited for mathematical reasoning traces where step-wise quality signals are meaningful and stable.

---

## Model used

This engine uses the PRM checkpoint:

- `Qwen/Qwen2.5-Math-7B-PRM800K`

Loaded with:

- `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`
- `AutoModel.from_pretrained(..., trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16, use_cache=False)`

The implementation uses special step separators:

- `<extra_0>` to separate candidate “steps” inside the model input.

---

## Method (high-level)

Given:
- a problem statement `problem`
- a reasoning trace split into base units (sentences/clauses)

1. **Base segmentation**
   Compute base offsets via:
   - `SegBase.get_base_offsets(trace, seg_base_unit=...)`
   and extract base strings `u_i`.

2. **Chunked PRM inference**
   For scalability, the base units are processed in chunks (`chunk_size`).  
   For each chunk, construct a chat-style input with:
   - system instruction: “reason step by step…”
   - user query: `problem`
   - assistant response: base units joined by `<extra_0>`

   Then run the PRM model forward and extract step-level scores.

3. **Extract step rewards at separators**
   The model output logits are converted into probabilities, then masked to positions where the input token equals the separator ID (`<extra_0>`).  
   The engine interprets the **positive-class probability** at each separator as the step reward.

4. **Windowed reward shift**
   Let `scores[i]` be the PRM reward for base unit `i`.  
   Compute a local delta using a window on each side:

   - `prev = mean(scores[i-window : i])`
   - `next = mean(scores[i : i+window])`
   - `delta[i] = next - prev`

   Then:
   - take magnitude `|delta|`
   - z-normalize to obtain `delta_z`
   - choose a global threshold as a percentile (`quantile`) of `delta_z`

5. **Warmup-scaled threshold (early-trace stabilization)**
   A *local threshold* is used to avoid overly aggressive splitting near the beginning:

   \[
   \tau_i = \tau \cdot \left(0.5 + 0.5 \cdot \min(i / \text{warmup}, 1)\right)
   \]

6. **Boundary insertion**
   Insert a boundary when:
   - `delta_z[i] > local_threshold`, and
   - the current segment is not trivially short (`offsets[i][1] - current_offset > 4`)

7. **Offsets output**
   Output merged character spans as `(start, end)` offsets.  
   Labels are returned as `"UNK"`.

---

## Key parameters

- `problem: str`  
  The problem statement provided to the PRM.  
  **Important:** PRM scoring is conditioned on the query. For reproducibility, supply the original problem text whenever available.

- `seg_base_unit: Literal["sent", "clause"]`  
  Base units used as candidate steps.

- `chunk_size: int` (default: `50`)  
  Number of base units per PRM forward pass.

- `window: int` (default: `4`)  
  Window size for local mean calculation on each side of a candidate boundary.

- `quantile: int` (default: `60`)  
  Percentile used to threshold z-scored delta magnitudes.

---

## Usage

```python
from rt_seg.seg_factory import RTSeg
from rt_seg.prm_base import RTPRMBase  # adjust import to your package structure

problem = "Compute the value of ..."

trace = "..."

segmentor = RTSeg(
    engines=RTPRMBase,
    seg_base_unit="sent",
)

offsets, labels = segmentor(
    trace,
    seg_base_unit="sent",
    problem=problem,
    chunk_size=50,
    window=4,
    quantile=60,
)

segments = [trace[s:e] for s, e in offsets]