---
layout: default
title: RTLLMTopKShift
parent: Probabilistic
nav_order: 4
---

# RTLLMTopKShift (Top-*k* distribution shift via Jensen–Shannon divergence)

## Idea

`RTLLMTopKShift` segments a reasoning trace by measuring **how much the model’s next-token distribution changes from one token to the next**.  
Rather than tracking only the probability of the true token (surprisal) or the overall uncertainty (entropy), this engine compares the **shape** of the predictive distribution itself—restricted to the **top-*k*** tokens for efficiency.

The underlying hypothesis is that reasoning step transitions often coincide with **distributional regime changes**: the set of likely continuations shifts, leading to high divergence between consecutive predictive distributions.

---

## Method (high-level)

Given a trace `x = (t_1, …, t_n)`:

1. **Forced decoding pass**  
   Iterate through the trace token-by-token. At each position `i`, compute the model’s next-token distribution, then force the true next token to continue the pass.

2. **Extract top-*k* predictive distributions**  
   At each step, take the top-*k* tokens from the next-token distribution:
   - `topk_ids_i`, `topk_probs_i` (renormalized to sum to 1)

3. **Compute distribution shift between consecutive steps**  
   For consecutive positions, compute the **Jensen–Shannon (JS) divergence** between the two top-*k* distributions:
   - Build a union vocabulary of the two top-*k* token sets
   - Embed each distribution into that union space (sparse vectors)
   - Compute `JS(p, q)` via `scipy.spatial.distance.jensenshannon`

4. **Punctuation anchoring**  
   Only consider a JS score as a segmentation cue when the previous token is a **hard punctuation anchor**:
   - `. ! ? \n`

5. **Normalize + threshold**  
   Z-score the JS values (`delta_z`). Set a threshold as a percentile (`quantile`) of `delta_z`.  
   Insert a boundary when:

   - `delta_z > threshold`

   (high JS divergence → likely boundary)

6. **Offsets output**  
   Convert boundary token positions into **character offsets** by decoding forced tokens and accumulating a character cursor.  
   Labels are currently returned as `"UNK"` for all segments.

---

## Models used

The engine supports (as implemented):

- `Qwen/Qwen2.5-7B-Instruct-1M`
- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`

Implementation details:
- Uses `AutoModelForCausalLM` and `AutoTokenizer`.
- Uses `tokenizer.apply_chat_template(...)` with a `system_prompt`.
- Runs with `device_map="auto"` and `torch_dtype="auto"`.

> For reproducibility, document the exact model(s) used in the reported experiments. Distribution-shift signals can differ substantially across architectures and instruction-tuning regimes.

---

## Key parameters

- `system_prompt: str`  
  System instruction used in chat formatting; impacts the predictive distribution.

- `model_name: str`  
  One of the supported model identifiers above.

- `top_k: int` (default: `20`)  
  Number of top tokens used to approximate the predictive distribution at each step.

- `quantile: int` (default: `90`)  
  Percentile threshold over z-scored JS divergences. Higher values typically yield fewer splits (selecting only the strongest shifts).

- `window: int` (default: `15`)  
  Present for API consistency; the current implementation compares *consecutive* distributions (i.e., effectively window = 1 for the JS computation).  
  The check `len(trace_ids) - window > idx >= window` gates boundary insertion away from the very beginning/end.

- `max_kv_tokens: int` (default: `512`)  
  Caps KV cache length via `DynamicCache.crop(...)`.

---

## Usage

```python
from rt_seg.seg_factory import RTSeg
from rt_seg.rt_llm_topk_shift import RTLLMTopKShift  # adjust import to your package structure

trace = "First step... Then second step... Finally conclude."

segmentor = RTSeg(
    engines=RTLLMTopKShift,
    seg_base_unit="clause",  # included for API consistency; this engine operates token-wise
)

offsets, labels = segmentor(
    trace,
    system_prompt="You are a helpful assistant that reads reasoning traces.",
    model_name="Qwen/Qwen2.5-7B-Instruct",
    top_k=20,
    quantile=90,
    max_kv_tokens=512,
)

segments = [trace[s:e] for s, e in offsets]
```