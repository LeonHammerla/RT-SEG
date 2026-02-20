---
layout: default
title: RTLLMFlatnessBreak
parent: Probabilistic Engines
nav_order: 5
---

# RTLLMFlatnessBreak (Flatness-break segmentation in surprisal)

## Idea

`RTLLMFlatnessBreak` segments a reasoning trace by detecting **distributional “regime changes”** in the model’s token-level **surprisal**.  
The core hypothesis is that coherent reasoning steps exhibit relatively stable predictability (a “flat” surprisal profile), while transitions between steps produce **abrupt shifts** in the mean (and potentially variance) of surprisal.

This engine implements a *flatness-break* detector: it compares surprisal statistics in a window before vs. after a candidate boundary.

---

## Method (high-level)

Given a trace `x = (t_1, …, t_n)` tokenized under a causal LM:

1. **Forced decoding pass (surprisal extraction)**  
   For each token position `i`, compute the log-probability of the *true next token* under the model:
   \[
   s_i = -\log p(t_i \mid t_{<i})
   \]
   The token `t_i` is then forced as the next input to continue the pass.

2. **Flatness-break score at candidate boundaries**  
   For each position `i`, compare the *previous* and *next* windows of surprisal:
   - `prev_window = s[i-window : i]`
   - `next_window = s[i : i+window]`

   Compute:
   - `mean_prev`, `mean_next`
   - `var_prev`, `var_next`

   The current implementation uses a mean-shift score normalized by prior variability:
   \[
   \text{score}_i = \frac{\mu_{\text{next}} - \mu_{\text{prev}}}{\sqrt{\sigma^2_{\text{prev}} + \epsilon}}
   \]

   *(A variance-change term is indicated in code as an optional extension.)*

3. **Candidate filtering at punctuation**  
   Scores are evaluated **only at hard punctuation anchors**, when the previous token is one of:
   - `. ! ? \n`

4. **Z-normalization + thresholding**  
   Scores are z-scored (`flatness_z`). A threshold is chosen as a percentile (`quantile`) of `flatness_z`.  
   A boundary is inserted when:
   - `flatness_z < threshold`

5. **Offsets output**  
   The engine converts boundary positions to **character offsets** by decoding tokens and accumulating a character cursor, returning `(start, end)` spans.  
   Labels are currently `"UNK"` for all segments.

---

## Models used

The engine supports (and was designed to run with) the following chat/instruction-tuned causal LMs:

- `Qwen/Qwen2.5-7B-Instruct-1M`
- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`

Implementation details:
- Uses `AutoModelForCausalLM` and `AutoTokenizer`.
- Uses `tokenizer.apply_chat_template(...)` with a `system_prompt` to contextualize the trace.
- Runs with `device_map="auto"` and `torch_dtype="auto"`.

> For reproducibility in the paper, list the exact model(s) used in experiments (and any prompt variants), as these affect surprisal dynamics.

---

## Key parameters

- `system_prompt: str`  
  System instruction used in the chat template; affects token probabilities and thus surprisal patterns.

- `model_name: str`  
  One of the supported model identifiers above.

- `window: int` (default: `15` in `_segment`, `6` in `_trace_pass`)  
  Size of the sliding windows used for pre/post statistics.

- `quantile: int` (default: `10`)  
  Percentile used to set the threshold on `flatness_z`. Changing this adjusts segmentation aggressiveness (subject to the sign convention; see Notes).

- `max_kv_tokens: int` (default: `512`)  
  Caps the KV cache context length via `DynamicCache.crop(...)` to limit memory / runtime.

---

## Usage

```python
from rt_seg.seg_factory import RTSeg
from rt_seg.rt_llm_flatness_break import RTLLMFlatnessBreak  # adjust import to your package structure

trace = "First step... Then second step... Finally conclude."

segmentor = RTSeg(
    engines=RTLLMFlatnessBreak,
    seg_base_unit="clause",  # included for API consistency; this engine operates token-wise
)

offsets, labels = segmentor(
    trace,
    system_prompt="You are a helpful assistant that reads reasoning traces.",
    model_name="Qwen/Qwen2.5-7B-Instruct",
    window=15,
    quantile=10,
    max_kv_tokens=512,
)

segments = [trace[s:e] for s, e in offsets]
```