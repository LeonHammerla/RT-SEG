---
layout: default
title: RTLLMSurprisal
parent: Probabilistic Engines
nav_order: 1
---

# RTLLMSurprisal (Surprisal-shift boundary detection)

## Idea

`RTLLMSurprisal` segments a reasoning trace by detecting **local shifts in token-level surprisal** under a causal language model.  
Surprisal is the negative log-probability of the *observed* next token. The core hypothesis is that **transitions between reasoning steps** correspond to changes in predictability: after a sentence boundary, the model’s expected continuation can shift, leading to systematic differences in surprisal statistics.

This engine is **forced-decoding based** (no generation): it computes surprisal for each token in the provided trace and inserts boundaries based on windowed surprisal differences.

---

## Method (high-level)

Given a trace `x = (t_1, …, t_n)` tokenized under a causal LM:

1. **Forced decoding pass (surprisal extraction)**  
   For each token position `i`, compute:
   \[
   s_i = -\log p(t_i \mid t_{<i})
   \]
   Then force the true token `t_i` as the next input and continue.

2. **Windowed surprisal shift**  
   For candidate boundary positions `i`, compute the difference between mean surprisal before and after `i`:
   - `prev = mean(s[i-window : i])`
   - `next = mean(s[i : i+window])`
   - `delta = next - prev`

3. **Candidate filtering at punctuation**  
   Deltas are computed only when the previous token corresponds to a **hard punctuation anchor**:
   - `. ! ? \n`

4. **Z-normalization + thresholding**  
   The deltas are z-scored (`delta_z`). A threshold is set via a percentile (`quantile`) over `delta_z`.  
   A segmentation boundary is inserted when:
   - `delta_z < threshold`

5. **Offsets output**  
   Candidate token boundaries are converted into character offsets by decoding tokens and accumulating a character cursor.  
   Returns `(start, end)` spans and assigns `"UNK"` labels by default.

---

## Models used

The engine supports the following causal LMs (as implemented):

- `Qwen/Qwen2.5-7B-Instruct-1M`
- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`

Implementation details:
- Uses `AutoModelForCausalLM` and `AutoTokenizer`.
- Wraps the trace using `tokenizer.apply_chat_template(...)` with a provided `system_prompt`.
- Runs with `device_map="auto"` and `torch_dtype="auto"`.

> For reproducibility, list the exact model(s) used in the reported experiments and the `system_prompt` variant, as these can affect surprisal statistics.

---

## Key parameters

- `system_prompt: str`  
  System instruction used in chat formatting; affects token probabilities.

- `model_name: str`  
  One of the supported model identifiers above.

- `window: int` (default: `15` in `_segment`, `6` in `_trace_pass`)  
  Window size for pre/post mean surprisal estimation.

- `quantile: int` (default: `10`)  
  Percentile used to set the threshold over `delta_z`. Adjusts segmentation aggressiveness (subject to sign convention; see Notes).

- `max_kv_tokens: int` (default: `512`)  
  Maximum KV cache length. Past this, the cache is cropped (Transformers `DynamicCache`), limiting memory growth.

---

## Usage

```python
from rt_seg.seg_factory import RTSeg
from rt_seg.rt_llm_surprisal import RTLLMSurprisal  # adjust import to your package structure

trace = "First step... Then second step... Finally conclude."

segmentor = RTSeg(
    engines=RTLLMSurprisal,
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