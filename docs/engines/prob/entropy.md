---
layout: default
title: RTLLMEntropy
parent: Probabilistic Engines
nav_order: 2
---

# RTLLMEntropy (Entropy-based boundary detection)

## Idea

`RTLLMEntropy` segments a reasoning trace by identifying **local increases/decreases in the language model’s predictive uncertainty**.  
Intuitively, when the model transitions between reasoning steps (e.g., after a sentence boundary), the distribution over next tokens can change abruptly. This engine operationalizes that via **token-level entropy** computed under *forced decoding* of the trace.

The method is **model-driven** but does not require generating text: it runs a single pass over the trace and computes an uncertainty signal.

---

## Method (high-level)

Given a trace `x = (t_1, …, t_n)` tokenized under a causal LM:

1. **Forced decoding pass**  
   For each position `i`, compute the next-token distribution `p(· | t_<i)` and its entropy:
   \[
   H_i = - \sum_v p(v|t_{<i}) \log p(v|t_{<i})
   \]
   The true next token `t_i` is then **forced** as input for the next step.

2. **Local window comparison**  
   For candidate boundary positions `i`, compare the mean entropy in a **preceding** and **following** window:
   - `prev = mean(H_{i-window : i})`
   - `next = mean(H_{i : i+window})`
   - `delta = next - prev`

3. **Candidate filtering at punctuation**  
   Deltas are computed **only at punctuation anchors** (hard punctuation), i.e. when the previous token corresponds to one of:
   - `. ! ? \n`

4. **Z-normalization + thresholding**  
   The delta values are z-scored (`delta_z`). A threshold is set as a percentile (`quantile`) of `delta_z`.  
   A boundary is inserted when:

   - `delta_z < threshold`

   (i.e., the entropy shift is sufficiently “low” relative to the distribution—see Notes below on interpretation.)

5. **Offsets output**  
   The boundary token positions are converted into **character offsets** and returned as `(start, end)` segments.  
   Labels are currently returned as `"UNK"` for all segments.

---

## Models used

The engine supports (and was designed to run with) the following chat/instruction-tuned causal LMs:

- `Qwen/Qwen2.5-7B-Instruct-1M`
- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`

Implementation details:
- Uses `AutoModelForCausalLM` and `AutoTokenizer`.
- Uses `tokenizer.apply_chat_template(...)` to wrap the trace with a `system_prompt`.
- Runs with `device_map="auto"` and `torch_dtype="auto"`.

> If your experiments used a subset of these models, list the exact ones in the paper/docs for reproducibility.

---

## Key parameters

- `system_prompt: str`  
  System instruction used in the chat template. This affects token distributions and therefore the entropy signal.

- `model_name: str`  
  One of the supported model identifiers above.

- `window: int` (default: `30` in `_segment`, `6` in `_trace_pass`)  
  Window size used to compute local mean entropies before/after a candidate boundary.

- `quantile: int` (default: `10`)  
  Percentile used to set the threshold over `delta_z`. Lower values typically imply a more conservative threshold (fewer splits), higher values typically imply more permissive splitting—subject to the sign convention (see Notes).

- `max_kv_tokens: int` (default: `512`)  
  Maximum length of cached context for the LM KV cache. If exceeded, the cache is truncated.

---

## Usage

```python
from rt_seg.seg_factory import RTSeg
from rt_seg.rt_llm_entropy import RTLLMEntropy  # adjust import to your package structure

trace = "First step... Then second step... Finally conclude."

segmentor = RTSeg(
    engines=RTLLMEntropy,
    seg_base_unit="clause",   # base unit may be ignored by this engine; included for API consistency
)

offsets, labels = segmentor(
    trace,
    system_prompt="You are a helpful assistant that reads reasoning traces.",
    model_name="Qwen/Qwen2.5-7B-Instruct",
    window=30,
    quantile=10,
    max_kv_tokens=512,
)

segments = [trace[s:e] for s, e in offsets]
```