---
layout: default
title: RTLLMForcedDecoderBased
parent: Probabilistic
nav_order: 3
---

# RTLLMForcedDecoderBased (Forced-decoding with separator-token competition)

## Idea

`RTLLMForcedDecoderBased` performs segmentation by **explicitly testing whether the model would prefer to emit a segmentation marker** at each position in the trace.

Instead of relying on a scalar uncertainty signal (entropy) or the likelihood of the observed token (surprisal), this engine compares:

- the log-probability of the **true next token** (forced decoding), vs.
- the log-probability of one or more **separator candidates** (e.g., a special token `<|seg|>` and optionally a single-token `"Step"`)

If separator candidates become relatively more likely than the true continuation—especially around punctuation—the engine inserts a boundary.

This can be interpreted as a lightweight proxy for:  
“Would the model naturally insert a step marker here?”

---

## Method (high-level)

### Separator candidates
The engine augments the tokenizer with a special separator token (default: `<|seg|>`), and also attempts to include `"Step"` / `" Step"` if they are **single-token** under the tokenizer.

Let `S` be the set of separator candidate token IDs.

### Pass 1: calibrate a threshold (score distribution)
For each position in the trace, under forced decoding:

1. Compute:
   - `true_logp = log p(t_i | context)`
   - `sep_logp = max_{s∈S} log p(s | context)`

2. Compute a **gap**:
   \[
   \text{gap}_i = \text{sep\_logp}_i - \text{true\_logp}_i
   \]
   Larger gap means the separator is more competitive.

3. Add a **punctuation prior** based on the last emitted character:
   - hard punctuation (`. ! ? ; : \n`) → `sep_bias = 1.0`
   - soft punctuation (`, \t`) → `sep_bias ≈ 0.5` (first pass) / `0.2` (second pass)
   - otherwise → `0.0`

4. Normalize `gap` over the trace using z-scoring, then compute:
   \[
   \text{score}_i = \alpha \cdot \text{gap\_z}_i + \beta \cdot \text{sep\_bias}_i
   \]

5. Choose a threshold as a percentile (`quantile`) of the scores:
   \[
   \tau = \text{percentile}(\text{score}, q=\text{quantile})
   \]

This produces `(τ, gap_mean, gap_std)` for reuse.

### Pass 2: segmentation with token insertion
Perform a second forced-decoding pass. At each step compute the same score, and if:

- `score > τ`, and
- the model is not already emitting a separator, and
- the previous emitted token was not `<|seg|>`

then the engine **inserts** the separator token into the model context (without consuming a trace token) and records a boundary at the current character cursor.

Finally, convert boundaries into `(start, end)` character offsets.

---

## Models used

The engine supports (as implemented):

- `Qwen/Qwen2.5-7B-Instruct-1M`
- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`

Implementation details:
- Uses `AutoModelForCausalLM` and `AutoTokenizer`.
- Uses `tokenizer.apply_chat_template(...)` with a `system_prompt`.
- Adds `sep_tok` as an additional special token and calls `model.resize_token_embeddings(...)`.
- Uses `device_map="auto"` and `torch_dtype="auto"`.

> For reproducibility in the paper: report the exact model(s), the `system_prompt`, and whether `"Step"` is used as a separator candidate for that model/tokenizer (it depends on tokenization).

---

## Key parameters

- `system_prompt: str`  
  System instruction used in chat formatting.

- `model_name: str`  
  One of the supported model identifiers above.

- `sep_tok: str` (default: `"<|seg|>"`)  
  Special separator token added to the tokenizer and used for insertion.

- `alpha: float` (default: `1.0`)  
  Weight on the normalized gap term (separator-vs-true competition).

- `beta: float` (default: `2.0` in `_segment`, `1.5` in internal passes)  
  Weight on the punctuation prior.

- `quantile: float` (default: `90.0`)  
  Percentile used to choose the insertion threshold. Larger values typically mean fewer insertions (more conservative).

- `max_kv_tokens: int` (default: `512`)  
  Caps the KV cache length via `DynamicCache.crop(...)`.

---

## Usage

```python
from rt_seg.seg_factory import RTSeg
from rt_seg.rt_llm_forced_decoder_based import RTLLMForcedDecoderBased  # adjust import to your package structure

trace = "First step... Then second step... Finally conclude."

segmentor = RTSeg(
    engines=RTLLMForcedDecoderBased,
    seg_base_unit="clause",  # included for API consistency; this engine operates token-wise
)

offsets, labels = segmentor(
    trace,
    system_prompt="You are a helpful assistant that reads reasoning traces.",
    model_name="Qwen/Qwen2.5-7B-Instruct",
    sep_tok="<|seg|>",
    alpha=1.0,
    beta=2.0,
    quantile=90.0,
    max_kv_tokens=512,
)

segments = [trace[s:e] for s, e in offsets]
```