---
layout: default
title: Engine Defaults & Evaluation Presets
parent: Reproducibility
nav_order: 2
---

# Engine Defaults & Evaluation Presets

This page is a **compact, reproducible reference** for:

1. The *default* engine hyperparameters used by `RTSeg` when you do not override kwargs.
2. The *default / recommended* evaluation settings used by the metrics utilities.

> All values below reflect the defaults defined in the code (factory defaults + metric function defaults).

---

## Global `RTSeg` defaults

When you construct `RTSeg(...)`:

| Setting | Default | Meaning |
|---|---|---|
| `label_fusion_type` | `"majority"` | How to fuse labels when multiple engines are used |
| `seg_base_unit` | `"clause"` | Base unit used by engines that rely on base offsets |
| `model_base` | `"Qwen/Qwen2.5-0.5B-Instruct"` | Default LLM used by several forced-decoding engines |

---

## Engine defaults (factory-provided)

The table below lists the default kwargs injected by `RTSeg` per engine.

Notes:
- `seg_base_unit` is injected into **every** engine call (whatever you set on `RTSeg`).
- Prompt values are stored under keys like `system_prompt_offset` in your prompt registry; here we reference them by key name (not the full prompt text).
- You can override any of these at call time: `offsets, labels = segmentor(trace, model_name="...", chunk_size=..., ...)`.

| Engine | Category | Default kwargs (as passed by `RTSeg`) |
|---|---|---|
| `RTRuleRegex` | Rule-based | `model_name=None`, `system_prompt=None`, `seg_base_unit=<RTSeg.seg_base_unit>` |
| `RTNewLine` | Rule-based | `model_name=None`, `system_prompt=None`, `seg_base_unit=<RTSeg.seg_base_unit>` |
| `RTLLMOffsetBased` | LLM (offset boundaries) | `model_name="Qwen/Qwen2.5-7B-Instruct"`, `system_prompt=load_prompt("system_prompt_offset")`, `prompt=""`, `chunk_size=300`, `seg_base_unit=<...>` |
| `RTLLMSegUnitBased` | LLM (segment units) | `model_name="Qwen/Qwen2.5-7B-Instruct"`, `system_prompt=load_prompt("system_prompt_sentbased")`, `prompt=""`, `chunk_size=100`, `seg_base_unit=<...>` |
| `RTLLMForcedDecoderBased` | Forced decoding | `model_name=<RTSeg.model_base>`, `system_prompt=load_prompt("system_prompt_forceddecoder")`, `seg_base_unit=<...>` |
| `RTLLMSurprisal` | Probabilistic (forced decoding) | `model_name=<RTSeg.model_base>`, `system_prompt=load_prompt("system_prompt_surprisal")`, `seg_base_unit=<...>` *(engine-level `window/quantile/max_kv_tokens` use the engine’s own defaults unless overridden)* |
| `RTLLMEntropy` | Probabilistic (forced decoding) | `model_name=<RTSeg.model_base>`, `system_prompt=load_prompt("system_prompt_surprisal")`, `seg_base_unit=<...>` *(engine-level `window/quantile/max_kv_tokens` use the engine’s own defaults unless overridden)* |
| `RTLLMTopKShift` | Probabilistic (forced decoding) | `model_name=<RTSeg.model_base>`, `system_prompt=load_prompt("system_prompt_surprisal")`, `seg_base_unit=<...>` *(engine-level `top_k/quantile/...` use engine defaults unless overridden)* |
| `RTLLMFlatnessBreak` | Probabilistic (forced decoding) | `model_name=<RTSeg.model_base>`, `system_prompt=load_prompt("system_prompt_surprisal")`, `seg_base_unit=<...>` |
| `RTBERTopicSegmentation` | Topic | `model_name="Qwen/Qwen2.5-1.5B-Instruct"`, `system_prompt=load_prompt("system_prompt_topic_label")`, `seg_base_unit=<...>` |
| `RTZeroShotSeqClassification` | Zero-shot | `model_name="facebook/bart-large-mnli"`, `system_prompt=""`, `labels=["verification","pivot","inference","framing","conclusion"]`, `seg_base_unit=<...>` |
| `RTZeroShotSeqClassificationRF` | Zero-shot (RF) | `model_name="facebook/bart-large-mnli"`, `system_prompt=""`, `seg_base_unit=<...>` |
| `RTZeroShotSeqClassificationTA` | Zero-shot (TA) | `model_name="facebook/bart-large-mnli"`, `system_prompt=""`, `seg_base_unit=<...>` |
| `RTPRMBase` | PRM-based | `model_name="Qwen/Qwen2.5-Math-7B-PRM800K"`, `system_prompt=""`, `seg_base_unit=<...>` |
| `RTEmbeddingBasedSemanticShift` | Semantic shift | `model_name="all-MiniLM-L6-v2"`, `system_prompt=""`, `seg_base_unit=<...>` |
| `RTEntailmentBasedSegmentation` | Entailment / NLI | `model_name="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"`, `system_prompt=""`, `seg_base_unit=<...>` |
| `RTLLMThoughtAnchor` | LLM schema labeling | `model_name="Qwen/Qwen2.5-7B-Instruct"`, `system_prompt=load_prompt("system_prompt_thought_anchor")`, `user_prompt=load_prompt("user_prompt_thought_anchor")`, `seg_base_unit=<...>` |
| `RTLLMReasoningFlow` | LLM schema labeling | `model_name="Qwen/Qwen2.5-7B-Instruct"`, `system_prompt=load_prompt("system_prompt_reasoning_flow")`, `user_prompt=load_prompt("user_prompt_reasoning_flow")`, `seg_base_unit=<...>` |
| `RTLLMArgument` | LLM schema labeling | `model_name="Qwen/Qwen2.5-7B-Instruct"`, `system_prompt=load_prompt("system_prompt_argument")`, `user_prompt=load_prompt("user_prompt_argument")`, `seg_base_unit=<...>` |

---

## Evaluation settings (defaults + recommended preset)

RT-SEG provides several metric families. The defaults below are important because they directly affect tolerance for “near misses”.

### Key parameters

| Parameter | Unit | Default (in metric code) | Where it applies |
|---|---:|---:|---|
| `sigma` | chars | `5.0` | Soft boundary scoring (`Soft_Boundary_F1`) |
| `window` | chars | `3` | Pairwise `Boundary_Similarity` in the evaluation registry |
| `slack` | chars | `10` | `Boundary_Cover` (optimistic diagnostic) |

Additionally, some helper/aggregate scorers default to:

| Function | Default | Meaning |
|---|---:|---|
| `ReasoningAgreementSuite(window_size=...)` | `3` | Jitter tolerance for boundary similarity (used by the suite) |
| `evaluate_approaches_bounding_similarity(..., window=...)` | `10` | Triadic boundary similarity aggregation default |

### Recommended “paper preset”

If you want one consistent, explicit setting to report (and to reproduce tables), a sensible preset that matches your test usage is:

- `sigma=5.0`
- `window=3`
- `slack=10`

When you report results, include these values in the caption/JSON header alongside:
- dataset/split identifier
- engine(s) + aligner
- `seg_base_unit`
- `model_name` and prompt identifiers (for LLM-based engines)
