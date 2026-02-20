---
layout: default
title: Engines
---

# Available Engines

## Rule-Based

* `RTRuleRegex`
* `RTNewLine`

## LLM-Based (Boundary Inference)

* `RTLLMOffsetBased`
* `RTLLMSegUnitBased`
* `RTLLMForcedDecoderBased`
* `RTLLMSurprisal`
* `RTLLMEntropy`
* `RTLLMTopKShift`
* `RTLLMFlatnessBreak`

## Discourse / Reasoning Schemas

* `RTLLMThoughtAnchor`
* `RTLLMReasoningFlow`
* `RTLLMArgument`

## Topic / Semantic / NLI

* `RTBERTopicSegmentation`
* `RTEmbeddingBasedSemanticShift`
* `RTEntailmentBasedSegmentation`
* `RTZeroShotSeqClassification`
* `RTZeroShotSeqClassificationRF`
* `RTZeroShotSeqClassificationTA`

## PRM-Based

* `RTPRMBase`

---

# Engine Configuration

You can override engine parameters at call time:

```python
offsets, labels = segmentor(
    trace,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    chunk_size=200,
)
```