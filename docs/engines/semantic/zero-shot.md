---
layout: default
title: RTZeroShotSeqClassification|TA|RF
parent: Semantic Engines
nav_order: 2
has_children: true
---

# RTZeroShotSeqClassification (Zero-shot schema labeling + segmentation)

## Idea

The `RTZeroShotSeqClassification` family segments a reasoning trace by first assigning a **discourse / reasoning label** to each base unit (sentence or clause) using **zero-shot sequence classification**, and then merging consecutive units that share the same predicted label.

This yields a segmentation that is explicitly interpretable: segment boundaries occur when the predicted **schema role** changes.

We provide:

- **Generic** zero-shot segmentation (`RTZeroShotSeqClassification`) with configurable label sets.
- Two specialized variants aligned with our paper’s schemas:
  - **Reasoning Flow** (`RTZeroShotSeqClassificationRF`)
  - **Thought Anchor** (`RTZeroShotSeqClassificationTA`)

---

## Method (high-level)

Given a trace and a base unit choice (`sent` or `clause`):

1. **Base segmentation**
   Compute base offsets via:
   - `SegBase.get_base_offsets(trace, seg_base_unit=...)`

2. **Zero-shot classification per base unit**
   For each base span `u_i`, run a zero-shot NLI classifier:
   - `label_i = argmax_label p(label | u_i)`

   The implementation uses the HuggingFace `pipeline("zero-shot-classification")` with `multi_label=False` (forced single best label).

3. **Merge adjacent spans with identical labels**
   Consecutive base units are merged into a segment as long as their predicted labels remain the same.
   A new segment starts when `label_i != label_{i-1}`.

Output:
- `final_offsets`: merged character offsets
- `final_labels`: the predicted schema label for each merged segment

---

## Models used

These engines use **NLI-style zero-shot classification models** via the HuggingFace `pipeline`.

Supported model names in the code:

- `facebook/bart-large-mnli`
- `FacebookAI/roberta-large-mnli`
- `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`

Implementation notes:
- Model is loaded with `device_map="auto"` and `torch_dtype="auto"`.
- Inference is done independently per base unit (sentence/clause).

> For reproducibility, report the exact model used and the exact label set used (below), since both directly determine the segmentation.

---

## Label sets (explicit)

### Generic default labels (`RTZeroShotSeqClassification`)
```text
Context
Planning
Fact
Restatement
Example
Reflection
Conclusion
```

### Reasoning Flow schema (`RTZeroShotSeqClassificationRF`)
```text
Context
Planning
Fact
Reasoning
Restatement
Assumption
Example
Reflection
Conclusion
```

### Thought Anchor schema (`RTZeroShotSeqClassificationTA`)
```text
Problem Setup
Plan Generation
Fact Retrieval
Active Computation
Uncertainty Management
Result Consolidation
Self Checking
Final Answer Emission
```