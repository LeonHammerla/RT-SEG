---
layout: default
title: RTBERTopicSegmentation
parent: Semantic Engines
nav_order: 3
---

# RTBERTopicSegmentation (Topic-driven segmentation via BERTopic)

## Idea

`RTBERTopicSegmentation` segments a reasoning trace by assigning a **topic** to each base unit (sentence or clause) using **BERTopic**, and then merging consecutive units that share the same topic label.

This engine is intended to capture **macro-level semantic structure**: changes in topic often align with shifts in subproblem, method, or discourse function. Compared to embedding-only shift detection, BERTopic introduces an explicit clustering step, yielding discrete topic IDs (and optionally interpretable topic labels).

For short traces (or when topic modeling fails), the engine falls back to **zero-shot discourse labeling**.

---

## Method (high-level)

Given a trace split into base units \(u_1, \dots, u_m\):

1. **Base segmentation**
   Compute base offsets via `SegBase.get_base_offsets(trace, seg_base_unit=...)` and extract base unit strings.

2. **Topic modeling with BERTopic**
   - Embed all base units with a SentenceTransformer embedding model.
   - Fit BERTopic over the documents + embeddings to infer topic assignments:
     - UMAP dimensionality reduction
     - HDBSCAN clustering
     - CountVectorizer for topic word representation

   Output:
   - `topics[i]`: topic index for unit \(u_i\)

3. **Topic labeling**
   Two labeling modes:

   **(a) Default BERTopic labels**
   - Uses `topic_model.topic_labels_` (derived from topic words)

   **(b) Custom labels via LLM (optional)**
   - For each topic, retrieve representative documents.
   - Prompt an instruction-tuned LM to generate a “correct label” string.
   - Use these generated labels for per-unit labels.

4. **Merge adjacent units with identical topics**
   As in label-transition segmentation: merge consecutive units as long as their topic label remains unchanged.

5. **Fallback behavior**
   If `len(documents) < 100` (too few units for stable topic modeling) **or** if BERTopic fails, the engine falls back to:

   - `RTZeroShotSeqClassification` with MNLI model and generic reasoning-flow labels:
     - `Context, Planning, Fact, Restatement, Example, Reflection, Conclusion`

---

## Models used

This engine can involve up to two model classes:

### 1) Embedding model (for BERTopic)
Supported in code:

- `all-MiniLM-L6-v2` (default)
- `Qwen/Qwen3-Embedding-0.6B`

Used via `SentenceTransformer(...).encode(...)`.

### 2) Optional LLM for topic label generation (`all_custom_labels=True`)
Supported in code:

- `Qwen/Qwen2.5-7B-Instruct-1M`
- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`

Used to generate human-readable topic labels from representative documents.

### 3) Fallback model (if BERTopic is skipped or fails)
- `facebook/bart-large-mnli` via `pipeline("zero-shot-classification")`

---

## Key parameters

- `seg_base_unit: Literal["sent", "clause"]`
  Base unit granularity for topic assignment.

- `embedding_model_name: str` (default: `"all-MiniLM-L6-v2"`)
  SentenceTransformer model used to embed base units for BERTopic.

- `all_custom_labels: bool` (default: `False`)
  If `True`, generates topic labels with an LLM using representative documents.

- `model_name: str`
  LLM used for `all_custom_labels=True`.

- `system_prompt: str` (default: `""`)
  System prompt used when generating custom topic labels.

---

## Usage

### Topic segmentation with default BERTopic topic labels

```python
from rt_seg.seg_factory import RTSeg
from rt_seg.bertopic_segmentation import RTBERTopicSegmentation  # adjust import to your package structure

trace = "..."

segmentor = RTSeg(engines=RTBERTopicSegmentation)

offsets, labels = segmentor(
    trace,
    seg_base_unit="sent",
    embedding_model_name="all-MiniLM-L6-v2",
)
```
---

### Topic segmentation with LLM-generated custom labels

```python
offsets, labels = segmentor(
    trace,
    seg_base_unit="sent",
    embedding_model_name="all-MiniLM-L6-v2",
    all_custom_labels=True,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    system_prompt="You are labeling topics in reasoning traces. Return short, descriptive topic names.",
)
```