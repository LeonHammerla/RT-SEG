---
layout: default
title: RTEmbeddingBasedSemanticShift
parent: Semantic Engines
nav_order: 1
---

# RTEmbeddingBasedSemanticShift (Embedding-based semantic shift segmentation)

## Idea

`RTEmbeddingBasedSemanticShift` segments a reasoning trace by tracking **semantic coherence** over a rolling segment representation.  
It embeds each base unit (sentence or clause) and compares it to the current segment’s **centroid embedding**. When the next unit is insufficiently similar to the current centroid (under an adaptive threshold), a **semantic shift** is detected and a new segment is started.

This engine is **model-agnostic** (works with any sentence embedding model) and does not rely on token-level LM probabilities. It is particularly useful when step boundaries correspond to topical or semantic transitions rather than punctuation or distributional changes.

---

## Method (high-level)

Given a trace split into base units \(u_1, \dots, u_m\) (sentences or clauses):

1. **Base segmentation**  
   Compute base offsets using `SegBase.get_base_offsets(trace, seg_base_unit=...)`, then extract strings:
   \[
   u_i = \text{trace}[s_i : e_i]
   \]

2. **Embed each unit**  
   Compute embeddings \(v_i = \mathrm{Embed}(u_i)\) using a SentenceTransformer model.

3. **Maintain a running segment centroid**  
   For the current segment, maintain a set of vectors and compute the centroid:
   \[
   c = \frac{1}{k} \sum_{j=1}^{k} v_j
   \]

4. **Similarity test**  
   Compute cosine similarity between the centroid \(c\) and the candidate unit vector \(v_i\):
   \[
   \mathrm{sim}(c, v_i) = \cos(c, v_i)
   \]

5. **Adaptive thresholding**  
   Maintain a running average similarity `running_avg_sim` (initialized to 1.0).  
   Compute a dynamic threshold:
   - `dynamic_threshold = running_avg_sim - tolerance`
   - `effective_threshold = max(dynamic_threshold, min_threshold)`

   Accept the unit if:
   - `sim >= effective_threshold`

6. **“Bridge/outlier” heuristic**  
   If the current unit fails the threshold, the engine checks whether the *next* unit would match the current segment centroid.  
   If yes, the current unit is treated as a potential “bridge” (e.g., short meta-note) and is kept in the current segment.

7. **Emit segments as offsets**  
   When a unit does not match (and is not a bridge), finalize the current segment and begin a new segment at the current unit’s offset.  
   Return character offsets and `"UNK"` labels.

---

## Models used

This engine uses **sentence embedding models** via `sentence-transformers`.

Supported in code:

- `all-MiniLM-L6-v2` (default)
- `Qwen/Qwen3-Embedding-0.6B`

Implementation notes:
- Embeddings are computed using `SentenceTransformer(model_name).encode(...)`.
- Similarity is computed with `sentence_transformers.util.cos_sim(...)`.

> For reproducibility, report which embedding model was used in experiments (and any batching or normalization settings, if changed).

---

## Key parameters

- `seg_base_unit: Literal["sent", "clause"]`  
  Base unit for computing embeddings.  
  - `"clause"` can capture finer semantic turns  
  - `"sent"` tends to produce fewer segments

- `model_name: str` (default: `"all-MiniLM-L6-v2"`)  
  SentenceTransformer embedding model.

- `tolerance: float` (default: `0.15`)  
  How far the similarity may fall below the running average before triggering a split.  
  Larger tolerance → fewer splits; smaller tolerance → more splits.

- `min_threshold: float` (default: `0.4`)  
  Absolute floor on the effective threshold (prevents splitting too aggressively when `running_avg_sim` drops).

---

## Usage

```python
from rt_seg.seg_factory import RTSeg
from rt_seg.semantic_shift import RTEmbeddingBasedSemanticShift  # adjust import to your package structure

trace = "First step... Then second step... Finally conclude."

segmentor = RTSeg(
    engines=RTEmbeddingBasedSemanticShift,
    seg_base_unit="sent",
)

offsets, labels = segmentor(
    trace,
    seg_base_unit="sent",
    model_name="all-MiniLM-L6-v2",
    tolerance=0.15,
    min_threshold=0.4,
)

segments = [trace[s:e] for s, e in offsets]
```