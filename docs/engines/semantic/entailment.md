---
layout: default
title: RTEntailmentBasedSegmentation
parent: Semantic Engines
nav_order: 4
---

# RTEntailmentBasedSegmentation (Coherence via NLI / entailment scoring)

## Idea

`RTEntailmentBasedSegmentation` segments a reasoning trace by measuring **local discourse coherence** using a **Natural Language Inference (NLI)** model.  
Instead of embedding similarity, it evaluates whether the next unit (sentence/clause) is *entailed by* or at least *consistent with* the recent context of the current segment.

When coherence drops below a dynamic threshold, the engine starts a new segment. This makes the method sensitive to **logical/semantic continuity** rather than topical similarity alone.

---

## Method (high-level)

Given base units \(u_1, \dots, u_m\) (sentences or clauses):

1. **Base segmentation**
   Compute base offsets via `SegBase.get_base_offsets(trace, seg_base_unit=...)`, then extract the unit strings.

2. **Local context construction**
   Maintain the current segment `current_segment`.  
   For the next unit \(u_i\), build a short context from the **last two units** in the current segment:
   \[
   c_i = u_{i-2} \,\Vert\, u_{i-1}
   \]
   (joined with whitespace)

3. **NLI-based coherence score**
   Feed `(premise=context, hypothesis=next_unit)` into an NLI model and compute class probabilities:

   - \(p(\text{entailment})\)
   - \(p(\text{neutral})\)
   - (optionally \(p(\text{contradiction})\), not used in the final score)

   The engine defines a scalar coherence score:
   \[
   \text{score} = p(\text{entailment}) + 0.4 \cdot p(\text{neutral})
   \]
   clipped to \([0,1]\).

4. **Adaptive thresholding**
   Maintain a running average coherence score `running_avg_score` (initialized to 0.85).  
   Compute a dynamic threshold:
   - `dynamic_threshold = max(min_threshold, running_avg_score - tolerance)`
   Accept the next unit if:
   - `score >= dynamic_threshold`

5. **Lookahead (“bridge/outlier”) heuristic**
   If the current unit fails, check the *next* unit \(u_{i+1}\) against the same context.  
   If \(u_{i+1}\) is coherent, keep the current unit to preserve flow (treat as bridge/outlier).

6. **Segment emission**
   If coherence fails (and lookahead does not rescue), finalize the current segment offset and start a new segment at the current unit.  
   Return the final character offsets and `"UNK"` labels.

---

## Models used

Although the `load_model` type annotation lists causal LMs, the implementation uses:

- `AutoModelForSequenceClassification`
- `AutoTokenizer`

The default `model_name` in `_segment` is:

- `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`

This is an NLI/XNLI-style model that provides logits over NLI classes.

> **Note (implementation detail):** The NLI label index mapping is assumed in `_predict_coherence` (entailment = `probs[0]`, neutral = `probs[1]`). If you switch models, verify the class order in `config.id2label`, as different checkpoints may order labels differently (e.g., contradiction/neutral/entailment).

---

## Key parameters

- `seg_base_unit: Literal["sent", "clause"]`  
  Base unit granularity for coherence testing.

- `model_name: str`  
  NLI sequence classification model (default above).

- `tolerance: float` (default: `0.15`)  
  How far the coherence score may fall below the running average before triggering a split.  
  Larger tolerance → fewer splits; smaller tolerance → more splits.

- `min_threshold: float` (default: `0.25`)  
  Absolute floor on the dynamic threshold, preventing very low thresholds when the running average drops.

---

## Usage

```python
from rt_seg.seg_factory import RTSeg
from rt_seg.entailment_segmentation import RTEntailmentBasedSegmentation  # adjust import to your package structure

trace = "..."

segmentor = RTSeg(
    engines=RTEntailmentBasedSegmentation,
    seg_base_unit="sent",
)

offsets, labels = segmentor(
    trace,
    seg_base_unit="sent",
    model_name="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    tolerance=0.15,
    min_threshold=0.25,
)

segments = [trace[s:e] for s, e in offsets]
```