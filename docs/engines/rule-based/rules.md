---
layout: default
title: RTRuleRegex
parent: Rule-Based Engines
nav_order: 1
---

# RTRuleRegex (Heuristic segmentation via discourse marker patterns)

## Idea

`RTRuleRegex` is a lightweight, fully deterministic segmentation engine that detects likely reasoning-step boundaries using **surface-level discourse markers** and simple heuristics.

The engine scans sentence/clause units and starts a new segment when it encounters markers associated with:

- **inference / conclusion** (e.g., “therefore”, “thus”),
- **contrast / correction** (e.g., “however”, “but”, “instead”),
- **revision / self-correction** (e.g., “this is wrong”, “wait”, “let me reconsider”),
- **goal shifts** (e.g., “next”, “we need to”, “consider”),
- **final answer cues** (e.g., “final answer”, “the answer is …”).

This serves as a strong baseline: it is fast, interpretable, and requires no models.

---

## Method (high-level)

1. **Base unit spans**
   Compute base offsets using `SegBase.get_base_offsets(trace, seg_base_unit=...)`.

2. **Marker matching per unit**
   For each unit \(u_i\), check whether it contains any regular-expression marker from the following categories:

   - `INFERENCE_MARKERS`
   - `CONTRAST_MARKERS`
   - `REVISION_MARKERS`
   - `GOAL_MARKERS`
   - `FINAL_MARKERS`

   Matching is case-insensitive and uses `re.search(...)` per marker.

3. **Boundary insertion**
   If a unit contains any marker (and it is not the first unit), a segment boundary is inserted at the start of that unit:
   - close previous segment at `sent_start`
   - start new segment at `sent_start`

4. **Output**
   Return character-offset segments spanning the full trace, with labels set to `"UNK"`.

---

## Marker sets (as implemented)

### Inference markers
```text
therefore, thus, hence, so, follows that, implies, we conclude
````

### Contrast markers

```text
but, however, though, instead, actually, nevertheless, on the other hand
```

### Revision/self-correction markers

```text
this is wrong, that was wrong, i was mistaken, let me reconsider, wait
```

### Goal/plan shift markers

```text
now we, let us, next, consider, we need to, the goal
```

### Final answer markers

```text
the answer is, therefore the answer, in conclusion, final answer
```

---

## Models used

None. This is a rule-based baseline (regex + base-unit offsets).

---

## Key parameters

* `seg_base_unit: Literal["sent", "clause"]`
  Determines whether marker checks run over sentence-level or clause-level spans.
  In practice:

  * `"sent"` yields fewer, more conservative boundaries
  * `"clause"` yields finer splits and can catch mid-sentence cue phrases

No other parameters are required.

---

## Usage

```python
from rt_seg.seg_factory import RTSeg
from rt_seg.rule_split_regex import RTRuleRegex  # adjust import to your package structure

trace = "We first compute X. However, this approach fails. Therefore we try Y. Final answer: ..."

segmentor = RTSeg(engines=RTRuleRegex, seg_base_unit="sent")
offsets, labels = segmentor(trace)

segments = [trace[s:e] for s, e in offsets]
for seg in segments:
    print(seg)
```
