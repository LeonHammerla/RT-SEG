---
layout: default
title: Evaluation & Metrics
parent: Reproducibility
nav_order: 1
---

# Evaluation & Metrics

RT-SEG represents segmentations as **character offsets**: a list of `(start, end)` spans that partition a trace string.
Evaluation therefore reduces to comparing **boundary positions** (segment ends) and/or **segment spans**.

This page documents the metrics implemented in RT-SEG’s evaluation utilities, how to interpret them, and how to report results.

---

## Terminology (offsets → boundaries)

Given offsets:

- `[(s0, e0), (s1, e1), ..., (sk, ek)]` with `e_k = len(trace)`

The **boundary indices** are the *end offsets* of all but the last segment:

- `B = { e0, e1, ..., e_{k-1} }`

Most boundary metrics operate on these boundary index sets.

---

## Primary metric used in experiments: Boundary Similarity (B-score)

RT-SEG’s main “agreement-style” score is a **lenient boundary similarity** that treats near-miss boundaries as matches.

### Definition (as implemented)

Let:

- `B1` = boundary indices of segmentation 1
- `B2` = boundary indices of segmentation 2
- `window` = jitter tolerance in characters (± window)

Directional match count:

- A boundary `b ∈ B1` is a match if `∃ b' ∈ B2` such that `|b - b'| ≤ window`.

Compute matches both directions:

- `m1 = matches(B1 → B2)`
- `m2 = matches(B2 → B1)`

Convert to precision/recall style:

- `p = m1 / |B1|`
- `r = m2 / |B2|`

Final score is the harmonic mean:

- `B = 2pr / (p + r)` (or 0 if `p + r = 0`)

### Interpretation

- **Range:** `0.0 … 1.0` (higher is better)
- **What it rewards:** boundaries that land “close enough” to the gold boundary (within `window` chars)
- **What it penalizes:** missing boundaries, or boundaries far away (outside the jitter window)

### Choosing `window`

`window` is in **characters**, not tokens or sentences.

- Small window (e.g. 3–5): strict; good when offsets are stable and annotation is precise.
- Larger window (e.g. 10–30): tolerant; good when offsets drift due to formatting, tokenization artifacts, or sentence splitting ambiguity.

---

## Gold-based metric groups (single-trace)

RT-SEG’s evaluation entry point groups metrics into categories. The most useful gold-based ones:

### 1) Classical segmentation metrics (character-label based)

These treat segmentation as a **sequence labeling** problem over characters.

- **P_k** (lower is better)  
  Probability that two positions at distance `k` are incorrectly judged as “same segment vs different segment”.
- **WindowDiff** (lower is better)  
  Counts disagreements in the *number of boundaries* within a sliding window.

Notes:
- `k` is chosen automatically as ~ half the average segment length (computed from gold labels).

### 2) Boundary accuracy (strict vs tolerant)

- **Boundary_F1** (higher is better)  
  Exact-match boundary precision/recall/F1 using boundary index sets.
- **Soft_Boundary_F1** (higher is better)  
  Distance-weighted boundary F1 using exponential decay with tolerance parameter `sigma`:

  `exp(-distance / sigma)`

- **Boundary_Displacement** (lower is better)  
  Mean absolute distance from each gold boundary to the nearest predicted boundary.

Parameters:
- `sigma` (characters): larger = more forgiving for near-misses.

### 3) Segment structure overlap

These compare **segment spans**, not only boundaries.

- **Mean_IoU** (higher is better)  
  For each gold segment, find the best matching predicted segment and compute Intersection-over-Union, then average.
- **Mean_Dice** (higher is better)  
  Similar but using Dice coefficient.
- **Segmentation_Bias**  
  `(num_pred_segments - num_gold_segments) / max(1, num_gold_segments)`

Interpretation:
- Bias > 0 → over-segmentation (too many segments)
- Bias < 0 → under-segmentation (too few segments)

---

## Gold-free diagnostics (pairwise agreement)

When you don’t have a gold segmentation, RT-SEG supports pairwise comparisons between methods/annotators:

- **Boundary_Similarity** (the B-score above) with `window`
- **Boundary_Density_JSD**  
  Jensen–Shannon divergence between boundary-position histograms across the trace (lower = more similar distribution).

This is useful for:
- annotator agreement studies
- comparing two engines without claiming either is “gold”

---

## “Optimistic” boundary cover (lenient diagnostic)

**Boundary_Cover** measures how well one segmentation’s boundaries are “covered” by another within a slack.

- `slack` (characters) is the tolerance.
- It is *optimistic* in the sense that it does **not** directly penalize extra boundaries in the covering segmentation.

This is best used as a **diagnostic**, not the headline metric.

---

## Minimal example: evaluate one method vs gold (single trace)
```python
from rt_segmentation.seg_eval_utils import evaluate_segmentations

trace = "Step 1: Get data. Data is [1, 2]. Step 2: Sum data. Sum is 3. Step 3: Square it. Result is 9."

segmentations = {
    "Gold": [(0, 31), (31, 59), (59, 84)],
    "MethodA": [(0, 31), (31, 84)],
}

tables = evaluate_segmentations(
    trace=trace,
    segmentations=segmentations,
    gold_key="Gold",
    sigma=5.0,     # soft boundary tolerance (chars)
    window=10,     # boundary similarity jitter (chars)
    slack=10,      # boundary cover jitter (chars)
)

for group, df in tables.items():
    print(f"\n=== {group} ===")
    print(df)
```
What you get:
- a dict of DataFrames keyed by metric group (e.g. `classical`, `boundary_accuracy`, `segment_structure`, `agreement`, `optimistic`)

---

## Minimal example: evaluate across a dataset (aggregate)

Use this when you have multiple traces and want means/stds per method:
```python
from rt_segmentation.seg_eval_utils import evaluate_aggregate_segmentations, aggregated_results_to_json

traces = [
    "Trace one ...",
    "Trace two ...",
]

segmentations_per_trace = [
    {
        "Gold": [(0, 10), (10, 20)],
        "MethodA": [(0, 12), (12, 20)],
    },
    {
        "Gold": [(0, 5), (5, 15)],
        "MethodA": [(0, 6), (6, 15)],
    },
]

agg = evaluate_aggregate_segmentations(
    traces=traces,
    segmentations=segmentations_per_trace,
    gold_key="Gold",
    sigma=5.0,
    window=10,
    slack=10,
)

# Optional: JSON-serializable structure for saving
payload = aggregated_results_to_json(agg)
print(payload.keys())  # linear_metrics, pairwise_agreement_metrics, per_method_agreement_metrics
```
---

## Recommended reporting format (for papers / benchmarks)

### 1) Always report metric hyperparameters

Because `window`, `sigma`, and `slack` directly control tolerance, include them in every table caption or JSON header:

- `window` (Boundary Similarity jitter, chars)
- `sigma` (Soft boundary decay scale, chars)
- `slack` (Boundary cover jitter, chars)

### 2) Report a compact “headline + diagnostics” set

A practical default bundle:

**Headline (pick 1–2):**
- `Boundary_Similarity` (with specified `window`)
- `Soft_Boundary_F1` (with specified `sigma`)

**Diagnostics:**
- `Boundary_Displacement` (lower is better)
- `Segmentation_Bias` (sign tells over/under segmentation)

### 3) Include runtime separately

If you track per-trace processing time (e.g. `ptime`), report:
- mean runtime per trace
- optionally p50/p95 if distributions are heavy-tailed (LLM engines often are)

### 4) Save results as JSON

Store:
- method identifier (engine(s) + aligner + base unit)
- metric means/stds
- evaluation hyperparameters (`window`, `sigma`, `slack`)
- dataset identifier / split name

This makes runs reproducible and comparable across machines.
