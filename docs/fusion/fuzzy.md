---
layout: default
title: OffsetFusionFuzzy
parent: Fusion
nav_order: 4
---

# OffsetFusionFuzzy (Distance-Tolerant Fusion)

## Idea

`OffsetFusionFuzzy` merges boundaries that are **close in character space**.

Useful when engines produce slightly misaligned offsets.

---

## Method

1. Collect all boundaries.
2. Sort them.
3. Merge boundaries within `max_distance` characters.
4. Convert merged boundaries to segments.

---

## Parameter

- `max_distance: int` (default: `1`)
  Maximum character distance for merging.

---

## Effect

- Tolerant to minor offset drift.
- Prevents fragmentation due to near-identical boundaries.

---

## Recommended for

- LLM offset-based engines.
- Mixed statistical + LLM pipelines.
