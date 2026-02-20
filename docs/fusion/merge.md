---
layout: default
title: OffsetFusionMerge
parent: Offset Fusion (Late Fusion Aligners)
nav_order: 1
---

# OffsetFusionMerge (Union Fusion)

## Idea

`OffsetFusionMerge` keeps **all boundaries proposed by any engine**.

It is the most permissive fusion strategy.

---

## Method

1. Collect union of all boundary positions.
2. Sort them.
3. Convert consecutive boundaries into `(start,end)` segments.

---

## Effect

- High boundary recall.
- Often over-segments.
- Useful when downstream tasks tolerate fine granularity.

---

## Recommended for

- Exploratory analysis
- Avoiding missed boundaries
- Post-hoc merging downstream
