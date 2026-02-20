---
layout: default
title: OffsetFusionGraph
parent: Fusion
nav_order: 5
---

# OffsetFusionGraph (Maximum-Weight Path Fusion)

## Idea

`OffsetFusionGraph` treats proposed segments as weighted edges and selects a **globally optimal segmentation path**.

It prefers full segments that are supported by multiple engines.

---

## Method

1. Collect all unique boundaries.
2. Treat each proposed segment `(start,end)` as a directed edge.
3. Assign edge weight = number of engines proposing it.
4. Use dynamic programming to compute maximum-weight path.
5. Backtrack to reconstruct best segmentation.

---

## Effect

- Preserves coherent spans supported by engines.
- Avoids mixing unrelated boundary combinations.
- More structured than simple voting.

---

## Recommended for

- When engines sometimes agree on large segments.
- When global coherence matters.

---

## Complexity

Potentially quadratic in number of boundaries.

