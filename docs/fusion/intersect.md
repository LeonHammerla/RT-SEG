---
layout: default
title: OffsetFusionIntersect
parent: Fusion
nav_order: 2
---

# OffsetFusionIntersect (Strict Consensus)

## Idea

`OffsetFusionIntersect` keeps only boundaries that are proposed by **all engines**.

This is the most conservative strategy.

---

## Method

1. Compute boundary sets per engine.
2. Compute set intersection.
3. Ensure coverage:
   - Add `0` if missing.
   - Add final trace boundary if missing.
4. Convert to segments.

---

## Effect

- High precision.
- Low recall.
- Produces coarse segmentation.

---

## Recommended for

- When engines are noisy.
- When precision is critical.
