---
layout: default
title: Fusion
nav_order: 5
---

# Offset Fusion Strategies

RT-SEG supports multi-engine late fusion.

Available aligners:

- OffsetFusionGraph
- OffsetFusionFuzzy
- OffsetFusionIntersect
- OffsetFusionMerge
- OffsetFusionVoting

| Strategy | Behavior |
|----------|----------|
| Intersect | Conservative |
| Merge | Permissive |
| Voting / Graph | Balanced |