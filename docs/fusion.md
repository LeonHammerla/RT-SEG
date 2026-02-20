---
layout: default
title: Fusion
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