---
layout: default
title: RT-Seg
nav_order: 1
---

<p align="center">
  <img src="assets/logo.svg" width="40%" />
</p>

# RT-SEG — Reasoning Trace Segmentation

RT-SEG is a modular research framework for the segmentation of reasoning traces into coherent structural and semantic units.

It provides:

- Rule-based segmentation
- Probabilistic and distributional boundary detection
- LLM-driven discourse schemas
- Topic and semantic shift segmentation
- Late-fusion of heterogeneous engines
- Reproducible database-backed experimentation

---

## Design Philosophy

RT-SEG treats reasoning traces as structured discourse objects.  
Segmentation hypotheses are represented as character-level offsets to guarantee:

- deterministic reconstruction
- reproducible evaluation
- engine-agnostic comparison
- systematic late fusion

---

## Quickstart

```python
from rt_seg import RTSeg, RTRuleRegex

segmentor = RTSeg(engines=RTRuleRegex)
offsets, labels = segmentor("Example reasoning trace...")
```

See the navigation sidebar for full documentation.

