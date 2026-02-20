---
layout: default
title: Extending
nav_order: 6
---

# Implementing a Custom Engine

```python
from typing import Tuple, List
from rt_segmentation.seg_base import SegBase

class MyEngine(SegBase):
    @staticmethod
    def _segment(trace: str, **kwargs) -> Tuple[List[tuple[int, int]], List[str]]:
        offsets = [(0, len(trace))]
        labels = ["UNK"]
        return offsets, labels
```

## Using Base Offsets

```python
base_offsets = SegBase.get_base_offsets(trace, seg_base_unit="clause")
```

---

# Implementing a Custom Aligner

```python
from typing import List, Tuple

class MyOffsetFusion:
    @staticmethod
    def fuse(engine_offsets: List[List[Tuple[int, int]]], **kwargs):
        return engine_offsets[0]
```
