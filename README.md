<p align="center">
  <img src="docs/assets/logo.svg" width="30%" style="max-width: 400px;">
</p>

# RT-SEG — Reasoning Trace Segmentation

`rt_seg` is a **Python 3.12.x** package for segmenting *reasoning traces* into coherent chunks and (optionally) assigning a label to each chunk.

The main entry point is:

```
RTSeg
```

(from `rt_segmentation.seg_factory`)

It orchestrates one or more **segmentation engines** and — if multiple engines are used — an **offset aligner** that fuses their boundaries into a single segmentation.

---


https://github.com/user-attachments/assets/396e4684-1792-4367-b275-ac515e6abbee


---

# Installation

## Install from PyPI (once published)

```bash
pip install rt-seg
```

## Development Install (repo checkout)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# Core Concepts

<p align="center">
  <img src="docs/assets/arch.svg" width="100%">
</p>

## What `RTSeg` Returns

`RTSeg(trace)` produces:

* `offsets`: `list[tuple[int, int]]` — character offsets into the trace
* `labels`: `list[str]` — one label per segment

You can reconstruct segments via:

```python
segments = [trace[s:e] for (s, e) in offsets]
```

---

## Segmentation Base Unit (`seg_base_unit`)

Most engines operate on a base segmentation first:

* `"clause"` (default) → finer granularity
* `"sent"` → coarser segmentation

---

# Quickstart — Single Engine

```python
from rt_seg import RTSeg
from rt_seg import RTRuleRegex

trace = "First step... Then second step... Finally conclude."

segmentor = RTSeg(
    engines=RTRuleRegex,
    seg_base_unit="clause",
)

offsets, labels = segmentor(trace)

for (s, e), label in zip(offsets, labels):
    print(label, "=>", trace[s:e])
```

---

# Multiple Engines + Late Fusion

If you pass multiple engines, you must provide an **aligner**.

```python
from rt_seg import RTSeg
from rt_seg import RTRuleRegex
from rt_seg import RTBERTopicSegmentation
from rt_seg import OffsetFusionGraph

segmentor = RTSeg(
    engines=[RTRuleRegex, RTBERTopicSegmentation],
    aligner=OffsetFusionGraph,
    label_fusion_type="concat",  # or "majority"
    seg_base_unit="clause",
)

offsets, labels = segmentor(trace)
```

## Label Fusion Modes

* `"majority"` — choose most frequent label
* `"concat"` — concatenate labels (useful for debugging)

---

# Available Engines

## Rule-Based

* `RTRuleRegex`
* `RTNewLine`

## Probabilistic

* `RTLLMForcedDecoderBased`
* `RTLLMSurprisal`
* `RTLLMEntropy`
* `RTLLMTopKShift`
* `RTLLMFlatnessBreak`

## LLM Discourse / Reasoning Schemas

* `RTLLMThoughtAnchor`
* `RTLLMReasoningFlow`
* `RTLLMArgument`

## LLM 

* `RTLLMOffsetBased`
* `RTLLMSegUnitBased`

## PRM-Based

* `RTPRMBase`
  
## Topic / Semantic / NLI

* `RTBERTopicSegmentation`
* `RTEmbeddingBasedSemanticShift`
* `RTEntailmentBasedSegmentation`
* `RTZeroShotSeqClassification`
* `RTZeroShotSeqClassificationRF`
* `RTZeroShotSeqClassificationTA`

---

# Engine Configuration

You can override engine parameters at call time:

```python
offsets, labels = segmentor(
    trace,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    chunk_size=200,
)
```

---

# Available Aligners

* `OffsetFusionGraph`
* `OffsetFusionFuzzy`
* `OffsetFusionIntersect`
* `OffsetFusionMerge`
* `OffsetFusionVoting`

| Strategy               | Behavior               |
| ---------------------- | ---------------------- |
| Intersect              | Conservative           |
| Merge                  | Permissive             |
| Voting / Graph / Fuzzy | Balanced (recommended) |

---

# Implementing a Custom Engine

```python
from typing import Tuple, List
from rt_seg import SegBase

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

---

# Running the TUI (Without Docker)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m tui
```

If needed:

```bash
python src/tui.py
```

---

# SurrealDB (Optional — Reproducible Experiments)

Required only for full experiment pipeline.

---

## 1️⃣ Start SurrealDB (Docker Recommended)

```bash
docker run --rm -it \
  -p 8000:8000 \
  -v "$(pwd)/data:/data" \
  surrealdb/surrealdb:latest \
  start --user root --pass root file:/data/surreal.db
```

Endpoints:

* WebSocket: `ws://127.0.0.1:8000/rpc`
* HTTP: `http://127.0.0.1:8000`

---

## 2️⃣ Import Database Snapshot

```bash
surreal import \
  --endpoint ws://127.0.0.1:8000/rpc \
  --username root \
  --password root \
  --namespace NR \
  --database RT \
  ./data/YOUR_EXPORT_FILE.surql
```

⚠️ Make sure namespace/database match your config.

---

## 3️⃣ Configure `data/sdb_login.json`

```json
{
  "user": "root",
  "pwd": "root",
  "ns": "NR",
  "db": "RT",
  "url": "ws://127.0.0.1:8000/rpc"
}
```

---

## 4️⃣ Run Experiment Scripts

```bash
python src/eval_main.py
python src/evo.py
```

---

# Docker + GPU Setup

## Requirements

* Linux
* NVIDIA GPU
* NVIDIA driver
* Docker
* NVIDIA Container Toolkit

Verify:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## CUDA Compatibility Rule

Host driver CUDA ≥ Container CUDA

| Host | Container | Result |
| ---- | --------- | ------ |
| 12.8 | 12.4      | ✅      |
| 12.8 | 13.1      | ❌      |
| 13.x | 12.4      | ✅      |

---

## Build Image

```bash
docker build -f docker/Dockerfile -t rt-seg:gpu .
```

---

## Run

```bash
./run_tui_app_docker.sh
```

Internally:

```bash
docker run -it --rm --gpus all rt-seg:gpu
```

---

# Summary

RT-SEG provides:

* Modular segmentation engines
* Late fusion strategies
* LLM-based reasoning segmentation
* Reproducible DB-backed experiments
* GPU Docker deployment

---
