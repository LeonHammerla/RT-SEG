---
layout: default
title: TUI (Docker + GPU)
nav_order: 8
---

# TUI Docker + GPU Setup

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
