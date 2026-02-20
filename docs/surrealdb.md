---
layout: default
title: Reproducibility
has_children: true
nav_order: 10
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
