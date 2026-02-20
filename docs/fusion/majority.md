---
layout: default
title: OffsetFusionVoting
parent: Fusion
nav_order: 3
---

# OffsetFusionVoting (Majority Fusion)

## Idea

`OffsetFusionVoting` keeps boundaries that receive enough votes across engines.

It balances recall and precision.

---

## Method

1. Count how often each boundary appears.
2. Keep boundaries where:

```

vote_count >= threshold

```

3. Default threshold:
```

majority = n_tools // 2 + 1

```

4. Convert kept boundaries to segments.

---

## Parameter

- `threshold: int | None`
- `None` → strict majority
- lower → more permissive
- higher → more conservative

---

## Effect

- Balanced segmentation.
- Robust default for multi-engine setups.

---

## Recommended for

- General-purpose late fusion.
- Ensembles of 3+ engines.
