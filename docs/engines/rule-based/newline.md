---
layout: default
title: RTNewLine
parent: Rule-Based Engines
nav_order: 2
---

# RTNewLine (Paragraph-based segmentation via blank lines)

## Idea

`RTNewLine` is a minimal rule-based baseline that segments a reasoning trace using **formatting structure** rather than lexical or model-based cues.

It splits the trace into segments at **blank-line boundaries**, i.e., occurrences of `\n\n`, which often correspond to paragraph breaks, step breaks, or deliberate spacing inserted by the generator or annotator.

This engine is particularly useful as:
- a fast baseline,
- a preprocessing heuristic when traces already contain meaningful line breaks,
- a robust fallback in environments without model dependencies.

---

## Method (high-level)

1. **Find segment start positions**
   Identify all positions in the trace immediately following either:
   - the start of the string (`\A`), or
   - a blank-line delimiter (`\n\n`)

   Concretely, the engine collects:
   - `positions = [m.end() for m in re.finditer(r'\n\n|\A', trace)]`

2. **Create consecutive spans**
   Pair each start position with the next start position (or end of trace for the last segment):
   - `offsets = zip(positions, positions[1:] + [len(trace)])`

3. **Return offsets**
   Emit character offsets and assign `"UNK"` labels to all segments.

---

## Models used

None. This engine is purely regex-based.

---

## Parameters

`RTNewLine` does not require any configuration parameters. Any `**kwargs` are ignored.

---

## Usage

```python
from rt_seg.seg_factory import RTSeg
from rt_seg.newline import RTNewLine  # adjust import to your package structure

trace = "Step 1: ...\n\nStep 2: ...\n\nFinal: ..."

segmentor = RTSeg(engines=RTNewLine)
offsets, labels = segmentor(trace)

segments = [trace[s:e] for s, e in offsets]
for seg in segments:
    print("---")
    print(seg)
```