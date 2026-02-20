---
layout: default
title: RTLLMArgument|TA|RF
parent: Downstream LLM Engines
nav_order: 1
---

# RTLLMArgument (Prompted LLM labeling for argument-style structure)

## Idea

`RTLLMArgument` is a **downstream prompting-based** segmentation engine: it uses an instruction-tuned LLM to assign a **discourse/argument label** to each base unit (sentence or clause), and then merges adjacent units that share the same predicted label.

In RT-SEG, we use the same *prompted labeling + merge-on-label-change* pattern for multiple downstream schemas, including:

- **Argument mining** (`RTLLMArgument`)
- **Thought Anchor schema** (`RTLLMThoughtAnchor`)
- **Reasoning Flow schema** (`RTLLMReasoningFlow`)

Conceptually, these engines treat segmentation as a **schema-induced partitioning** problem: boundaries occur where the discourse function changes.

---

## Method (high-level)

Given a trace split into base units \(u_1, \dots, u_m\):

1. **Base segmentation**
   Compute base offsets via:
   - `SegBase.get_base_offsets(trace, seg_base_unit=...)`
   Extract strings:
   - `u_i = trace[s_i:e_i]`

2. **Context-window prompting**
   For each target unit \(u_i\), build a local context window of size `context_window` around it:
   - `PREVIOUS_SEGMENT`: first unit in the window, or `[START OF TRACE]`
   - `TARGET_SEGMENT`: the current unit
   - `NEXT_SEGMENT`: last unit in the window, or `[END OF TRACE]`

   A user prompt template (provided externally) is filled as:
   ```text
   PREVIOUS_SEGMENT = ...
   TARGET_SEGMENT   = ...
   NEXT_SEGMENT     = ...


3. **LLM inference (label per base unit)**
   For each prompted instance, query an instruction-tuned LLM using a `system_prompt` and `user_prompt`.
   The engine expects the model to return a JSON object with a `label` field, e.g.:

   ```json
   {"label": "Claim"}
   ```

   The label is extracted via `json.loads(response)["label"]`.

   The engine includes a retry loop (`max_retry`) to handle invalid JSON or empty labels.

4. **Merge adjacent units with identical labels**
   Consecutive base units are merged into a segment as long as their predicted labels are the same.
   Boundaries occur when `label_i != label_{i-1}`.

Output:

* `cleaned_final_offsets`: merged character offsets
* `final_labels`: one label per merged segment

---

## Models used

This engine uses an **instruction-tuned causal LLM** via:

* `AutoModelForCausalLM`
* `AutoTokenizer`

Supported model identifiers in the code:

* `Qwen/Qwen2.5-7B-Instruct-1M`
* `Qwen/Qwen2.5-7B-Instruct`
* `mistralai/Mixtral-8x7B-Instruct-v0.1`

Implementation notes:

* Prompts are formatted using `tokenizer.apply_chat_template(...)`.
* Generation uses `model.generate(..., max_new_tokens=128)`.
* The engine expects **structured JSON** output.

> For reproducibility, the exact prompts (`system_prompt`, `user_prompt`) and model choice should be documented alongside experimental results, as they directly define the labeling scheme.

---

## Inputs and prompts

This engine requires two prompt strings:

* `system_prompt: str`
  A high-level instruction defining the role/task of the model.

* `user_prompt: str`
  A template string that must include the placeholders:

  * `{PREVIOUS_SEGMENT}`
  * `{TARGET_SEGMENT}`
  * `{NEXT_SEGMENT}`

The label space itself is implicitly defined by the prompt (and any schema instructions in it).
In practice, for **argument mining**, prompts typically constrain labels to an explicit set (e.g., Claim / Evidence / Premise / Conclusion), but RT-SEG does not enforce a fixed ontology in code—this is controlled by the prompt.

---

## Key parameters

* `seg_base_unit: Literal["sent", "clause"]`
  Base unit granularity for prompting and labeling.

* `context_window: int` (default: `2`)
  Number of units to include on each side of the target unit when constructing the local context.

* `max_retry: int` (default: `30`)
  Maximum number of retries if generation fails (invalid JSON, missing label, etc.).

* `model_name: str`
  One of the supported instruction-tuned LLMs above.

---

## Usage

```python
from rt_seg.seg_factory import RTSeg
from rt_seg.llm_argument import RTLLMArgument  # adjust import to your package structure

trace = "..."

system_prompt = "You label segments in reasoning traces using an argument mining schema."
user_prompt = """
Given the surrounding context, assign a single argument label to TARGET_SEGMENT.
Return JSON strictly as: {"label": "..."}.

PREVIOUS: {PREVIOUS_SEGMENT}
TARGET: {TARGET_SEGMENT}
NEXT: {NEXT_SEGMENT}
"""

segmentor = RTSeg(engines=RTLLMArgument, seg_base_unit="sent")

offsets, labels = segmentor(
    trace,
    seg_base_unit="sent",
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    context_window=2,
    max_retry=30,
)

segments = [trace[s:e] for s, e in offsets]
```
