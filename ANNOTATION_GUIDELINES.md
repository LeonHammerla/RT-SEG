# Reasoning Trace Segmentation Guidelines

## 1. Task Overview

You are given a *reasoning trace* produced by a language model.  
Your task is to segment the trace into **reasoning segments** by inserting boundaries **between sentences**.

Each segment should correspond to **one coherent reasoning step**.

You do **not** need to judge correctness, logic, or factual accuracy.

---

## 2. Definition of a Reasoning Segment

> A **reasoning segment** is a contiguous span of text that performs a *single epistemic function* in the reasoning process.

An epistemic function describes *what role* a span plays in the reasoning (e.g., setting up assumptions, deriving a consequence, revising a claim).

---

## 3. When to Insert a Boundary

Insert a boundary **between two sentences** if the later sentence introduces a **new epistemic function**.

Boundaries may occur even if:
- [the topic stays the same] ?
- the same entities are discussed
- similar vocabulary is reused

---

## 4. Epistemic Functions (Reference Inventory)

Use the following functions as guidance for identifying boundaries.  
You **do not need to label them**, only decide whether the function changes.

1. **Assumption / Setup**  
   Introducing premises, constraints, or known facts. [Grounds, Backing, Warrant]

2. **Intermediate Inference**  
   Deriving new information from prior content. [Claim]

3. **Contrast / Objection**  
   Introducing alternatives, counterarguments, or limitations. [Rebuttal, Qualifier]

4. **Revision / Correction**  
   Retracting, correcting, or reconsidering earlier reasoning.

5. **Goal Management**  
   Stating subgoals, plans, or next reasoning steps.

6. **Conclusion / Answer**  
   Presenting a final result or summarizing outcome.

---

## 5. Hard Constraints

- Boundaries may only be placed **between sentences**
- **Do not split within a sentence**
- Segments must be **contiguous and non-overlapping**
- Every character in the text must belong to exactly one segment

---

## 6. Granularity Rules

- [**Do not split** minor paraphrases or restatements] ?
- **Do not split** arithmetic or symbolic chains unless interrupted
- **Do split** when the *role* of the text changes, even subtly
- When in doubt, **do not split**

---

## 7. Literature Examples
Overall:
- discourse segmentation
- argumentation / reasoning structure
- cognitive models of problem solving

### Discourse Segmentation

Justify:
- sentence level segmentation
- functional shifts
- non-topical boundaries

#### Example 1: Mann & Thompson (1988) — RST (Rethorical Structure Theory)
Defines:
- Introduces discourse units defined by function, not topic.
```bibtex
@article{MANNTHOMPSON+1988+243+281,
    url = {https://doi.org/10.1515/text.1.1988.8.3.243},
    title = {Rhetorical Structure Theory: Toward a functional theory of text organization},
    title = {},
    author = {WILLIAM C. MANN and SANDRA A. THOMPSON},
    pages = {243--281},
    volume = {8},
    number = {3},
    journal = {Text - Interdisciplinary Journal for the Study of Discourse},
    doi = {doi:10.1515/text.1.1988.8.3.243},
    year = {1988},
    lastchecked = {2026-01-22}
}
```

#### Example 2: Carlson, Marcu & Okurowski (2001) — Building a Discourse-Tagged Corpus in the Framework of Rhetorical Structure Theory
Defines:
- Defines Elementary Discourse Units (EDUs), segmented conservatively.
```bibtex
@inproceedings{carlson-etal-2001-building,
    title = "Building a Discourse-Tagged Corpus in the Framework of {R}hetorical {S}tructure {T}heory",
    author = "Carlson, Lynn  and
      Marcu, Daniel  and
      Okurovsky, Mary Ellen",
    booktitle = "Proceedings of the Second {SIG}dial Workshop on Discourse and Dialogue",
    year = "2001",
    url = "https://aclanthology.org/W01-1605/"
}
```

### Argumentation & reasoning structure (your conceptual foundation)

Justify:
- inference vs. premise vs. conclusion
- revision and objection as first-class units
- 
#### Example 1: Toulmin (1958) - The Uses of Argument
Defines:
- Classic model separating claims, grounds, warrants (inference segments, contrast/objection, conclusion)
```bibtex
@book{toulmin2003uses,
  title={The uses of argument},
  author={Toulmin, Stephen E},
  year={2003},
  publisher={Cambridge university press}
}
```
and
```bibtex
@article{karbach1987using,
  title={Using Toulmin's model of argumentation},
  author={Karbach, Joan},
  journal={Journal of Teaching Writing},
  volume={6},
  number={1},
  pages={81--92},
  year={1987}
}
```
#### Example 2: Stede (2011) — Argumentation mining
Defines:
- Explicitly argues that argument units are discourse-level spans, not logical steps.
```bibtex
@book{stede2019argumentation,
  title={Argumentation mining},
  author={Stede, Manfred and Schneider, Jodi and Hirst, Graeme},
  year={2019},
  publisher={Springer}
}
```
#### Example 3: Argument Mining Overview
Defines:
- Overview of argument mining literature, might be helpful?
```bibtex
@article{lawrence-reed-2019-argument,
    title = "Argument Mining: A Survey",
    author = "Lawrence, John  and
      Reed, Chris",
    journal = "Computational Linguistics",
    volume = "45",
    number = "4",
    month = dec,
    year = "2019",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/J19-4006/",
    doi = "10.1162/coli_a_00364",
    pages = "765--818",
    abstract = "Argument mining is the automatic identification and extraction of the structure of inference and reasoning expressed as arguments presented in natural language. Understanding argumentative structure makes it possible to determine not only what positions people are adopting, but also why they hold the opinions they do, providing valuable insights in domains as diverse as financial market prediction and public relations. This survey explores the techniques that establish the foundations for argument mining, provides a review of recent advances in argument mining techniques, and discusses the challenges faced in automatically extracting a deeper understanding of reasoning expressed in language in general."
}
```

### Reasoning, problem solving & cognitive steps

Justify:
- “epistemic function” language

#### Example 1: Newell & Simon (1971) — Human problem solving
Defines:
- Reasoning proceeds via state transitions, not atomic inferences (goal management, revision, multi.sentence steps)
```bibtex
@article{simon1971human,
  title={Human problem solving: The state of the theory in 1970.},
  author={Simon, Herbert A and Newell, Allen},
  journal={American psychologist},
  volume={26},
  number={2},
  pages={145},
  year={1971},
  publisher={American Psychological Association}
}
```
#### Example 2: Polya (1945) — How to Solve It
Defines:
- Defines reasoning in phases: understand, plan, execute, review
```bibtex
@article{polya1957solve,
  title={How to solve it},
  author={Polya, George},
  year={1957},
  publisher={Princeton Press}
}
```

### Recent LLM / reasoning trace–adjacent work

Justify:
- Introduces reasoning traces as explicit text objects.

#### Example 1: Wei et al. (2022) — Chain-of-Thought
Defines:
- Introduces CoT traces
```bibtex
@misc{wei2023chainofthoughtpromptingelicitsreasoning,
      title={Chain-of-Thought Prompting Elicits Reasoning in Large Language Models}, 
      author={Jason Wei and Xuezhi Wang and Dale Schuurmans and Maarten Bosma and Brian Ichter and Fei Xia and Ed Chi and Quoc Le and Denny Zhou},
      year={2023},
      eprint={2201.11903},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2201.11903}, 
}
```
---

## 8. Canonical Examples

### Example 1: Inference shift (split)

