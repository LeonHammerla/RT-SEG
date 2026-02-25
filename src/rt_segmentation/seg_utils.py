import json
import os
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def bp():
    return os.path.realpath(os.path.join(os.path.realpath(__file__), "../../.."))


@lru_cache(maxsize=1)
def sdb_login():
    with open(f"{bp()}/data/sdb_login.json", "r") as f:
        config = json.load(f)
    return config


@lru_cache(maxsize=10)
def load_prompt(prompt_id: str):
    # with open(f"{bp()}/data/prompts.json", "r") as f:
    with resources.files("rt_segmentation").joinpath("prompts.json").open("r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts[prompt_id]


@lru_cache(maxsize=10)
def load_example_trace(trace_id: str):
    with resources.files("rt_segmentation").joinpath("example_traces.json").open("r", encoding="utf-8") as f:
    # with open(f"{bp()}/data/example_traces.json", "r") as f:
        traces = json.load(f)
    return traces[trace_id]

