import copy
import json
import os

from datasets import load_dataset
from surrealdb import Surreal, RecordID
import re
from tqdm import tqdm

from .seg_utils import bp, sdb_login, load_prompt, load_example_trace


BEGIN_THOUGHT = "<|begin_of_thought|>"
END_THOUGHT = "<|end_of_thought|>"


def upload_rf_data(clear: bool = True):
    login_data = sdb_login()
    with Surreal(login_data["url"]) as db:
        db.signin({"username": login_data["user"], "password": login_data["pwd"]})
        db.use(login_data["ns"], login_data["db"])
        if clear:
            db.query(f"REMOVE TABLE sample;")
            db.query(f"DEFINE TABLE sample SCHEMALESS;")
            db.query(f"DEFINE INDEX idx_id ON sample FIELDS id;")

            db.query(f"REMOVE TABLE rtrace;")
            db.query(f"DEFINE TABLE rtrace SCHEMALESS;")
            db.query(f"DEFINE INDEX idx_id ON rtrace FIELDS id;")

            db.query(f"REMOVE TABLE has_rt;")
            db.query(f"DEFINE TABLE has_rt SCHEMALESS TYPE RELATION IN sample OUT rtrace;")
            db.query(f"DEFINE INDEX idx_has_rt_id ON has_rt FIELDS id;")
            db.query(f"DEFINE INDEX idx_has_rt_in ON has_rt FIELDS in;")
            db.query(f"DEFINE INDEX idx_has_rt_out ON has_rt FIELDS out;")

            db.query(f"REMOVE TABLE reasoning_flow_gold;")
            db.query(f"DEFINE TABLE reasoning_flow_gold SCHEMALESS;")
            db.query(f"DEFINE INDEX idx_id ON reasoning_flow_gold FIELDS id;")

            db.query(f"REMOVE TABLE has_reasoning_flow_gold;")
            db.query(f"DEFINE TABLE has_reasoning_flow_gold SCHEMALESS TYPE RELATION IN rtrace OUT reasoning_flow_gold;")
            db.query(f"DEFINE INDEX idx_reasoning_flow_gold_id ON has_reasoning_flow_gold FIELDS id;")
            db.query(f"DEFINE INDEX idx_reasoning_flow_gold_in ON has_reasoning_flow_gold FIELDS in;")
            db.query(f"DEFINE INDEX idx_reasoning_flow_gold_out ON has_reasoning_flow_gold FIELDS out;")


        files = os.listdir(f"{bp()}/data/label_studio/rf_data")
        for file in tqdm(files, desc="Uploading RF data"):
            with open(f"{bp()}/data/label_studio/rf_data/{file}", "r") as f:
                data = json.load(f)

            sample_id = RecordID("sample", data["doc_id"])
            db.upsert(sample_id, {"question": data["raw_text"]["question"],
                                        "meta": data["metadata"]})
            trace_id = RecordID("rtrace", data["doc_id"])
            db.upsert(trace_id, {"rt": data["raw_text"]["response"],
                                 "model": data["metadata"]["generator"],
                                 "source": data["metadata"]["source"],
                                 "domain": data["metadata"]["domain"],
                                 "batch": data["metadata"]["batch"]})
            db.insert_relation(
                "has_rt", {"in": sample_id, "out": trace_id}
            )

            offsets, labels = [], []
            for node in data["nodes"]:
                if node["source"] == "response":
                    offsets.append((node["start"], node["end"]))
                    labels.append(node["label"])

            split_id = RecordID("reasoning_flow_gold", data["doc_id"])
            db.upsert(split_id, {"split": offsets,
                                 "labels": labels})

            db.insert_relation(
                "has_reasoning_flow_gold", {"in": trace_id, "out": split_id}
            )



def get_reasoning_trace(sample):
    response = sample["conversations"][1]["value"]
    try:
        start = response.index(BEGIN_THOUGHT) + len(BEGIN_THOUGHT)
        end = response.index(END_THOUGHT, start)
    except ValueError as exc:
        # raise ValueError("Reasoning trace markers not found in sample") from exc
        return ""

    return response[start:end].strip()


def add_reasoning_trace_length(sample):
    reasoning_trace = get_reasoning_trace(sample)
    return {"rt_len": len(reasoning_trace.split())}


def load_rf_questions():
    files = os.listdir(f"{bp()}/data/label_studio/rf_data")
    questions = []
    traces = []
    nodes = []
    for file in tqdm(files, desc="Uploading RF data"):
        with open(f"{bp()}/data/label_studio/rf_data/{file}", "r") as f:
            data = json.load(f)
            questions.append(data["raw_text"]["question"])
            traces.append(data["raw_text"]["response"])
            nodes.append(data["nodes"])
    return questions, traces, nodes


def contains_chinese(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def upload_rf_data_extended(clear: bool = True, n_samples: int = 100):
    login_data = sdb_login()
    with Surreal(login_data["url"]) as db:
        db.signin({"username": login_data["user"], "password": login_data["pwd"]})
        db.use(login_data["ns"], login_data["db"])
        if clear:
            db.query(f"REMOVE TABLE sample;")
            db.query(f"DEFINE TABLE sample SCHEMALESS;")
            db.query(f"DEFINE INDEX idx_id ON sample FIELDS id;")

            db.query(f"REMOVE TABLE rtrace;")
            db.query(f"DEFINE TABLE rtrace SCHEMALESS;")
            db.query(f"DEFINE INDEX idx_id ON rtrace FIELDS id;")

            db.query(f"REMOVE TABLE has_rt;")
            db.query(f"DEFINE TABLE has_rt SCHEMALESS TYPE RELATION IN sample OUT rtrace;")
            db.query(f"DEFINE INDEX idx_has_rt_id ON has_rt FIELDS id;")
            db.query(f"DEFINE INDEX idx_has_rt_in ON has_rt FIELDS in;")
            db.query(f"DEFINE INDEX idx_has_rt_out ON has_rt FIELDS out;")

            db.query(f"REMOVE TABLE reasoning_flow_gold;")
            db.query(f"DEFINE TABLE reasoning_flow_gold SCHEMALESS;")
            db.query(f"DEFINE INDEX idx_id ON reasoning_flow_gold FIELDS id;")

            db.query(f"REMOVE TABLE has_reasoning_flow_gold;")
            db.query(f"DEFINE TABLE has_reasoning_flow_gold SCHEMALESS TYPE RELATION IN rtrace OUT reasoning_flow_gold;")
            db.query(f"DEFINE INDEX idx_reasoning_flow_gold_id ON has_reasoning_flow_gold FIELDS id;")
            db.query(f"DEFINE INDEX idx_reasoning_flow_gold_in ON has_reasoning_flow_gold FIELDS in;")
            db.query(f"DEFINE INDEX idx_reasoning_flow_gold_out ON has_reasoning_flow_gold FIELDS out;")

    ds = load_dataset("NovaSky-AI/Sky-T1_data_17k")["train"]
    ds = ds.map(add_reasoning_trace_length)
    ds = ds.sort("rt_len")

    rf_questions, rt_traces, all_nodes = load_rf_questions()

    all_samples = copy.deepcopy(rf_questions)
    all_traces = copy.deepcopy(rt_traces)

    rf_questions = set(rf_questions)
    for sample in tqdm(ds, desc="..."):
        if len(all_samples) == n_samples:
            break
        if sample["rt_len"] < 200:
            continue
        if contains_chinese(sample["conversations"][1]["value"]):
            continue
        if sample["conversations"][0]["value"] in rf_questions:
            pass
        else:
            all_samples.append(sample["conversations"][0]["value"])
            all_traces.append(get_reasoning_trace(sample))
            all_nodes.append(None)
            rf_questions.add(sample["conversations"][0]["value"])

    print(len(all_samples), len(all_traces), len(all_nodes))
    print([len(tr) for tr in all_traces])
    print(sum([len(rt) for rt in all_traces[:31]]))
    print(sum([len(rt) for rt in all_traces[31:]]))

    with Surreal(login_data["url"]) as db:
        db.signin({"username": login_data["user"], "password": login_data["pwd"]})
        db.use(login_data["ns"], login_data["db"])
        for idx, (question, trace, nodes) in tqdm(enumerate(zip(all_samples, all_traces, all_nodes)), desc="Uploading RF data"):

            sample_id = RecordID("sample", idx)
            db.upsert(sample_id, {"question": question})
            trace_id = RecordID("rtrace", idx)
            db.upsert(trace_id, {"rt": trace})
            db.insert_relation(
                "has_rt", {"in": sample_id, "out": trace_id}
            )

            if nodes is not None:
                offsets, labels = [], []
                for node in nodes:
                    if node["source"] == "response":
                        offsets.append((node["start"], node["end"]))
                        labels.append(node["label"])

                split_id = RecordID("reasoning_flow_gold", idx)
                db.upsert(split_id, {"split": offsets,
                                     "labels": labels})

                db.insert_relation(
                    "has_reasoning_flow_gold", {"in": trace_id, "out": split_id}
                )
                idx += 1


if __name__ == "__main__":
    upload_rf_data_extended()