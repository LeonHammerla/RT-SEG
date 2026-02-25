import json
import os
import random
from typing import List, Dict, Tuple, Literal, Any
from surrealdb import Surreal, RecordID
import re
from tqdm import tqdm

from rt_segmentation import bp, sdb_login, load_prompt, load_example_trace
from rt_segmentation import SegBase



def export_rt(trace: str,
              trace_id: Any,
              model: str,
              ds_origin: str,
              seg_base_unit: Literal["sent", "clause"] = "clause",):
    offsets = SegBase.get_base_offsets(trace, seg_base_unit=seg_base_unit)
    html_parts = []

    box_style = (
        "border: 1px solid #d1d5db; "
        "border-radius: 8px; "
        "padding: 16px; "
        "background: #ffffff; "
        "box-shadow: 0 2px 4px rgba(0,0,0,0.08); "
        "font-family: system-ui, sans-serif; "
        "line-height: 1.6;"
    )
    sentences = [trace[off[0]:off[-1]] for off in offsets]

    for i, sentence in enumerate(sentences, start=1):
        # Escape double quotes in sentence if any (safe for JSON + HTML)
        safe_sentence = sentence.replace('"', '\\"')

        box = (
            f'<div style="margin-bottom: 20px;">'
            f'<div style="{box_style}">'
            f'{safe_sentence}'
            f'</div></div>'
        )
        html_parts.append(box)

    full_html = "".join(html_parts)

    return {
        "id": trace_id,
        "data": {"html": full_html,
                 "model": model,
                 "ds_origin": ds_origin,
                 "offsets": offsets,
                 "sdb_id": trace_id}
    }


def export_gold_set():
    login_data = sdb_login()
    missing = []
    ds = []
    tasks = [
        "gpqa",
        "aime25",
        # "hle"
             ]
    models = ["gpt-oss:120b", "deepseek-r1:70b",
             "deepseek-r1:32b", "magistral:24b",
             "gpt-oss:20b", "qwen3:1.7b",
             "qwen3:8b", "qwen3:32b",
             "deepseek-r1:8b", "deepseek-r1:1.5b"]
    for task in tqdm(tasks, desc="Exporting gold set"):
        for model in models:
            with Surreal(login_data["url"]) as db:
                db.signin({"username": login_data["user"], "password": login_data["pwd"]})
                db.use(login_data["ns"], login_data["db"])
                max_len = 20000
                tries = 0
                while tries < 5:
                    res = db.query(f'SELECT rt, id, ds_origin, model from rtrace where string::len(rt) < {max_len} and model="{model}" and ds_origin="{task}" and correct=true')
                    if res:
                        target = random.choice(res)
                        ds.append(export_rt(target.get("rt"), target.get("id").id, model, task))
                        break
                    else:
                        print(f"Missing data :: {model} {task}")
                    tries += 1
                    max_len += 10000

    model = f"human_stage1"
    with Surreal(login_data["url"]) as db:
        db.signin({"username": login_data["user"], "password": login_data["pwd"]})
        db.use(login_data["ns"], login_data["db"])
        res = db.query(
            f'SELECT rt, id, ds_origin, model from rtrace where string::len(rt) < 20000 and model="{model}" and ds_origin="nemo"')
        targets = random.sample(res, 4)
        for target in targets:
            ds.append(export_rt(target.get("rt"), target.get("id").id, model, "nemo"))

    print(len(ds), len(missing))

    with open(f"{bp()}/data/label_studio/ls_data.json", "w") as f:
        json.dump(ds, f, indent=4)


def export_rt_rf(trace: str,
                 seg_base_unit: Literal["sent", "clause"] = "clause",):
    offsets = SegBase.get_base_offsets(trace, seg_base_unit=seg_base_unit)
    html_parts = []

    box_style = (
        "border: 1px solid #d1d5db; "
        "border-radius: 8px; "
        "padding: 16px; "
        "background: #ffffff; "
        "box-shadow: 0 2px 4px rgba(0,0,0,0.08); "
        "font-family: system-ui, sans-serif; "
        "line-height: 1.6;"
    )
    sentences = [trace[off[0]:off[-1]] for off in offsets]

    for i, sentence in enumerate(sentences, start=1):
        # Escape double quotes in sentence if any (safe for JSON + HTML)
        safe_sentence = sentence.replace('"', '\\"')

        box = (
            f'<div style="margin-bottom: 20px;">'
            f'<div style="{box_style}">'
            f'{safe_sentence}'
            f'</div></div>'
        )
        html_parts.append(box)

    full_html = "".join(html_parts)

    return full_html, offsets


def export_rf_data_gold_set():
    ds = []
    files = os.listdir(f"{bp()}/data/label_studio/rf_data")
    for file in files:
        with open(f"{bp()}/data/label_studio/rf_data/{file}", "r") as f:
            data = json.load(f)

        full_html, offsets = export_rt_rf(data["raw_text"]["response"])
        ds.append({
            "id": data["doc_id"],
            "data": {"html": full_html,
                     "offsets": offsets,
                     "origin_id": data["doc_id"]}
        })

    print(len(ds))

    with open(f"{bp()}/data/label_studio/ls_data_rf.json", "w") as f:
        json.dump(ds, f, indent=4)


def extract_first_index(xpath_string):
    match = re.search(r'\[(\d+)\]', xpath_string)
    if match:
        return int(match.group(1))

    return None


def import_annotated_data():
    anno_id_map = {"ve": 8, "ha": 1}
    id_anno_map = {v: k for k, v in anno_id_map.items()}

    login_data = sdb_login()
    with open(f"{bp()}/data/label_studio/annotated_data.json", "r") as f:
        data = json.load(f)

    with Surreal(login_data["url"]) as db:
        db.signin({"username": login_data["user"], "password": login_data["pwd"]})
        db.use(login_data["ns"], login_data["db"])
        for (k, v) in anno_id_map.items():
            db.query(f"REMOVE TABLE thought_anchor_gold_{k};")
            db.query(f"DEFINE TABLE thought_anchor_gold_{k} SCHEMALESS;")
            db.query(f"DEFINE INDEX idx_id ON thought_anchor_gold_{k} FIELDS id;")

            db.query(f"REMOVE TABLE has_thought_anchor_gold_{k};")
            db.query(f"DEFINE TABLE has_thought_anchor_gold_{k} SCHEMALESS TYPE RELATION IN rtrace OUT thought_anchor_gold_{k};")
            db.query(f"DEFINE INDEX idx_rt_id ON has_thought_anchor_gold_{k} FIELDS id;")
            db.query(f"DEFINE INDEX idx_rt_in ON has_thought_anchor_gold_{k} FIELDS in;")
            db.query(f"DEFINE INDEX idx_rt_out ON has_thought_anchor_gold_{k} FIELDS out;")


    for sample in tqdm(data, desc="Uploading Results"):
        sample_offsets = sample["data"]["offsets"]
        sample_id = sample["data"]["origin_id"]

        with Surreal(login_data["url"]) as db:
            db.signin({"username": login_data["user"], "password": login_data["pwd"]})
            db.use(login_data["ns"], login_data["db"])

            record_id = RecordID("rtrace", sample_id)

            rec = db.query(f"SELECT * FROM {record_id};")[0]

            for annotation in sample["annotations"]:
                sample_annotator_id = annotation["completed_by"]
                annotation_offsets = []
                annotation_labels = []
                for result in annotation["result"]:
                    start_clause = extract_first_index(result["value"]["start"]) - 1
                    end_clause = extract_first_index(result["value"]["end"]) - 1

                    start_offset = sample_offsets[start_clause][0] + result["value"]["startOffset"]
                    end_offset = sample_offsets[end_clause][0] + result["value"]["endOffset"]
                    annotation_offsets.append((start_offset, end_offset))
                    annotation_labels.append(result["value"]["hypertextlabels"])

                split_id = RecordID(f"thought_anchor_gold_{id_anno_map[sample_annotator_id]}", sample_id)
                db.upsert(split_id, {"split": annotation_offsets, "labels": annotation_labels})
                db.insert_relation(f"has_thought_anchor_gold_{id_anno_map[sample_annotator_id]}", {"in": record_id, "out": split_id})
                for offset in annotation_offsets:
                    print(rec["rt"][offset[0]:offset[1]])
                    print(20*"-")





if __name__ == "__main__":
    export_gold_set()