import json
import os

from surrealdb import Surreal, RecordID

from tqdm import tqdm

from .seg_utils import bp, sdb_login, load_prompt, load_example_trace


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

