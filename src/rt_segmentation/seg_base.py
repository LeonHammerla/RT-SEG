from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from surrealdb import Surreal, RecordID
from tqdm import tqdm

from .seg_utils import bp, sdb_login, load_prompt, load_example_trace


class SegBase(ABC):

    @staticmethod
    @abstractmethod
    def _segment(trace: str, **kwargs) -> List[Tuple[int, int]]:
        pass

    @staticmethod
    def sdb_segment(instance: "SegBase",
                exp_id: str,
                clear: bool = True,
                **kwargs):
        login_data = sdb_login()
        with Surreal(login_data["url"]) as db:
            db.signin({"username": login_data["user"], "password": login_data["pwd"]})
            db.use(login_data["ns"], login_data["db"])
            if clear:
                db.query(f"REMOVE TABLE {exp_id};")
                db.query(f"DEFINE TABLE {exp_id} SCHEMALESS;")
                db.query(f"DEFINE INDEX idx_id ON {exp_id} FIELDS id;")

                db.query(f"REMOVE TABLE has_{exp_id};")
                db.query(f"DEFINE TABLE has_{exp_id} SCHEMALESS TYPE RELATION IN rtrace OUT {exp_id};")
                db.query(f"DEFINE INDEX idx_rt_id ON has_{exp_id} FIELDS id;")
                db.query(f"DEFINE INDEX idx_rt_in ON has_{exp_id} FIELDS in;")
                db.query(f"DEFINE INDEX idx_rt_out ON has_{exp_id} FIELDS out;")

            results = db.query(
                f"SELECT *, ->has_{exp_id}->{exp_id}.* as seg from rtrace where ->has_{exp_id}->{exp_id} == [] and string::len(rt) < 20000 ")

            for res in tqdm(results, desc=f"Segmenting traces :: {exp_id}"):
                rt = res.get("rt")
                try:
                    offsets = instance._segment(trace=rt, **kwargs)
                except Exception as e:
                    print(e)
                    continue

                split_id = RecordID(f"{exp_id}", res.get("id").id)
                db.upsert(split_id, {"split": offsets})
                db.insert_relation(f"has_{exp_id}", {"in": res.get("id"), "out": split_id})