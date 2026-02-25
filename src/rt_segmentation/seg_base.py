from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Literal

from .base_segmentor import UnitSegmentor


class SegBase(ABC):
    @staticmethod
    def get_base_offsets(trace: str,
                         seg_base_unit: Literal["sent", "clause"]):
        if seg_base_unit == "clause":
            offsets = UnitSegmentor.get_math_aware_clauses(trace)
        elif seg_base_unit == "sent":
            offsets = UnitSegmentor.get_math_aware_sents(trace)
        else:
            raise ValueError(f"Invalid seg_base_unit: {seg_base_unit}")
        return offsets


    @staticmethod
    @abstractmethod
    def _segment(trace: str, **kwargs) -> Tuple[List[Tuple[int, int]], List[str]]:
        pass

    """
    @staticmethod
    def sdb_segment_native_ds(instance: "SegBase",
                exp_id: str,
                clear: bool = True,
                **kwargs):
        login_data = sdb_login()
        login_data["db"] = "RT"
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
                    s = time.time()
                    offsets = instance._segment(trace=rt, **kwargs)
                    e = time.time()
                except Exception as e:
                    print(e)
                    continue

                split_id = RecordID(f"{exp_id}", res.get("id").id)
                db.upsert(split_id, {"split": offsets, "ptime": e - s})
                db.insert_relation(f"has_{exp_id}", {"in": res.get("id"), "out": split_id})
    """


