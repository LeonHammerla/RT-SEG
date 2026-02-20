"""
Adjusted for your new search space:

- Search variables:
  1) subset of ALL_MODELS (now strings like "<ClassName>_clause" or "<ClassName>_sent") -> bitmask
  2) one ALIGNERS element -> aligner_idx

- BASE_UNITS removed

Persistence:
- SQLite with PRIMARY KEY = (mask, aligner_idx)
- Also stores approach_id = get_name(mods, aligner) as the "id" you requested
- Crash-safe + resume: INSERT OR IGNORE

Progress:
- tqdm per epoch (generation)

IMPORTANT:
- With spawn, do NOT load data at import time. Worker initializer loads once per worker.
"""

import os
import time
import sqlite3
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ------------------- YOUR PROJECT IMPORTS (as in your snippet) -------------------
import copy
from functools import lru_cache

from surrealdb import Surreal
from rt_segmentation.seg_eval_utils import clean_offsets, evaluate_approaches_bounding_similarity
from rt_segmentation import (
    OffsetFusionFuzzy, OffsetFusionIntersect, OffsetFusionMerge, OffsetFusionGraph, OffsetFusionVoting,
    sdb_login, bp,
)
# ------------------- IMPORT YOUR PROJECT CODE -------------------
# IMPORTANT: do NOT call load_data() at import time if using spawn.
# We'll load data per-worker in an initializer.
import copy
from functools import lru_cache

from tqdm import tqdm
from surrealdb import Surreal

from rt_segmentation.seg_eval_utils import clean_offsets, evaluate_approaches_bounding_similarity
from rt_segmentation import (
    OffsetFusionFuzzy, OffsetFusionIntersect, OffsetFusionMerge, OffsetFusionGraph, OffsetFusionVoting,
    SegBase,
    RTLLMOffsetBased,
    RTLLMForcedDecoderBased,
    RTLLMSegUnitBased,
    RTRuleRegex,
    RTNewLine,
    RTPRMBase,
    RTLLMReasoningFlow,
    RTLLMArgument,
    RTLLMThoughtAnchor,
    RTLLMEntropy,
    RTLLMTopKShift,
    RTLLMFlatnessBreak,
    RTLLMSurprisal,
    RTBERTopicSegmentation,
    RTZeroShotSeqClassification,
    RTEntailmentBasedSegmentation,
    RTEmbeddingBasedSemanticShift,
    RTZeroShotSeqClassificationTA,
    RTZeroShotSeqClassificationRF,
    sdb_login, bp,
)

# ----------------------------- USER TO FILL THESE -----------------------------
ALL_MODELS: List[Any] = [
    RTLLMOffsetBased,
    RTLLMForcedDecoderBased,
    RTLLMSegUnitBased,
    RTRuleRegex,
    RTNewLine,
    RTPRMBase,
    RTEntailmentBasedSegmentation,
    RTLLMEntropy,
    RTLLMTopKShift,
    RTLLMFlatnessBreak,
    RTLLMSurprisal,
    RTBERTopicSegmentation,
    RTZeroShotSeqClassification,
    RTLLMReasoningFlow,
    RTLLMArgument,
    RTLLMThoughtAnchor,
    RTEmbeddingBasedSemanticShift,
    RTZeroShotSeqClassificationRF,
    RTZeroShotSeqClassificationTA,
]
ALL_MODELS_CL = [f"{mod.__name__}_clause" for mod in ALL_MODELS]
ALL_MODELS_SE = [f"{mod.__name__}_sent" for mod in ALL_MODELS]
ALL_MODELS = ALL_MODELS_CL + ALL_MODELS_SE

ALIGNERS = [OffsetFusionGraph, OffsetFusionFuzzy, OffsetFusionIntersect, OffsetFusionMerge, OffsetFusionVoting]


# ----------------------- GLOBALS (worker-loaded) -----------------------
BASE_DATA = None
BASE_DATA_HUMAN = None
TRACES = None


@lru_cache(maxsize=None)
def load_data(targets: List[str]):
    gold_keys = [
        "thought_anchor_gold_ve",
        "though_anchor_gold_ve",
        "thought_anchor_gold_ha",
        "though_anchor_gold_ha",
    ]


    login_data = sdb_login()
    with Surreal(login_data["url"]) as db:
        db.signin({"username": login_data["user"], "password": login_data["pwd"]})
        db.use(login_data["ns"], login_data["db"])
        res = db.query("SELECT *, ->?->?.* from rtrace")

    traces = []
    human_anno_data = dict()
    model_anno_data = dict()

    for rtrace in tqdm(res, desc="Gathering data"):
        traces.append(rtrace["rt"])
        for anno in rtrace["->?"]["->?"]:
            table_name = anno.get("id").table_name
            if table_name in gold_keys:
                human_anno_data.setdefault(table_name, []).append(clean_offsets(anno["split"], rtrace["rt"]))
            elif table_name in targets:
                model_anno_data.setdefault(table_name, []).append(clean_offsets(anno["split"], rtrace["rt"]))
            else:
                continue

    return model_anno_data, human_anno_data, traces


def score(mods: List[str], aligner) -> float:
    global BASE_DATA, BASE_DATA_HUMAN, TRACES

    segs = [BASE_DATA[mod] for mod in mods]
    if len(segs) == 0:
        return 0.0
    elif len(segs) == 1:
        fused = segs[0]
    else:
        if aligner is not None:
            fused = []
            for eles in zip(*segs):
                fused.append(aligner.fuse(list(eles)))
        else:
            return 0.0

    current_data = copy.deepcopy(BASE_DATA_HUMAN)
    current_data["model"] = fused
    target_data = []
    for idx in range(len(current_data["model"])):
        target_data.append({k: v[idx] for (k, v) in current_data.items()})

    return float(evaluate_approaches_bounding_similarity(TRACES, target_data, window=3))


def get_name(mods: List[str], aligner) -> str:
    return f"{'_'.join(mods)}_{aligner.__name__}"

# ----------------------- CONFIG ENCODING -----------------------
N_MODELS = len(ALL_MODELS)
N_ALIGNERS = len(ALIGNERS)


# ----------------------- GLOBALS (worker-loaded) -----------------------
BASE_DATA = None
BASE_DATA_HUMAN = None
TRACES = None


@lru_cache(maxsize=None)
def load_data(targets: Tuple[str, ...]):
    gold_keys = [
        "thought_anchor_gold_ve",
        "though_anchor_gold_ve",
        "thought_anchor_gold_ha",
        "though_anchor_gold_ha",
    ]

    login_data = sdb_login()
    with Surreal(login_data["url"]) as db:
        db.signin({"username": login_data["user"], "password": login_data["pwd"]})
        db.use(login_data["ns"], login_data["db"])
        res = db.query("SELECT *, ->?->?.* from rtrace")

    traces = []
    human_anno_data = dict()
    model_anno_data = dict()

    for rtrace in tqdm(res, desc="Gathering data"):
        traces.append(rtrace["rt"])
        for anno in rtrace["->?"]["->?"]:
            table_name = anno.get("id").table_name
            if table_name in gold_keys:
                human_anno_data.setdefault(table_name, []).append(clean_offsets(anno["split"], rtrace["rt"]))
            elif table_name in targets:
                model_anno_data.setdefault(table_name, []).append(clean_offsets(anno["split"], rtrace["rt"]))
            else:
                continue

    return model_anno_data, human_anno_data, traces


def score(mods: List[str], aligner) -> float:
    global BASE_DATA, BASE_DATA_HUMAN, TRACES

    segs = [BASE_DATA[m] for m in mods]
    if len(segs) == 0:
        return 0.0
    elif len(segs) == 1:
        fused = segs[0]
    else:
        if aligner is None:
            return 0.0
        fused = []
        for eles in zip(*segs):
            fused.append(aligner.fuse(list(eles)))

    current_data = copy.deepcopy(BASE_DATA_HUMAN)
    current_data["model"] = fused
    target_data = []
    for idx in range(len(current_data["model"])):
        target_data.append({k: v[idx] for (k, v) in current_data.items()})

    return float(evaluate_approaches_bounding_similarity(TRACES, target_data, window=3))


def get_name(mods: List[str], aligner) -> str:
    return f"{'_'.join(mods)}_{aligner.__name__}"


# ----------------------- CONFIG ENCODING -----------------------
N_MODELS = len(ALL_MODELS)
N_ALIGNERS = len(ALIGNERS)


@dataclass(frozen=True)
class ConfigKey:
    mask: int
    aligner_idx: int

    def as_tuple(self) -> Tuple[int, int]:
        return (self.mask, self.aligner_idx)


def decode_config(cfg: ConfigKey) -> Tuple[List[str], Any]:
    mods = [ALL_MODELS[i] for i in range(N_MODELS) if (cfg.mask >> i) & 1]
    aligner = ALIGNERS[cfg.aligner_idx]
    return mods, aligner


# ----------------------- WORKER INIT + EVAL -----------------------
def _worker_init():
    global BASE_DATA, BASE_DATA_HUMAN, TRACES
    if BASE_DATA is None:
        # Load all targets once per worker
        BASE_DATA, BASE_DATA_HUMAN, TRACES = load_data(tuple(ALL_MODELS))


def _eval_one(cfg_tup: Tuple[int, int]) -> Tuple[int, int, str, float]:
    mask, aligner_idx = cfg_tup
    cfg = ConfigKey(mask, aligner_idx)
    mods, aligner = decode_config(cfg)
    s = score(mods, aligner)
    approach_id = get_name(mods, aligner)
    return (mask, aligner_idx, approach_id, float(s))


# ----------------------- SQLITE STORAGE -----------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS results (
  mask        INTEGER NOT NULL,
  aligner_idx INTEGER NOT NULL,
  approach_id TEXT    NOT NULL,
  score       REAL    NOT NULL,
  ts_unix     INTEGER NOT NULL,
  PRIMARY KEY (mask, aligner_idx)
);
CREATE INDEX IF NOT EXISTS idx_score ON results(score DESC);
CREATE INDEX IF NOT EXISTS idx_ts ON results(ts_unix);
"""

PRAGMAS = [
    ("journal_mode", "WAL"),
    ("synchronous", "NORMAL"),
    ("temp_store", "MEMORY"),
    ("cache_size", "-200000"),
]


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=60)
    conn.execute("PRAGMA busy_timeout=60000;")
    for k, v in PRAGMAS:
        conn.execute(f"PRAGMA {k}={v};")
    for stmt in SCHEMA_SQL.strip().split(";"):
        s = stmt.strip()
        if s:
            conn.execute(s + ";")
    conn.commit()
    return conn


def insert_batch(conn: sqlite3.Connection, rows: List[Tuple[int, int, str, float, int]]) -> int:
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO results(mask, aligner_idx, approach_id, score, ts_unix) VALUES(?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return conn.total_changes - before


def get_unique_count(conn: sqlite3.Connection) -> int:
    (n,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
    return int(n)


# ----------------------- EVOLUTIONARY SEARCH -----------------------
def random_mask(rng: np.random.Generator, p_on: float = 0.25) -> int:
    while True:
        bits = rng.random(N_MODELS) < p_on
        if bits.any():
            mask = 0
            for i, b in enumerate(bits):
                if b:
                    mask |= (1 << i)
            return mask


def mutate(cfg: ConfigKey, rng: np.random.Generator,
           bit_mut_rate: float = 1.0 / max(1, N_MODELS),
           flip_aligner_p: float = 0.05) -> ConfigKey:
    mask = cfg.mask
    for i in range(N_MODELS):
        if rng.random() < bit_mut_rate:
            mask ^= (1 << i)
    if mask == 0:
        mask = 1 << int(rng.integers(0, N_MODELS))

    aligner_idx = cfg.aligner_idx
    if rng.random() < flip_aligner_p:
        aligner_idx = int(rng.integers(0, N_ALIGNERS))

    return ConfigKey(mask, aligner_idx)


def crossover(a: ConfigKey, b: ConfigKey, rng: np.random.Generator) -> ConfigKey:
    mix = 0
    for i in range(N_MODELS):
        bit = ((a.mask >> i) & 1) if (rng.random() < 0.5) else ((b.mask >> i) & 1)
        if bit:
            mix |= (1 << i)
    if mix == 0:
        mix = a.mask if a.mask != 0 else (b.mask or (1 << int(rng.integers(0, N_MODELS))))

    aligner_idx = a.aligner_idx if rng.random() < 0.5 else b.aligner_idx
    return ConfigKey(mix, aligner_idx)


def tournament_select(pop: List[ConfigKey], scores: np.ndarray, rng: np.random.Generator, k: int = 7) -> ConfigKey:
    idxs = rng.integers(0, len(pop), size=k)
    best = idxs[0]
    for j in idxs[1:]:
        if scores[j] > scores[best]:
            best = j
    return pop[best]


def fetch_scores_for_keys(conn: sqlite3.Connection, keys: List[Tuple[int, int]]) -> dict:
    """
    Fast batched lookup: keys are (mask, aligner_idx)
    """
    if not keys:
        return {}

    score_map = {}
    chunk_size = 2000
    for i in range(0, len(keys), chunk_size):
        chunk = keys[i:i + chunk_size]
        values = ",".join(["(?,?)"] * len(chunk))
        flat = []
        for (m, a) in chunk:
            flat.extend([m, a])
        q = (
            "WITH wanted(mask, aligner_idx) AS (VALUES " + values + ") "
            "SELECT r.mask, r.aligner_idx, r.score "
            "FROM results r "
            "JOIN wanted w USING(mask, aligner_idx)"
        )
        for m, a, s in conn.execute(q, flat).fetchall():
            score_map[(m, a)] = float(s)
    return score_map


def run_search(
    db_path: str = "evo_results.sqlite",
    target_unique: int = 300_000_000,
    population_size: int = 4096,
    offspring_per_gen: int = 16384,
    max_workers: Optional[int] = None,
    submit_batch: int = 5000,
    seed: int = 1337,
):
    max_workers = max_workers or (os.cpu_count() or 4)
    rng = np.random.default_rng(seed)

    conn = open_db(db_path)
    existing = get_unique_count(conn)
    print(f"[DB] {existing:,} unique configs already stored at {db_path}")

    pop = [
        ConfigKey(
            mask=random_mask(rng, p_on=0.25),
            aligner_idx=int(rng.integers(0, N_ALIGNERS)),
        )
        for _ in range(population_size)
    ]

    gen = 0
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init) as ex:
        while True:
            current_unique = get_unique_count(conn)
            if current_unique >= target_unique:
                print(f"[DONE] reached target {target_unique:,} unique configs")
                break

            gen += 1

            # --- build candidates: current pop + offspring ---
            candidates: List[ConfigKey] = []
            candidates.extend(pop)

            while len(candidates) < (population_size + offspring_per_gen):
                pa = pop[int(rng.integers(0, len(pop)))]
                pb = pop[int(rng.integers(0, len(pop)))]
                child = mutate(crossover(pa, pb, rng), rng)
                candidates.append(child)

            # de-dup within epoch
            seen = set()
            uniq_candidates: List[ConfigKey] = []
            for c in candidates:
                t = c.as_tuple()
                if t not in seen:
                    seen.add(t)
                    uniq_candidates.append(c)

            # --- evaluate + store with per-epoch tqdm ---
            ts = int(time.time())
            pending_rows: List[Tuple[int, int, str, float, int]] = []
            inserted_total = 0
            evaluated = 0
            total_this_gen = len(uniq_candidates)

            with tqdm(total=total_this_gen, desc=f"GEN {gen}", unit="cfg") as pbar:
                for i in range(0, total_this_gen, submit_batch):
                    batch = uniq_candidates[i:i + submit_batch]
                    futures = [ex.submit(_eval_one, c.as_tuple()) for c in batch]

                    for fut in as_completed(futures):
                        mask, aligner_idx, approach_id, s = fut.result()
                        pending_rows.append((mask, aligner_idx, approach_id, s, ts))

                        evaluated += 1
                        pbar.update(1)

                        if len(pending_rows) >= 5000:
                            inserted_total += insert_batch(conn, pending_rows)
                            pending_rows.clear()
                            pbar.set_postfix(inserted=inserted_total)

                    if pending_rows:
                        inserted_total += insert_batch(conn, pending_rows)
                        pending_rows.clear()
                        pbar.set_postfix(inserted=inserted_total)

            # --- selection for next epoch ---
            # get scores for uniq_candidates from DB (covers duplicates from prior runs too)
            key_list = [c.as_tuple() for c in uniq_candidates]
            score_map = fetch_scores_for_keys(conn, key_list)
            cand_scores = np.array([score_map.get(k, -1e30) for k in key_list], dtype=np.float64)

            # elitism + tournament fill
            E = max(1, population_size // 16)
            elite_idx = np.argpartition(-cand_scores, E - 1)[:E]
            elites = [uniq_candidates[i] for i in elite_idx]

            new_pop = elites[:]
            while len(new_pop) < population_size:
                parent = tournament_select(uniq_candidates, cand_scores, rng, k=7)
                child = mutate(parent, rng, bit_mut_rate=1.0 / max(1, N_MODELS), flip_aligner_p=0.08)
                new_pop.append(child)
            pop = new_pop

            # reporting
            current_unique2 = get_unique_count(conn)
            best_score = float(np.max(cand_scores)) if len(cand_scores) else float("nan")
            mean_score = float(np.mean(cand_scores)) if len(cand_scores) else float("nan")
            print(
                f"[GEN {gen:05d}] evaluated={evaluated:,} inserted_new={inserted_total:,} "
                f"db_unique={current_unique2:,}/{target_unique:,} best={best_score:.6f} mean={mean_score:.6f}"
            )


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    run_search(
        db_path=f"{bp()}/data/local_db/evo_results.sqlite",
        target_unique=300_000_000,
        population_size=4096,
        offspring_per_gen=16384,
        max_workers=os.cpu_count() or 8,
        submit_batch=5000,
        seed=1337,
    )
