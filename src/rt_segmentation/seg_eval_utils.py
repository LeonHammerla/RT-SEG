import json
from functools import lru_cache
from typing import List, Tuple, Dict, Optional
from nltk.metrics.agreement import AnnotationTask
import numpy as np
import pandas as pd
from itertools import combinations


class TraceSegmentEvaluator:
    @staticmethod
    def _to_labels(trace_len: int, offsets: List[Tuple[int, int]]) -> np.ndarray:
        """Converts offsets to a sequence of segment IDs for character-level metrics."""
        labels = np.zeros(trace_len, dtype=int)
        for i, (start, end) in enumerate(offsets):
            labels[start:end] = i + 1
        return labels

    @staticmethod
    def calculate_pk(ref_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """Beeferman's P_k: Measures probability of disagreement between two points."""
        n = len(ref_labels)
        # Window size: half the average segment length
        avg_len = n / (np.max(ref_labels) if np.max(ref_labels) > 0 else 1)
        k = max(1, int(round(avg_len / 2)))

        err = 0
        for i in range(n - k):
            if (ref_labels[i] == ref_labels[i + k]) != (pred_labels[i] == pred_labels[i + k]):
                err += 1
        return err / (n - k)

    @staticmethod
    def calculate_windowdiff(ref_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """WindowDiff: Better than P_k at penalizing near-misses and false boundaries."""
        n = len(ref_labels)
        avg_len = n / (np.max(ref_labels) if np.max(ref_labels) > 0 else 1)
        k = max(1, int(round(avg_len / 2)))

        # Convert to boundary counts
        ref_b = (ref_labels[:-1] != ref_labels[1:]).astype(int)
        pred_b = (pred_labels[:-1] != pred_labels[1:]).astype(int)

        diff = 0
        for i in range(len(ref_b) - k + 1):
            if np.sum(ref_b[i:i + k]) != np.sum(pred_b[i:i + k]):
                diff += 1
        return diff / (len(ref_b) - k + 1)

    @staticmethod
    def calculate_boundary_f1(gold: List[Tuple[int, int]], pred: List[Tuple[int, int]]):
        """Precision/Recall of actual boundary indices."""
        gold_ends = set(end for _, end in gold[:-1])
        pred_ends = set(end for _, end in pred[:-1])

        tp = len(gold_ends.intersection(pred_ends))
        precision = tp / len(pred_ends) if pred_ends else 0.0
        recall = tp / len(gold_ends) if gold_ends else 1.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    def evaluate(self, trace: str, segmentations: Dict[str, List[Tuple[int, int]]], gold_key: str = None):
        trace_len = len(trace)
        results = []
        labels_map = {name: self._to_labels(trace_len, off) for name, off in segmentations.items()}

        for name, off in segmentations.items():
            # 1. Intrinsic Metrics (No Gold Standard needed)
            lengths = [e - s for s, e in off]
            res = {
                "Approach": name,
                "Seg_Count": len(off),
                "Avg_Len": np.mean(lengths),
                "Len_Std": np.std(lengths),
                "Min_Len": np.min(lengths)
            }

            # 2. Gold-Standard Metrics
            if gold_key and name != gold_key:
                p, r, f1 = self.calculate_boundary_f1(segmentations[gold_key], off)
                res.update({
                    "Precision": round(p, 3),
                    "Recall": round(r, 3),
                    "F1": round(f1, 3),
                    "P_k": round(self.calculate_pk(labels_map[gold_key], labels_map[name]), 3),
                    "WindowDiff": round(self.calculate_windowdiff(labels_map[gold_key], labels_map[name]), 3)
                })
            results.append(res)

        return pd.DataFrame(results).sort_values("F1", ascending=False if gold_key else True)


class SegmentationAgreement:
    @staticmethod
    def boundary_similarity(seg1: List[Tuple[int, int]], seg2: List[Tuple[int, int]], trace_len: int):
        """
        Calculates Boundary Similarity (B).
        1.0 is perfect agreement, 0.0 is total disagreement.
        It is less strict than F1 because it uses a distance-weighted approach.
        """
        # Create boundary bitstrings
        b1 = np.zeros(trace_len)
        b2 = np.zeros(trace_len)
        for _, end in seg1[:-1]: b1[end] = 1
        for _, end in seg2[:-1]: b2[end] = 1

        # Calculate matching boundaries within a small epsilon (e.g., 2 characters)
        # For simplicity in this script, we use a basic windowed overlap
        matches = 0
        epsilon = 3  # Allowing a 3-character "jitter"

        b1_indices = np.where(b1 == 1)[0]
        b2_indices = np.where(b2 == 1)[0]

        if len(b1_indices) == 0 or len(b2_indices) == 0:
            return 1.0 if len(b1_indices) == len(b2_indices) else 0.0

        for idx in b1_indices:
            if any(abs(idx - j) <= epsilon for j in b2_indices):
                matches += 1

        precision = matches / len(b2_indices)
        recall = matches / len(b1_indices)

        return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    @staticmethod
    def krippendorff_alpha_binary(segmentations_list: List[np.ndarray]):
        """
        Simplified Alpha for binary boundary arrays.
        Requires the 'nltk' library or a custom implementation of the Alpha formula.
        """
        # In practice, for boundary detection, we treat each character as an item
        # and the presence/absence of a boundary as the category.

        task_data = []
        for coder_id, labels in enumerate(segmentations_list):
            for item_id, val in enumerate(labels):
                task_data.append((str(coder_id), str(item_id), val))

        task = AnnotationTask(data=task_data)
        return task.alpha()


class ReasoningAgreementSuite:
    def __init__(self, window_size: int = 3):
        self.window = window_size  # Jitter tolerance

    def _get_boundary_indices(self, offsets: List[Tuple[int, int]]) -> set:
        """Extracts the end-points of segments as the boundaries."""
        return {end for _, end in offsets[:-1]}

    def boundary_similarity(self, seg1: List[Tuple[int, int]], seg2: List[Tuple[int, int]], trace_len: int) -> float:
        """
        Calculates Boundary Similarity (B).
        A lenient metric that rewards 'near hits' within the window size.
        """
        b1 = self._get_boundary_indices(seg1)
        b2 = self._get_boundary_indices(seg2)

        if not b1 and not b2: return 1.0
        if not b1 or not b2: return 0.0

        def count_matches(set_a, set_b):
            matches = 0
            for val in set_a:
                # Check if any boundary in B is within the 'jitter' window of A
                if any(abs(val - other) <= self.window for other in set_b):
                    matches += 1
            return matches

        m1 = count_matches(b1, b2)
        m2 = count_matches(b2, b1)

        # Harmonic mean of the directional matches (Precision/Recall style)
        p = m1 / len(b1)
        r = m2 / len(b2)
        return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

    def calculate_alpha_matrix(self, segmentations: Dict[str, List[Tuple[int, int]]], trace_len: int):
        """
        Computes a pairwise agreement matrix using Boundary Similarity.
        This represents a 'Boundary-Based' consensus.
        """
        names = list(segmentations.keys())
        n = len(names)
        matrix = np.ones((n, n))

        for i, j in combinations(range(n), 2):
            sim = self.boundary_similarity(segmentations[names[i]],
                                           segmentations[names[j]],
                                           trace_len)
            matrix[i, j] = matrix[j, i] = sim

        return pd.DataFrame(matrix, index=names, columns=names)

    def group_consensus_score(self, segmentations: Dict[str, List[Tuple[int, int]]], trace_len: int) -> float:
        """
        Returns the global agreement across all provided approaches.
        Similar to a 'Fleiss Kappa' but for boundaries.
        """
        matrix = self.calculate_alpha_matrix(segmentations, trace_len)
        # Average of the upper triangle (excluding diagonal)
        upper_tri = matrix.values[np.triu_indices(len(matrix), k=1)]
        return np.mean(upper_tri) if upper_tri.size > 0 else 1.0


class OptimisticSegmentEvaluator:
    def __init__(self, slack=10):
        self.slack = slack  # Number of characters to allow for "near-misses"

    def calculate_boundary_cover(self, seg_a, seg_b):
        """
        Measures how well the boundaries of A are 'covered' by B.
        Optimistic because it doesn't penalize 'extra' steps in B.
        """
        b_a = [end for _, end in seg_a[:-1]]
        b_b = [end for _, end in seg_b[:-1]]

        if not b_a: return 1.0

        covered = 0
        for boundary in b_a:
            # If any boundary in B is 'near' the boundary in A, it's a hit
            if any(abs(boundary - other) <= self.slack for other in b_b):
                covered += 1

        return covered / len(b_a)

    def fuzzy_agreement(self, seg_dict: Dict[str, List[Tuple[int, int]]]):
        """
        Compares multiple segmentations by finding the 'Best Fit' overlap.
        """
        names = list(seg_dict.keys())
        results = []

        for a, b in combinations(names, 2):
            # We calculate cover in both directions
            cover_a_in_b = self.calculate_boundary_cover(seg_dict[a], seg_dict[b])
            cover_b_in_a = self.calculate_boundary_cover(seg_dict[b], seg_dict[a])

            # The 'Optimistic' score is the maximum of the two (Directional Cover)
            # or the average for a more balanced view.
            results.append({
                "Pair": f"{a} vs {b}",
                "Optimistic_Score": max(cover_a_in_b, cover_b_in_a),
                "Symmetry_Score": (cover_a_in_b + cover_b_in_a) / 2
            })
        return pd.DataFrame(results)


class AdvancedMetricsSuite:
    """
    Complements the existing evaluation suite with:
    - Distance-weighted soft boundary F1
    - Segment overlap metrics (IoU / Dice)
    - Boundary displacement statistics
    - Over/under-segmentation bias
    - Boundary density divergence (gold-free)
    """

    # ---------- helpers ----------

    @staticmethod
    def _boundaries(offsets: List[Tuple[int, int]]) -> np.ndarray:
        return np.array([end for _, end in offsets[:-1]])

    @staticmethod
    def _segments_to_mask(offsets: List[Tuple[int, int]], trace_len: int) -> np.ndarray:
        mask = np.zeros(trace_len, dtype=int)
        for i, (s, e) in enumerate(offsets):
            mask[s:e] = i + 1
        return mask

    # ---------- 1. soft boundary F1 (distance-weighted) ----------

    @staticmethod
    def soft_boundary_f1(
        gold: List[Tuple[int, int]],
        pred: List[Tuple[int, int]],
        sigma: float = 5.0,
    ):
        """
        Distance-weighted soft boundary precision / recall / F1.
        Uses exponential decay based on distance.
        """
        g = AdvancedMetricsSuite._boundaries(gold)
        p = AdvancedMetricsSuite._boundaries(pred)

        if len(g) == 0 and len(p) == 0:
            return 1.0, 1.0, 1.0
        if len(g) == 0 or len(p) == 0:
            return 0.0, 0.0, 0.0

        def score(a, b):
            return np.mean([np.exp(-np.min(np.abs(b - x)) / sigma) for x in a])

        precision = score(p, g)
        recall = score(g, p)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

        return precision, recall, f1

    # ---------- 2. segment overlap (IoU / Dice) ----------

    @staticmethod
    def segment_iou_dice(
        gold: List[Tuple[int, int]],
        pred: List[Tuple[int, int]],
        trace_len: int,
    ):
        """
        Computes mean IoU and Dice over best-matching segments.
        """
        g_mask = AdvancedMetricsSuite._segments_to_mask(gold, trace_len)
        p_mask = AdvancedMetricsSuite._segments_to_mask(pred, trace_len)

        g_ids = np.unique(g_mask[g_mask > 0])
        p_ids = np.unique(p_mask[p_mask > 0])

        ious, dices = [], []

        for g in g_ids:
            g_region = g_mask == g
            best_iou, best_dice = 0.0, 0.0

            for p in p_ids:
                p_region = p_mask == p
                inter = np.sum(g_region & p_region)
                union = np.sum(g_region | p_region)

                if union == 0:
                    continue

                iou = inter / union
                dice = (2 * inter) / (np.sum(g_region) + np.sum(p_region))

                best_iou = max(best_iou, iou)
                best_dice = max(best_dice, dice)

            ious.append(best_iou)
            dices.append(best_dice)

        return float(np.mean(ious)), float(np.mean(dices))

    # ---------- 3. boundary displacement ----------

    @staticmethod
    def boundary_displacement(
        gold: List[Tuple[int, int]],
        pred: List[Tuple[int, int]],
    ):
        """
        Mean absolute distance between each gold boundary and nearest predicted one.
        Lower is better.
        """
        g = AdvancedMetricsSuite._boundaries(gold)
        p = AdvancedMetricsSuite._boundaries(pred)

        if len(g) == 0 or len(p) == 0:
            return float("inf")

        return float(np.mean([np.min(np.abs(p - x)) for x in g]))

    # ---------- 4. over / under segmentation bias ----------

    @staticmethod
    def segmentation_bias(
        gold: List[Tuple[int, int]],
        pred: List[Tuple[int, int]],
    ):
        """
        > 0 : over-segmentation
        < 0 : under-segmentation
        """
        return (len(pred) - len(gold)) / max(1, len(gold))

    # ---------- 5. boundary density divergence (gold-free) ----------

    @staticmethod
    def boundary_density_divergence(
        seg1: List[Tuple[int, int]],
        seg2: List[Tuple[int, int]],
        trace_len: int,
        bins: int = 20,
    ):
        """
        Compares how boundaries are distributed across the trace.
        Jensen–Shannon divergence (symmetric, bounded).
        """
        def density(seg):
            b = AdvancedMetricsSuite._boundaries(seg)
            hist, _ = np.histogram(b, bins=bins, range=(0, trace_len), density=True)
            return hist + 1e-9

        p = density(seg1)
        q = density(seg2)
        m = 0.5 * (p + q)

        js = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
        return float(js)

    # ---------- convenience: evaluate pair ----------

    @staticmethod
    def evaluate_pair(
        gold: List[Tuple[int, int]],
        pred: List[Tuple[int, int]],
        trace_len: int,
        sigma: float = 5.0,
    ) -> Dict[str, float]:
        sp, sr, sf1 = AdvancedMetricsSuite.soft_boundary_f1(gold, pred, sigma)
        iou, dice = AdvancedMetricsSuite.segment_iou_dice(gold, pred, trace_len)

        return {
            "Soft_Precision": sp,
            "Soft_Recall": sr,
            "Soft_F1": sf1,
            "Mean_IoU": iou,
            "Mean_Dice": dice,
            "Boundary_Displacement": AdvancedMetricsSuite.boundary_displacement(gold, pred),
            "Segmentation_Bias": AdvancedMetricsSuite.segmentation_bias(gold, pred),
        }


def test_segmentation():
    # --- Example Usage ---
    trace = "Step 1: Get data. Data is [1, 2]. Step 2: Sum data. Sum is 3. Step 3: Square it. Result is 9."

    segment_data = {
        "Ground_Truth": [(0, 31), (31, 59), (59, 84)],
        "Regex_Splitter": [(0, 31), (31, 84)],  # Missed Step 3
        "LLM_Fine": [(0, 16), (16, 31), (31, 46), (46, 59), (59, 84)]  ,# Over-segmented
        "Streber": [(0, 31), (31, 59), (59, 84)],
    }

    evaluator = TraceSegmentEvaluator()
    comparison_df = evaluator.evaluate(trace, segment_data, gold_key="Ground_Truth")
    print(f"{10*'='} TraceSegmentEvaluator {10*'='}")
    print(comparison_df.to_string(index=False))

    print(f"{10 * '='} SegmentationAgreement {10 * '='}")
    for (segn, segd) in segment_data.items():
        if segn != "Ground_Truth":
            print(f"{segn} - Boundary Sim: {SegmentationAgreement.boundary_similarity(segd, segment_data['Ground_Truth'], len(trace))}")
            print(f"{segn} - Kripp A: {SegmentationAgreement.krippendorff_alpha_binary([segd, segment_data['Ground_Truth']])}")

    print(f"{10 * '='} ReasoningAgreementSuite {10 * '='}")

    suite = ReasoningAgreementSuite(window_size=3)
    matrix_df = suite.calculate_alpha_matrix(segment_data, len(trace))
    global_score = suite.group_consensus_score(segment_data, len(trace))

    print("Pairwise Boundary Agreement Matrix:")
    print(matrix_df.round(3))
    print(f"\nGlobal Boundary Consensus Score: {global_score:.3f}")

    print(f"{10 * '='} OptimisticSegmentEvaluator {10 * '='}")

    opt = OptimisticSegmentEvaluator()
    for (segn, segd) in segment_data.items():
        if segn != "Ground_Truth":
            print(f"{segn} - Boundary Cover: {opt.calculate_boundary_cover(segd, segment_data['Ground_Truth'])}")
    print(opt.fuzzy_agreement(segment_data))

    print(f"{10 * '='} AdvancedMetricSuite {10 * '='}")

    missing = AdvancedMetricsSuite()
    trace_len = len(trace)

    for name, seg in segment_data.items():
        if name == "Ground_Truth":
            continue

        metrics = missing.evaluate_pair(
            gold=segment_data["Ground_Truth"],
            pred=seg,
            trace_len=trace_len,
            sigma=5.0,  # tolerance for near-miss boundaries
        )

        print(f"\n{name}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:24s}: {v:.3f}")
            else:
                print(f"  {k:24s}: {v}")

    print(f"\n{10 * '='} Boundary Density Divergence (Gold-free) {10 * '='}")

    for a, b in combinations(segment_data.keys(), 2):
        div = missing.boundary_density_divergence(
            segment_data[a],
            segment_data[b],
            trace_len=trace_len,
        )
        print(f"{a} vs {b}: JSD = {div:.4f}")

@lru_cache(maxsize=None)
def get_eval_registry():
    return {

        # --------------------------------------------------
        # A. Classical segmentation metrics (strict, legacy)
        # --------------------------------------------------
        "classical": {
            "requires_gold": True,
            "functions": {
                "P_k": lambda gold_seg, pred_seg, gold_lbl=None, pred_lbl=None, **_: (
                    TraceSegmentEvaluator.calculate_pk(gold_lbl, pred_lbl)
                ),
                "WindowDiff": lambda gold_seg, pred_seg, gold_lbl=None, pred_lbl=None, **_: (
                    TraceSegmentEvaluator.calculate_windowdiff(gold_lbl, pred_lbl)
                ),
            },
        },

        # --------------------------------------------------
        # B. Boundary accuracy (strict vs tolerant)
        # --------------------------------------------------
        "boundary_accuracy": {
            "requires_gold": True,
            "functions": {
                "Boundary_F1": lambda gold_seg, pred_seg, **_: (
                    TraceSegmentEvaluator.calculate_boundary_f1(gold_seg, pred_seg)[2]
                ),
                "Soft_Boundary_F1": lambda gold_seg, pred_seg, sigma=5.0, **_: (
                    AdvancedMetricsSuite.soft_boundary_f1(gold_seg, pred_seg, sigma)[2]
                ),
                "Boundary_Displacement": lambda gold_seg, pred_seg, **_: (
                    AdvancedMetricsSuite.boundary_displacement(gold_seg, pred_seg)
                ),
            },
        },

        # --------------------------------------------------
        # C. Segment-level structural similarity
        # --------------------------------------------------
        "segment_structure": {
            "requires_gold": True,
            "functions": {
                "Mean_IoU": lambda gold_seg, pred_seg, trace_len=None, **_: (
                    AdvancedMetricsSuite.segment_iou_dice(
                        gold_seg, pred_seg, trace_len
                    )[0]
                ),
                "Mean_Dice": lambda gold_seg, pred_seg, trace_len=None, **_: (
                    AdvancedMetricsSuite.segment_iou_dice(
                        gold_seg, pred_seg, trace_len
                    )[1]
                ),
                "Segmentation_Bias": lambda gold_seg, pred_seg, **_: (
                    AdvancedMetricsSuite.segmentation_bias(gold_seg, pred_seg)
                ),
            },
        },

        # --------------------------------------------------
        # D. Agreement-based metrics (gold optional)
        # --------------------------------------------------
        "agreement": {
            "requires_gold": False,
            "pairwise": True,
            "functions": {
                "Boundary_Similarity": lambda seg_a, seg_b, trace_len=None, window=3, **_: (
                    ReasoningAgreementSuite(window).boundary_similarity(
                        seg_a, seg_b, trace_len
                    )
                ),
                "Boundary_Density_JSD": lambda seg_a, seg_b, trace_len=None, **_: (
                    AdvancedMetricsSuite.boundary_density_divergence(
                        seg_a, seg_b, trace_len
                    )
                ),
            },
        },

        # --------------------------------------------------
        # E. Optimistic / lenient diagnostics
        # --------------------------------------------------
        "optimistic": {
            "requires_gold": True,
            "functions": {
                "Boundary_Cover": lambda gold_seg, pred_seg, slack=10, **_: (
                    OptimisticSegmentEvaluator(slack)
                    .calculate_boundary_cover(pred_seg, gold_seg)
                ),
            },
        },
    }


def evaluate_segmentations(
    trace: str,
    segmentations: dict,
    gold_key: str = None,
    groups: list = None,
    **kwargs,
):
    """
    Unified evaluation entry point.

    Parameters
    ----------
    trace : str
        The original trace text
    segmentations : Dict[str, List[(int, int)]]
        Named segmentation outputs
    gold_key : str
        Key of the gold segmentation (optional)
    groups : list
        Which metric groups to compute (default: all)
    kwargs :
        Hyperparameters (sigma, window, slack, etc.)

    Returns
    -------
    Dict[str, pd.DataFrame]
        One DataFrame per metric group
    """

    trace_len = len(trace)
    labels = {
        k: TraceSegmentEvaluator._to_labels(trace_len, v)
        for k, v in segmentations.items()
    }

    results = {}
    groups = groups or list(get_eval_registry().keys())

    for group in groups:
        spec = get_eval_registry()[group]
        rows = []

        # ---------- pairwise agreement ----------
        if spec.get("pairwise", False):
            for a, b in combinations(segmentations.keys(), 2):
                row = {"A": a, "B": b}
                for name, fn in spec["functions"].items():
                    row[name] = fn(
                        segmentations[a],
                        segmentations[b],
                        trace_len=trace_len,
                        **kwargs,
                    )
                rows.append(row)

        # ---------- gold-based metrics ----------
        else:
            for name, seg in segmentations.items():
                if spec["requires_gold"] and name == gold_key:
                    continue

                row = {"Method": name}

                for metric, fn in spec["functions"].items():
                    try:
                        if spec["requires_gold"]:
                            row[metric] = fn(
                                segmentations[gold_key],
                                seg,
                                gold_lbl=labels[gold_key],
                                pred_lbl=labels[name],
                                trace_len=trace_len,
                                **kwargs,
                            )
                        else:
                            row[metric] = fn(
                                seg,
                                trace_len=trace_len,
                                **kwargs,
                            )
                    except Exception:
                        row[metric] = np.nan

                rows.append(row)

        results[group] = pd.DataFrame(rows)

    return results


def evaluate_aggregate_segmentations(traces: list,
                                     segmentations: list,
                                     gold_key: str = None,
                                     groups: list = None,
                                     **kwargs):
    all_results = []

    # Collect per-trace evaluations
    for trace_id, trace in enumerate(traces):
        res = evaluate_segmentations(trace=trace,
                                     segmentations=segmentations[trace_id],
                                     gold_key=gold_key,
                                     groups=groups,
                                     **kwargs)
        for group, df in res.items():
            df = df.copy()
            df['TraceID'] = trace_id
            df['Group'] = group
            all_results.append(df)

    full_df = pd.concat(all_results, ignore_index=True)

    # Separate linear vs pairwise/agreement metrics
    linear_metrics = ['F1', 'Soft_Boundary_F1', 'Mean_IoU', 'Mean_Dice',
                      'Boundary_Displacement', 'Segmentation_Bias', 'P_k', 'WindowDiff']
    agreement_metrics = ['Boundary_Similarity', 'Boundary_Density_JSD', 'Boundary_Cover']

    # Only keep columns that exist
    linear_cols = [c for c in linear_metrics if c in full_df.columns]
    agreement_cols = [c for c in agreement_metrics if c in full_df.columns]

    # Aggregate linear metrics by Method
    if linear_cols:
        linear_df = full_df[full_df['Group'].isin(['classical', 'boundary_accuracy', 'segment_structure', 'optimistic'])]
        agg_linear = linear_df.groupby('Method')[linear_cols].agg(['mean', 'std'])
    else:
        agg_linear = pd.DataFrame()

    # Pairwise agreement metrics (have A/B)
    pairwise_cols = ['Boundary_Similarity', 'Boundary_Density_JSD']
    if any(c in full_df.columns for c in pairwise_cols):
        pairwise_df = full_df[(full_df['Group'] == 'agreement') & full_df[pairwise_cols].notna().any(axis=1)]
        agg_pairwise = pairwise_df.groupby(['A', 'B'])[pairwise_cols].mean()
    else:
        agg_pairwise = pd.DataFrame()

    # Per-method "agreement" metrics (have Method, not A/B)
    per_method_agreement_cols = ['Boundary_Cover']
    if any(c in full_df.columns for c in per_method_agreement_cols):
        per_method_df = full_df[
            (full_df['Group'] == 'optimistic') & full_df[per_method_agreement_cols].notna().any(axis=1)]
        agg_per_method_agreement = per_method_df.groupby('Method')[per_method_agreement_cols].mean()
    else:
        agg_per_method_agreement = pd.DataFrame()

    return {
        'linear_metrics': agg_linear,
        'pairwise_agreement_metrics': agg_pairwise,
        'per_method_agreement_metrics': agg_per_method_agreement
    }


def aggregated_results_to_json(agg_results: dict) -> dict:
    """
    Convert the output of evaluate_aggregate_segmentations to a clean JSON structure.

    Parameters
    ----------
    agg_results : dict
        Output of evaluate_aggregate_segmentations, e.g.
        {
            'linear_metrics': pd.DataFrame,
            'pairwise_agreement_metrics': pd.DataFrame,
            'per_method_agreement_metrics': pd.DataFrame
        }

    Returns
    -------
    dict
        JSON-serializable dict of aggregated metrics.
    """

    json_dict = {}

    # Linear metrics: convert multi-index DataFrame to nested dict
    if not agg_results['linear_metrics'].empty:
        linear_json = {}
        for method, row in agg_results['linear_metrics'].iterrows():
            linear_json[method] = {metric: {"mean": float(row[(metric, 'mean')]),
                                            "std": float(row[(metric, 'std')])}
                                   for metric in agg_results['linear_metrics'].columns.levels[0]}
        json_dict['linear_metrics'] = linear_json

    # Pairwise agreement metrics
    if not agg_results['pairwise_agreement_metrics'].empty:
        pairwise_json = {}
        for (a, b), row in agg_results['pairwise_agreement_metrics'].iterrows():
            pairwise_json[f"{a} vs {b}"] = {metric: float(row[metric]) for metric in agg_results['pairwise_agreement_metrics'].columns}
        json_dict['pairwise_agreement_metrics'] = pairwise_json

    # Per-method agreement metrics (like Boundary_Cover)
    if not agg_results['per_method_agreement_metrics'].empty:
        per_method_json = {}
        for method, row in agg_results['per_method_agreement_metrics'].iterrows():
            per_method_json[method] = {metric: float(row[metric]) for metric in agg_results['per_method_agreement_metrics'].columns}
        json_dict['per_method_agreement_metrics'] = per_method_json

    return json_dict


if __name__ == "__main__":
    trace = "Step 1: Get data. Data is [1, 2]. Step 2: Sum data. Sum is 3. Step 3: Square it. Result is 9."

    segment_data = {
        "Ground_Truth": [(0, 31), (31, 59), (59, 84)],
        "Regex_Splitter": [(0, 31), (31, 84)],  # Missed Step 3
        "LLM_Fine": [(0, 16), (16, 31), (31, 46), (46, 59), (59, 84)],  # Over-segmented
        "Streber": [(0, 31), (31, 59), (59, 84)],
    }

    # Run single-trace evaluation
    tables = evaluate_segmentations(
        trace=trace,
        segmentations=segment_data,
        gold_key="Ground_Truth",
        sigma=5.0,
        window=3,
        slack=10,
    )

    print("=== Single Trace Evaluation ===")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        for name, df in tables.items():
            print(f"\n--- {name} ---")
            print(df)

    # Aggregate across multiple traces (here using the same trace 3 times)
    agg_tables = evaluate_aggregate_segmentations(
        traces=[trace, trace, trace],
        segmentations=[segment_data, segment_data, segment_data],
        gold_key="Ground_Truth",
        sigma=5.0,
        window=3,
        slack=10,
    )

    print("\n=== Aggregated Linear Metrics ===")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(agg_tables['linear_metrics'])

    print("\n=== Aggregated Agreement Metrics ===")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(agg_tables['pairwise_agreement_metrics'])

    print("\n=== Aggregated Agreement Metrics ===")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(agg_tables['per_method_agreement_metrics'])

    print(json.dumps(aggregated_results_to_json(agg_tables)))