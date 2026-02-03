from collections import Counter
from itertools import chain
from typing import Literal, List, Tuple, Any
from abc import ABC, abstractmethod


class OffsetFusion(ABC):
    @staticmethod
    @abstractmethod
    def fuse(segmentations: List[List[Tuple[int, int]]],
             **kwargs) -> List[Tuple[int, int]]:
        pass

class OffsetFusionFlatten(OffsetFusion):
    @staticmethod
    def fuse(segmentations: List[List[Tuple[int, int]]],
             **kwargs):
        """Extract all boundary positions from segmentations"""
        boundaries = set()
        for seg in segmentations:
            for start, end in seg:
                boundaries.add(start)
                boundaries.add(end)
        return sorted(boundaries)

class OffsetFusionVoting(OffsetFusion):
    @staticmethod
    def fuse(segmentations: List[List[Tuple[int, int]]],
             threshold: Any = None,
             **kwargs):
        """
        Return boundaries voted by majority.
        threshold: number of tools required to include boundary (default = majority)
        """
        n_tools = len(segmentations)
        if threshold is None:
            threshold = n_tools // 2 + 1  # majority

        # Count votes for each boundary
        all_boundaries = list(chain.from_iterable([list(map(lambda x: x[0], seg)) + [seg[-1][1]] for seg in segmentations]))
        counts = Counter(all_boundaries)

        # Keep boundaries meeting threshold
        fused_boundaries = sorted([b for b, c in counts.items() if c >= threshold])

        # Convert back to segments
        fused_segments = [(fused_boundaries[i], fused_boundaries[i + 1]) for i in range(len(fused_boundaries) - 1)]
        return fused_segments

class OffsetFusionMerge(OffsetFusion):
    @staticmethod
    def fuse(segmentations: List[List[Tuple[int, int]]],
             **kwargs):
        boundaries = sorted(set(OffsetFusionFlatten.fuse(segmentations)))
        return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

class OffsetFusionIntersect(OffsetFusion):
    @staticmethod
    def fuse(segmentations: List[List[Tuple[int, int]]],
             **kwargs):
        sets = [set(OffsetFusionFlatten.fuse([seg])) for seg in segmentations]
        common = sorted(set.intersection(*sets))
        if common[0] != 0:
            common = [0] + common
        if common[-1] != max(OffsetFusionFlatten.fuse(segmentations)):
            common.append(max(OffsetFusionFlatten.fuse(segmentations)))
        return [(common[i], common[i + 1]) for i in range(len(common) - 1)]

class OffsetFusionFuzzy(OffsetFusion):
    @staticmethod
    def fuse(segmentations: List[List[Tuple[int ,int]]],
             max_distance: int = 1,
             **kwargs):
        """
        Merge boundaries that are within max_distance characters
        """
        all_boundaries = sorted(OffsetFusionFlatten.fuse(segmentations))
        fused_boundaries = []
        current = all_boundaries[0]
        for b in all_boundaries[1:]:
            if b - current <= max_distance:
                continue  # merge close boundaries
            fused_boundaries.append(current)
            current = b
        fused_boundaries.append(current)
        # Convert to segments
        return [(fused_boundaries[i], fused_boundaries[i + 1]) for i in range(len(fused_boundaries) - 1)]

class OffsetFusionGraph(OffsetFusion):
    @staticmethod
    def fuse(segmentations: List[List[Tuple[int, int]]],
             **kwargs):
        """
        Treat each proposed segment as an edge with weight = number of votes.
        Return path maximizing total weight.
        """
        all_boundaries = sorted(set(OffsetFusionFlatten.fuse(segmentations)))
        # Build edges with weights
        edge_weights = {}
        for start in all_boundaries:
            for end in all_boundaries:
                if end <= start:
                    continue
                weight = sum(1 for seg in segmentations for s, e in seg if s == start and e == end)
                if weight > 0:
                    edge_weights[(start, end)] = weight
        # Dynamic programming for longest path
        longest = {b: (0, None) for b in all_boundaries}  # pos -> (weight, previous)
        for b in all_boundaries:
            for e in all_boundaries:
                if e <= b or (b, e) not in edge_weights:
                    continue
                new_weight = longest[b][0] + edge_weights[(b, e)]
                if new_weight > longest[e][0]:
                    longest[e] = (new_weight, b)
        # Backtrack from last position
        end = all_boundaries[-1]
        path = []
        while end is not None:
            path.append(end)
            end = longest[end][1]
        path = list(reversed(path))
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


class LabelFusion:
    @staticmethod
    def fuse(all_tool_segments: List[List[Tuple[int, int]]],
                                    all_tool_labels: List[List[str]],
                                    new_segments: List[Tuple[int, int]],
                                    mode: Literal["majority", "concat"] = "majority"):
        """
        all_tool_segments: list of lists of (start,end) tuples, one per tool
        all_tool_labels: list of lists of labels, one per tool
        new_segments: list of (start,end) tuples
        mode: "majority" or "concat"

        Returns: list of labels for new_segments
        """
        aligned_labels = []

        for s_new, e_new in new_segments:
            overlapping_labels = []
            # Iterate over tools
            for tool_segs, tool_labels in zip(all_tool_segments, all_tool_labels):
                for (s_old, e_old), label in zip(tool_segs, tool_labels):
                    overlap = max(0, min(e_new, e_old) - max(s_new, s_old))
                    if overlap > 0:
                        overlapping_labels.append(label)
            if not overlapping_labels:
                fused_label = "UNKNOWN"
            elif mode == "majority":
                fused_label = Counter(overlapping_labels).most_common(1)[0][0]
            elif mode == "concat":
                fused_label = "+".join(sorted(set(overlapping_labels)))
            else:
                raise ValueError("mode must be 'majority' or 'concat'")
            aligned_labels.append(fused_label)

        return aligned_labels


if __name__ == "__main__":
    segmentations = [
        [(0, 5), (5, 10), (10, 20)],
        [(0, 2), (2, 6), (6, 15), (15, 20)],
        [(0, 5), (5, 15), (15, 20)]
    ]
    labels = [["A", "B", "C"], ["a", "B", "C", "D"], ["A", "B", "C"]]

    print("Majority Voting:", OffsetFusionVoting.fuse(segmentations))
    print("Union:", OffsetFusionMerge.fuse(segmentations))
    print("Intersection:", OffsetFusionIntersect.fuse(segmentations))
    print("Fuzzy Merge:", OffsetFusionFuzzy.fuse(segmentations, max_distance=1))
    print("Graph-based:", OffsetFusionGraph.fuse(segmentations))

    print("Labels", LabelFusion.fuse(segmentations, labels, [(0, 5), (5, 6), (6, 10), (10, 15), (15, 20)], "concat"))
