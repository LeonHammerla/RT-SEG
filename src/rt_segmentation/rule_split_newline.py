import re

from .seg_base import SegBase


class RTNewLine(SegBase):
    @staticmethod
    def _segment(trace: str, **kwargs) -> tuple[list[tuple[int, int]], list[str]]:
        # Find positions after each \n\n or at start
        positions = [m.end() for m in re.finditer(r'\n\n|\A', trace)]
        # Pair consecutive positions
        offsets = list(zip(positions, positions[1:] + [len(trace)]))
        return offsets, ["UNK" for _ in offsets]
