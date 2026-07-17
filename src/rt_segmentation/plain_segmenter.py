from typing import Literal

from .seg_base import SegBase


class RTPlainSegmenter(SegBase):
    """Return the selected base-unit segmentation without further grouping."""

    @staticmethod
    def _segment(
        trace: str,
        seg_base_unit: Literal["sent", "clause"],
        **kwargs,
    ) -> tuple[list[tuple[int, int]], list[str]]:
        offsets = SegBase.get_base_offsets(
            trace,
            seg_base_unit=seg_base_unit,
        )
        return offsets, ["UNK" for _ in offsets]
