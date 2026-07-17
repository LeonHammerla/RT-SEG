from unittest.mock import patch

import pytest

from rt_seg import RTPlainSegmenter as PublicRTPlainSegmenter
from rt_segmentation import RTPlainSegmenter, RTSeg
from rt_segmentation.base_segmentor import UnitSegmentor


def test_plain_segmenter_is_exported_from_public_package() -> None:
    assert PublicRTPlainSegmenter is RTPlainSegmenter


@pytest.mark.parametrize(
    ("base_unit", "parser_method", "expected_offsets"),
    [
        ("sent", "get_math_aware_sents", [(0, 16), (16, 32)]),
        ("clause", "get_math_aware_clauses", [(0, 6), (6, 16), (16, 32)]),
    ],
)
def test_plain_segmenter_returns_selected_base_offsets_unchanged(
    base_unit,
    parser_method,
    expected_offsets,
) -> None:
    trace = "First sentence. Second sentence."

    with patch.object(
        UnitSegmentor,
        parser_method,
        return_value=expected_offsets,
    ) as mocked_parser:
        segmenter = RTSeg(
            engines=[RTPlainSegmenter],
            aligner=None,
            label_fusion_type="concat",
            seg_base_unit=base_unit,
        )
        offsets, labels = segmenter(trace=trace)

    mocked_parser.assert_called_once_with(trace)
    assert offsets == expected_offsets
    assert labels == ["UNK"] * len(expected_offsets)
    assert segmenter.exp_id == f"RTPlainSegmenter_{base_unit}"


@pytest.mark.parametrize("base_unit", ["sent", "clause"])
def test_plain_segmenter_handles_no_base_offsets(base_unit) -> None:
    with patch.object(
        UnitSegmentor,
        "get_math_aware_sents" if base_unit == "sent" else "get_math_aware_clauses",
        return_value=[],
    ):
        offsets, labels = RTPlainSegmenter._segment(
            trace="",
            seg_base_unit=base_unit,
        )

    assert offsets == []
    assert labels == []


def test_plain_segmenter_rejects_unknown_base_unit() -> None:
    with pytest.raises(ValueError, match="Invalid seg_base_unit"):
        RTPlainSegmenter._segment(
            trace="A trace.",
            seg_base_unit="paragraph",
        )
