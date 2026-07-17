from types import SimpleNamespace
from unittest.mock import patch

import pytest

from eval_utils.prm800k_utils import _label_resegmented_steps
from rt_segmentation.base_segmentor import UnitSegmentor
from rt_segmentation.rule_split_newline import RTNewLine, RTNewLineVerbose


MATH_HEAVY_TRACE = (
    "First use $32$, then expand "
    "$$a_1 + a_2 + a_3 + a_4 + a_5 = 160.$$ "
    "Second verify `value = 32` before ```python\nanswer = value * 5\n``` "
    "Third we can now conclude."
)


def _assert_valid_partition(trace: str, offsets: list[tuple[int, int]]) -> None:
    if not offsets:
        assert not trace.strip()
        return
    assert offsets[0][0] == 0
    assert offsets[-1][1] == len(trace)
    for index, (start, end) in enumerate(offsets):
        assert start < end
        assert trace[start:end].strip()
        if index:
            assert offsets[index - 1][1] == start


def test_protected_span_mask_is_length_and_whitespace_preserving() -> None:
    masked = UnitSegmentor._mask_protected_spans(MATH_HEAVY_TRACE)

    assert len(masked) == len(MATH_HEAVY_TRACE)
    assert "$32$" not in masked
    assert "a_1 + a_2" not in masked
    assert "answer = value" not in masked
    for original_character, masked_character in zip(
        MATH_HEAVY_TRACE, masked, strict=True
    ):
        if original_character.isspace():
            assert masked_character == original_character


def test_sentence_offsets_remain_source_coordinates_with_multiple_blocks() -> None:
    class FakeNLP:
        def __call__(self, masked_text):
            assert len(masked_text) == len(MATH_HEAVY_TRACE)
            second_start = masked_text.index("Second")
            third_start = masked_text.index("Third")
            math_internal_start = MATH_HEAVY_TRACE.index("a_2")
            code_internal_start = MATH_HEAVY_TRACE.index("answer")
            return SimpleNamespace(
                sents=[
                    SimpleNamespace(start_char=0, end_char=math_internal_start - 1),
                    SimpleNamespace(
                        start_char=math_internal_start, end_char=second_start - 1
                    ),
                    SimpleNamespace(start_char=second_start, end_char=third_start - 1),
                    SimpleNamespace(
                        start_char=code_internal_start, end_char=third_start - 1
                    ),
                    SimpleNamespace(start_char=third_start, end_char=len(masked_text)),
                ]
            )

    with patch.object(UnitSegmentor, "load_spacy_model", return_value=FakeNLP()):
        offsets = UnitSegmentor.get_math_aware_sents(MATH_HEAVY_TRACE)

    _assert_valid_partition(MATH_HEAVY_TRACE, offsets)
    assert [MATH_HEAVY_TRACE[start:end].lstrip().split()[0] for start, end in offsets] == [
        "First",
        "Second",
        "Third",
    ]


def test_constituency_clause_offsets_remain_source_coordinates() -> None:
    class FakeTree:
        children = []
        is_leaf = lambda self: True

    class FakeNLP:
        def __call__(self, masked_text):
            starts = [
                0,
                MATH_HEAVY_TRACE.index("a_2"),
                masked_text.index("Second"),
                MATH_HEAVY_TRACE.index("answer"),
                masked_text.index("Third"),
            ]
            ends = starts[1:] + [len(masked_text)]
            sentences = []
            for start, end in zip(starts, ends, strict=True):
                word = SimpleNamespace(start_char=start, end_char=end)
                sentences.append(
                    SimpleNamespace(words=[word], constituency=FakeTree())
                )
            return SimpleNamespace(sentences=sentences)

    with patch.object(
        UnitSegmentor, "load_stanza_constituency", return_value=FakeNLP()
    ):
        offsets = UnitSegmentor.get_math_aware_clauses(MATH_HEAVY_TRACE)

    _assert_valid_partition(MATH_HEAVY_TRACE, offsets)
    assert [MATH_HEAVY_TRACE[start:end].lstrip().split()[0] for start, end in offsets] == [
        "First",
        "Second",
        "Third",
    ]


def test_dependency_clause_offsets_remain_source_coordinates() -> None:
    class FakeSentence:
        def __init__(self, start, end):
            self.start_char = start
            self.end_char = end

        def __iter__(self):
            return iter(())

    class FakeNLP:
        pipe_names = []

        def add_pipe(self, factory_name, *, before):
            assert factory_name == "sentencizer"
            assert before == "parser"
            self.pipe_names.append(factory_name)

        def __call__(self, masked_text):
            boundary = masked_text.index("Second")
            return SimpleNamespace(
                sents=[
                    FakeSentence(0, boundary),
                    FakeSentence(boundary, len(masked_text)),
                ]
            )

    with patch.object(UnitSegmentor, "load_spacy_model", return_value=FakeNLP()):
        clauses = UnitSegmentor.get_math_aware_clauses_dep(MATH_HEAVY_TRACE)

    assert [(start, end) for start, end, _ in clauses] == [
        (0, MATH_HEAVY_TRACE.index("Second")),
        (MATH_HEAVY_TRACE.index("Second"), len(MATH_HEAVY_TRACE)),
    ]
    assert clauses[1][2].startswith("Second")
    assert "`value = 32`" in clauses[1][2]


@pytest.mark.parametrize(
    ("segmenter", "trace", "expected_text"),
    [
        (
            RTNewLineVerbose,
            "alpha\n\nbeta\n\n\ngamma",
            ["alpha\n\n", "beta\n\n\n", "gamma"],
        ),
        (
            RTNewLine,
            "alpha\nbeta\n\n\ngamma",
            ["alpha\nbeta\n\n\n", "gamma"],
        ),
        (RTNewLineVerbose, "\n\nalpha\n\n", ["\n\nalpha\n\n"]),
        (RTNewLineVerbose, "alpha\n\n   \n\n", ["alpha\n\n   \n\n"]),
    ],
)
def test_newline_segmenters_collapse_runs_without_whitespace_segments(
    segmenter, trace, expected_text
) -> None:
    offsets, labels = segmenter._segment(trace)

    _assert_valid_partition(trace, offsets)
    assert [trace[start:end] for start, end in offsets] == expected_text
    assert labels == ["UNK"] * len(offsets)


@pytest.mark.parametrize("trace", ["", "\n", "\n\n   \n"])
def test_newline_segmenters_return_no_segment_for_whitespace_only_trace(trace) -> None:
    for segmenter in (RTNewLine, RTNewLineVerbose):
        assert segmenter._segment(trace) == ([], [])


def test_resegmented_offset_validation_accepts_complete_partition() -> None:
    trace = "correct work error work"
    offsets, labels, error_index = _label_resegmented_steps(
        new_offsets=[(0, 13), (13, len(trace))],
        old_error_offset=(13, 18),
        reasoning_trace=trace,
        correct_label="0",
        error_label="-1",
        unannotated_label="x",
    )

    assert offsets == [[0, 13], [13, len(trace)]]
    assert labels == ["0", "-1"]
    assert error_index == 1


@pytest.mark.parametrize(
    ("offsets", "message"),
    [
        ([], "at least one"),
        ([(1, 23)], "start at zero"),
        ([(0, 0), (0, 23)], "Invalid RT-SEG offset"),
        ([(0, 8), (9, 23)], "contiguous"),
        ([(0, 10), (9, 23)], "contiguous"),
        ([(0, 22)], "complete reasoning trace"),
    ],
)
def test_resegmented_offset_validation_rejects_invalid_partitions(
    offsets, message
) -> None:
    trace = "correct work error work"
    with pytest.raises(ValueError, match=message):
        _label_resegmented_steps(
            new_offsets=offsets,
            old_error_offset=(13, 18),
            reasoning_trace=trace,
            correct_label="0",
            error_label="-1",
            unannotated_label="x",
        )


def test_resegmented_offset_validation_rejects_whitespace_only_segment() -> None:
    trace = "\n\nreasoning"
    with pytest.raises(ValueError, match="only whitespace"):
        _label_resegmented_steps(
            new_offsets=[(0, 2), (2, len(trace))],
            old_error_offset=(2, len(trace)),
            reasoning_trace=trace,
            correct_label="0",
            error_label="-1",
            unannotated_label="x",
        )
