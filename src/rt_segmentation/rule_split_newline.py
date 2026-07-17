import re

from .seg_base import SegBase


def _split_after_newline_runs(
    trace: str,
    separator_pattern: str,
) -> tuple[list[tuple[int, int]], list[str]]:
    """Split after newline runs without emitting empty/whitespace segments."""
    if not trace or not trace.strip():
        return [], []

    starts = [0]
    for match in re.finditer(separator_pattern, trace):
        boundary = match.end()
        if boundary >= len(trace):
            continue
        # A leading run, or a run preceded only by whitespace since the last
        # boundary, belongs to the following reasoning segment.
        if trace[starts[-1]:boundary].strip():
            starts.append(boundary)

    # If the trace ends in whitespace after the last accepted boundary, merge
    # it into the preceding reasoning segment.
    while len(starts) > 1 and not trace[starts[-1]:].strip():
        starts.pop()

    offsets = list(zip(starts, starts[1:] + [len(trace)]))
    return offsets, ["UNK" for _ in offsets]


class RTNewLine(SegBase):
    @staticmethod
    def _segment(trace: str, **kwargs) -> tuple[list[tuple[int, int]], list[str]]:
        return _split_after_newline_runs(trace, r'\n{2,}')


class RTNewLineVerbose(SegBase):
    @staticmethod
    def _segment(trace: str, **kwargs) -> tuple[list[tuple[int, int]], list[str]]:
        return _split_after_newline_runs(trace, r'\n+')
