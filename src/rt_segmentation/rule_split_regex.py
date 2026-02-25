from typing import List, Dict, Tuple, Literal, Any
from typing import List
import re
from nltk.tokenize import PunktSentenceTokenizer

from .seg_base import SegBase


class RTRuleRegex(SegBase):
    INFERENCE_MARKERS = [
        r"\btherefore\b", r"\bthus\b", r"\bhence\b", r"\bso\b",
        r"\bfollows that\b", r"\bimplies\b", r"\bwe conclude\b"
    ]

    CONTRAST_MARKERS = [
        r"\bbut\b", r"\bhowever\b", r"\bthough\b", r"\binstead\b",
        r"\bactually\b", r"\bnevertheless\b", r"\bon the other hand\b"
    ]

    REVISION_MARKERS = [
        r"\bthis is wrong\b", r"\bthat was wrong\b", r"\bi was mistaken\b",
        r"\blet me reconsider\b", r"\bwait\b"
    ]

    GOAL_MARKERS = [
        r"\bnow we\b", r"\blet us\b", r"\bnext\b",
        r"\bconsider\b", r"\bwe need to\b", r"\bthe goal\b"
    ]

    FINAL_MARKERS = [
        r"\bthe answer is\b", r"\btherefore the answer\b",
        r"\bin conclusion\b", r"\bfinal answer\b"
    ]

    _sent_tokenizer = PunktSentenceTokenizer()

    # -----------------------------
    # Marker helpers
    # -----------------------------
    @staticmethod
    def has_marker(text: str, markers: List[str]) -> bool:
        text = text.lower()
        return any(re.search(m, text) for m in markers)

    @staticmethod
    def introduces_inference(sentence: str) -> bool:
        return RTRuleRegex.has_marker(sentence, RTRuleRegex.INFERENCE_MARKERS)

    @staticmethod
    def goal_shift(sentence: str) -> bool:
        return RTRuleRegex.has_marker(sentence, RTRuleRegex.GOAL_MARKERS)

    @staticmethod
    def is_final(sentence: str) -> bool:
        return RTRuleRegex.has_marker(sentence, RTRuleRegex.FINAL_MARKERS)

    @staticmethod
    def starts_new_segment(sentence: str) -> bool:
        return (
            RTRuleRegex.introduces_inference(sentence)
            or RTRuleRegex.has_marker(sentence, RTRuleRegex.CONTRAST_MARKERS)
            or RTRuleRegex.has_marker(sentence, RTRuleRegex.REVISION_MARKERS)
            or RTRuleRegex.goal_shift(sentence)
            or RTRuleRegex.is_final(sentence)
        )

    # -----------------------------
    # Sentence spans (core change)
    # -----------------------------
    @staticmethod
    def sentence_spans(text: str) -> List[tuple]:
        """
        Returns a list of (start, end) sentence spans into the original text.
        """
        return list(RTRuleRegex._sent_tokenizer.span_tokenize(text))

    # -----------------------------
    # Segment offsets
    # -----------------------------
    @staticmethod
    def _segment(trace: str,
                 seg_base_unit: Literal["sent", "clause"],
                 **kwargs) -> tuple[list[Any], list[str]] | list[Any]:

        spans = SegBase.get_base_offsets(trace, seg_base_unit=seg_base_unit)

        if not spans:
            return []

        segments = []
        current_start = spans[0][0]

        for i, (sent_start, sent_end) in enumerate(spans):
            sentence_text = trace[sent_start:sent_end]

            if i > 0 and RTRuleRegex.starts_new_segment(sentence_text):
                # close previous segment exactly at this sentence start
                segments.append((current_start, sent_start))
                current_start = sent_start

        # final segment goes to end of text
        segments.append((current_start, len(trace)))
        return segments, ["UNK" for _ in segments]