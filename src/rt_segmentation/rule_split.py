from typing import List, Dict, Tuple, Literal
from surrealdb import Surreal, RecordID
from typing import List
import re
from nltk.tokenize import PunktSentenceTokenizer

from .seg_utils import bp, sdb_login, load_prompt, load_example_trace


class RTNewLine:
    @staticmethod
    def paragraph_ranges_regex(text: str) -> list[tuple[int, int]]:
        # Find positions after each \n\n or at start
        positions = [m.end() for m in re.finditer(r'\n\n|\A', text)]
        # Pair consecutive positions
        return list(zip(positions, positions[1:] + [len(text)]))

    @staticmethod
    def new_line_split():
        login_data = sdb_login()
        with Surreal(login_data["url"]) as db:
            db.signin({"username": login_data["user"], "password": login_data["pwd"]})
            db.use(login_data["ns"], login_data["db"])
            db.query("REMOVE TABLE newline_split;")
            db.query("DEFINE TABLE newline_split SCHEMALESS;")
            db.query("DEFINE INDEX idx_newline_split_id ON newline_split FIELDS id;")

            db.query("REMOVE TABLE has_newline_split;")
            db.query("DEFINE TABLE has_newline_split SCHEMALESS TYPE RELATION IN rtrace OUT newline_split;")
            db.query("DEFINE INDEX idx_rt_id ON has_newline_split FIELDS id;")
            db.query("DEFINE INDEX idx_rt_in ON has_newline_split FIELDS in;")
            db.query("DEFINE INDEX idx_rt_out ON has_newline_split FIELDS out;")

            results = db.query("SELECT * from rtrace")

            for res in results:
                rt = res.get("rt")
                offsets = RTNewLine.paragraph_ranges_regex(rt)

                split_id = RecordID("newline_split", res.get("id").id)
                db.upsert(split_id, {"split": offsets})
                db.insert_relation("has_newline_split", {"in": res.get("id"), "out": split_id})


class RTRuleBased:
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
        return RTRuleBased.has_marker(sentence, RTRuleBased.INFERENCE_MARKERS)

    @staticmethod
    def goal_shift(sentence: str) -> bool:
        return RTRuleBased.has_marker(sentence, RTRuleBased.GOAL_MARKERS)

    @staticmethod
    def is_final(sentence: str) -> bool:
        return RTRuleBased.has_marker(sentence, RTRuleBased.FINAL_MARKERS)

    @staticmethod
    def starts_new_segment(sentence: str) -> bool:
        return (
            RTRuleBased.introduces_inference(sentence)
            or RTRuleBased.has_marker(sentence, RTRuleBased.CONTRAST_MARKERS)
            or RTRuleBased.has_marker(sentence, RTRuleBased.REVISION_MARKERS)
            or RTRuleBased.goal_shift(sentence)
            or RTRuleBased.is_final(sentence)
        )

    # -----------------------------
    # Sentence spans (core change)
    # -----------------------------
    @staticmethod
    def sentence_spans(text: str) -> List[tuple]:
        """
        Returns a list of (start, end) sentence spans into the original text.
        """
        return list(RTRuleBased._sent_tokenizer.span_tokenize(text))

    # -----------------------------
    # Segment offsets
    # -----------------------------
    @staticmethod
    def segment(text: str) -> List[Dict[str, int]]:
        spans = RTRuleBased.sentence_spans(text)

        if not spans:
            return []

        segments = []
        current_start = spans[0][0]

        for i, (sent_start, sent_end) in enumerate(spans):
            sentence_text = text[sent_start:sent_end]

            if i > 0 and RTRuleBased.starts_new_segment(sentence_text):
                # close previous segment exactly at this sentence start
                segments.append((current_start, sent_start))
                current_start = sent_start

        # final segment goes to end of text
        segments.append((current_start, len(text)))

        return segments

    @staticmethod
    def rule_split():
        login_data = sdb_login()
        with Surreal(login_data["url"]) as db:
            db.signin({"username": login_data["user"], "password": login_data["pwd"]})
            db.use(login_data["ns"], login_data["db"])
            db.query("REMOVE TABLE rule_split;")
            db.query("DEFINE TABLE rule_split SCHEMALESS;")
            db.query("DEFINE INDEX idx_rule_split_id ON rule_split FIELDS id;")

            db.query("REMOVE TABLE has_rule_split;")
            db.query("DEFINE TABLE has_rule_split SCHEMALESS TYPE RELATION IN rtrace OUT rule_split;")
            db.query("DEFINE INDEX idx_rt_id ON has_rule_split FIELDS id;")
            db.query("DEFINE INDEX idx_rt_in ON has_rule_split FIELDS in;")
            db.query("DEFINE INDEX idx_rt_out ON has_rule_split FIELDS out;")

            results = db.query("SELECT * from rtrace")

            for res in results:
                rt = res.get("rt")
                offsets = RTRuleBased.segment(rt)

                split_id = RecordID("rule_split", res.get("id").id)
                db.upsert(split_id, {"split": offsets})
                db.insert_relation("has_rule_split", {"in": res.get("id"), "out": split_id})