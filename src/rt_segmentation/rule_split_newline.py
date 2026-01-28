from typing import List, Dict, Tuple, Literal
from surrealdb import Surreal, RecordID
from typing import List
import re
from nltk.tokenize import PunktSentenceTokenizer

from .seg_utils import bp, sdb_login, load_prompt, load_example_trace
from .seg_base import SegBase


class RTNewLine(SegBase):
    @staticmethod
    def _segment(trace: str, **kwargs) -> list[tuple[int, int]]:
        # Find positions after each \n\n or at start
        positions = [m.end() for m in re.finditer(r'\n\n|\A', trace)]
        # Pair consecutive positions
        return list(zip(positions, positions[1:] + [len(trace)]))
