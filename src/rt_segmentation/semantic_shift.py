from functools import lru_cache
from typing import List, Dict, Tuple, Literal, Any
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import PunktSentenceTokenizer
import numpy as np

from .seg_base import SegBase


class RTEmbeddingBasedSemanticShift(SegBase):
    @staticmethod
    @lru_cache(maxsize=1)
    def load_tokenizer():
        return PunktSentenceTokenizer()

    @staticmethod
    @lru_cache(maxsize=1)
    def load_embedding_model(model_name: Literal[
        "all-MiniLM-L6-v2",
        "Qwen/Qwen3-Embedding-0.6B",]):
        embedding_model = SentenceTransformer(model_name)
        return embedding_model

    @staticmethod
    def _get_centroid(vectors):
        return np.mean(vectors, axis=0)


    @staticmethod
    def _segment(
            trace: str,
            seg_base_unit: Literal["sent", "clause"],
            model_name: Literal["all-MiniLM-L6-v2"] = "all-MiniLM-L6-v2",
            tolerance: float = 0.15,
            min_threshold: float = 0.4,
            **kwargs
    ) -> tuple[list[Any], list[str]]:


        offsets = SegBase.get_base_offsets(trace, seg_base_unit=seg_base_unit)

        # trace = nltk.sent_tokenize(trace)
        strace = [trace[tr[0]:tr[1]] for tr in offsets]
        embeddings = RTEmbeddingBasedSemanticShift.load_embedding_model(model_name).encode(strace)
        # Current Segment State
        current_segment_sents = [strace[0]]
        current_segment_vecs = [embeddings[0]]

        running_avg_sim = 1.0

        final_offsets = []

        curr_offset = offsets[0]

        i = 1
        while i < len(strace):
            sent = strace[i]
            vec = embeddings[i]
            offset = offsets[i]

            centroid = RTEmbeddingBasedSemanticShift._get_centroid(current_segment_vecs)
            sim = util.cos_sim(centroid, vec).item()

            dynamic_threshold = running_avg_sim - tolerance

            effective_threshold = max(dynamic_threshold, min_threshold)
            is_match = sim >= effective_threshold

            if not is_match and (i + 1 < len(strace)):
                next_vec = embeddings[i + 1]
                sim_next = util.cos_sim(next_vec, centroid).item()

                # If the NEXT sentence fits the current segment, the current one
                # might be a "bridge" or "outlier" (e.g. "Note: remember X").
                # We keep it to preserve flow.
                if sim_next >= effective_threshold:
                    is_match = True


            if is_match:
                current_segment_sents.append(sent)
                current_segment_vecs.append(vec)

                running_avg_sim = (running_avg_sim + sim) / 2
                curr_offset = (curr_offset[0], offset[1])
            else:
                final_offsets.append(curr_offset)
                curr_offset = offset

                current_segment_sents = [sent]
                current_segment_vecs = [vec]
                running_avg_sim = 1.0
            i += 1

        if current_segment_sents:
            final_offsets.append(curr_offset)

        return final_offsets, ["UNK" for _ in final_offsets]