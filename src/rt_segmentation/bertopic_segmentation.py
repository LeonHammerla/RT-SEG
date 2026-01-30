import logging
from functools import lru_cache
from typing import List
from typing import Tuple, Literal

from bertopic import BERTopic
from cuml import UMAP, HDBSCAN
from nltk.tokenize import PunktSentenceTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .seg_base import SegBase
from .zeroshot_seq_classification import RTZeroShotSeqClassification

logger = logging.getLogger(__name__)


class RTBERTopicSegmentation(SegBase):
    @staticmethod
    @lru_cache(maxsize=1)
    def load_tokenizer():
        return PunktSentenceTokenizer()

    @staticmethod
    @lru_cache(maxsize=1)
    def load_model(model_name: Literal[
        "Qwen/Qwen2.5-7B-Instruct-1M",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Qwen/Qwen2.5-7B-Instruct"]
                   ):

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    @staticmethod
    @lru_cache(maxsize=1)
    def load_topic_model(embedding_model_name: Literal[
        "all-MiniLM-L6-v2",
        "Qwen/Qwen3-Embedding-0.6B",]):
        logger.info(f"Loading Dimensionality Reduction and Clustering models for BERTopic...")
        umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine', random_state=42,
                          build_algo="nn_descent")
        hdbscan_model = HDBSCAN(min_cluster_size=20, metric='euclidean', cluster_selection_method='eom',
                                prediction_data=True)
        vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

        logger.info(f"Loading Embedding model for BERTopic...")
        embedding_model = RTBERTopicSegmentation.load_embedding_model(embedding_model_name)

        topic_model = BERTopic(

            # Pipeline models
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            # Hyperparameters
            top_n_words=10,
            verbose=True
        )

        return topic_model

    @staticmethod
    @lru_cache(maxsize=1)
    def load_embedding_model(model_name: Literal[
        "all-MiniLM-L6-v2",
        "Qwen/Qwen3-Embedding-0.6B",]):
        embedding_model = SentenceTransformer(model_name)
        return embedding_model

    @staticmethod
    def chunks(s: str, n: int) -> list[str]:
        if n <= 0:
            raise ValueError("Chunk length must be positive")
        return [s[i:i + n] for i in range(0, len(s), n)]

    @staticmethod
    def prepare_prompt(documents: List[str]) -> str:
        main_prompt = """
        Segments:
        {documents}

        Correct label:
        """
        documents = [f"- {doc}\n" for doc in documents]
        return main_prompt.format(documents=documents)

    @staticmethod
    def get_topic_label(topic_model: BERTopic, labelling_model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                        system_prompt: str) -> List[str]:
        custom_labels = {}

        for row in tqdm(topic_model.get_topic_info().itertuples()):
            """if row.Topic == -1:
                continue"""
            documents = row.Representative_Docs
            prompt = RTBERTopicSegmentation.prepare_prompt(documents)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(labelling_model.device)

            generated_ids = labelling_model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            custom_labels[row.Topic] = response

        return custom_labels

    @staticmethod
    def _segment(
            trace: str,
            model_name: Literal[
                "Qwen/Qwen2.5-7B-Instruct-1M",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "Qwen/Qwen2.5-7B-Instruct"],
            embedding_model_name: Literal[
                "Qwen/Qwen2.5-7B-Instruct-1M",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "Qwen/Qwen2.5-7B-Instruct"] = "all-MiniLM-L6-v2",

            chunk_size: int = 100,
            prompt: str = "",
            system_prompt: str = "",
            max_retry: int = 30,
            all_custom_labels: bool = False,
            **kwargs
    ) -> Tuple[List[Tuple[int, int]], List[str]]:
        offsets = list(RTBERTopicSegmentation.load_tokenizer().span_tokenize(trace))
        documents = [trace[o[0]:o[1]] for o in offsets]
        documents = [s for s in documents if s]
        if len(documents) < 100:
            logger.info(f"Too few sentences ({len(documents)}) for BERTopic, Trying ZeroShotSeqClassification instead.")
            return RTZeroShotSeqClassification._segment(trace=trace, model_name="facebook/bart-large-mnli")

        topic_model = RTBERTopicSegmentation.load_topic_model(embedding_model_name)

        embeddings = topic_model.embedding_model.encode(documents)
        topics, probs = topic_model.fit_transform(documents, embeddings)
        if all_custom_labels:
            labelling_model, tokenizer = RTBERTopicSegmentation.load_model(model_name)
            custom_labels = RTBERTopicSegmentation.get_topic_label(topic_model=topic_model,
                                                                   labelling_model=labelling_model, tokenizer=tokenizer,
                                                                   system_prompt=system_prompt)
            labels = [custom_labels[t] for t in topics]
        else:
            topic_labels = topic_model.topic_labels_
            labels = [topic_labels[t] for t in topics]
        return offsets, labels
