from types import SimpleNamespace

import torch
import pytest
from datasets import Dataset

from eval_utils.first_error_detect_prm800k import (
    _add_and_validate_special_tokens,
    _resize_and_validate_model_embeddings,
    _validate_flash_attention_environment,
    extract_first_error_samples,
)


def _dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "reasoning_trace": "first second third fourth",
                "reasoning_steps": [[0, 5], [6, 12], [13, 18], [19, 25]],
                "step_ratings": ["0", "0", "-1", "x"],
                "rtseg_labels": ["setup", "inference", "mistake", "answer"],
            }
        ]
    )


def test_extract_first_error_samples_builds_prefix_sequences() -> None:
    result = extract_first_error_samples(
        dataset=_dataset(),
        correct_per_error=2,
        seed=7,
        step_token="[STEP]",
        trace_label_token="[LABEL]",
        correct_step_label="0",
        error_step_label="-1",
    )

    assert sorted(result["labels"]) == [0, 0, 1]
    error_sample = result.filter(lambda row: row["labels"] == 1)[0]
    assert error_sample["text"] == (
        "[STEP] first [LABEL] setup "
        "[STEP] second [LABEL] inference "
        "[STEP] third [LABEL] mistake"
    )
    assert "fourth" not in error_sample["text"]
    assert error_sample["source_index"] == 0
    assert error_sample["target_step_index"] == 2


def test_extract_first_error_samples_is_seeded() -> None:
    dataset = Dataset.from_list(
        [
            {
                "reasoning_trace": "a b c d e",
                "reasoning_steps": [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
                "step_ratings": ["0", "0", "0", "0", "-1"],
                "rtseg_labels": ["A", "B", "C", "D", "E"],
            }
        ]
    )
    kwargs = {
        "dataset": dataset,
        "correct_per_error": 2,
        "seed": 11,
        "step_token": "[STEP]",
        "trace_label_token": "[LABEL]",
        "correct_step_label": "0",
        "error_step_label": "-1",
    }

    first = extract_first_error_samples(**kwargs)
    second = extract_first_error_samples(**kwargs)

    assert first["target_step_index"] == second["target_step_index"]
    assert first["text"] == second["text"]


def test_extract_retains_errors_and_balances_correct_prefixes_globally() -> None:
    dataset = Dataset.from_list(
        [
            {
                "reasoning_trace": "error remainder",
                "reasoning_steps": [[0, 5], [6, 15]],
                "step_ratings": ["-1", "x"],
                "rtseg_labels": ["mistake", "answer"],
            },
            {
                "reasoning_trace": "one two error remainder",
                "reasoning_steps": [[0, 3], [4, 7], [8, 13], [14, 23]],
                "step_ratings": ["0", "0", "-1", "x"],
                "rtseg_labels": ["setup", "work", "mistake", "answer"],
            },
        ]
    )

    result = extract_first_error_samples(
        dataset=dataset,
        correct_per_error=2,
        seed=7,
        step_token="[STEP]",
        trace_label_token="[LABEL]",
        correct_step_label="0",
        error_step_label="-1",
    )

    assert result["labels"].count(1) == 2
    assert result["labels"].count(0) == 2
    assert len(result) == 4
    assert sorted(
        row["source_index"] for row in result if row["labels"] == 1
    ) == [0, 1]
    assert set(
        row["target_step_index"] for row in result if row["labels"] == 0
    ) == {0, 1}


def test_global_correct_selection_balances_traces_before_using_longer_ones() -> None:
    dataset = Dataset.from_list(
        [
            {
                "reasoning_trace": "a error",
                "reasoning_steps": [[0, 1], [2, 7]],
                "step_ratings": ["0", "-1"],
                "rtseg_labels": ["work", "mistake"],
            },
            {
                "reasoning_trace": "a b c error",
                "reasoning_steps": [[0, 1], [2, 3], [4, 5], [6, 11]],
                "step_ratings": ["0", "0", "0", "-1"],
                "rtseg_labels": ["work", "work", "work", "mistake"],
            },
        ]
    )

    result = extract_first_error_samples(
        dataset=dataset,
        correct_per_error=1,
        seed=7,
        step_token="[STEP]",
        trace_label_token="[LABEL]",
        correct_step_label="0",
        error_step_label="-1",
    )

    correct_source_indices = sorted(
        row["source_index"] for row in result if row["labels"] == 0
    )
    assert correct_source_indices == [0, 1]


def test_special_tokens_and_model_embeddings_are_resized() -> None:
    class FakeTokenizer:
        def __init__(self):
            self.vocabulary = {"[UNK]": 0, "base": 1}
            self.unk_token_id = 0

        def __len__(self):
            return len(self.vocabulary)

        def get_vocab(self):
            return dict(self.vocabulary)

        def add_special_tokens(self, tokens):
            added = 0
            for token in tokens["additional_special_tokens"]:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
                    added += 1
            return added

        def convert_tokens_to_ids(self, token):
            return self.vocabulary.get(token, self.unk_token_id)

        def tokenize(self, token):
            return [token] if token in self.vocabulary else ["[UNK]"]

    class FakeModel:
        def __init__(self):
            self.input_embeddings = torch.nn.Embedding(2, 4)
            self.output_embeddings = torch.nn.Linear(4, 2, bias=False)
            self.classifier = torch.nn.Linear(4, 2)
            self.config = SimpleNamespace(vocab_size=2, num_labels=2)

        def get_input_embeddings(self):
            return self.input_embeddings

        def get_output_embeddings(self):
            return self.output_embeddings

        def resize_token_embeddings(self, vocabulary_size):
            self.input_embeddings = torch.nn.Embedding(vocabulary_size, 4)
            self.output_embeddings = torch.nn.Linear(
                4, vocabulary_size, bias=False
            )
            self.config.vocab_size = vocabulary_size

    tokenizer = FakeTokenizer()
    tokenizer_result = _add_and_validate_special_tokens(
        tokenizer=tokenizer,
        special_tokens=["[STEP]", "[LABEL]"],
    )
    model = FakeModel()
    model_result = _resize_and_validate_model_embeddings(
        model=model,
        tokenizer=tokenizer,
        expected_label_count=2,
    )

    assert tokenizer_result["added_token_count"] == 2
    assert tokenizer_result["vocabulary_size_after"] == 4
    assert tokenizer_result["token_ids"] == {"[STEP]": 2, "[LABEL]": 3}
    assert model_result["input_vocabulary_size_after"] == 4
    assert model_result["output_vocabulary_size_after"] == 4
    assert model_result["config_vocabulary_size_after"] == 4
    assert model_result["classification_label_count"] == 2
    assert model_result["classifier_output_size"] == 2


def test_training_rejects_non_flash_attention_backend() -> None:
    with pytest.raises(ValueError, match="flash_attention_2"):
        _validate_flash_attention_environment(
            attention_implementation="sdpa",
            use_bf16=True,
            use_fp16=False,
        )


def test_training_requires_exactly_one_half_precision_dtype() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        _validate_flash_attention_environment(
            attention_implementation="flash_attention_2",
            use_bf16=False,
            use_fp16=False,
        )
