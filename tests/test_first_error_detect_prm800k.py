from types import SimpleNamespace

import torch
import pytest
from datasets import Dataset

from eval_utils.first_error_detect_prm800k import (
    _add_and_validate_special_tokens,
    _balanced_class_weights,
    _make_weighted_trainer_class,
    _positive_class_probabilities,
    _resize_and_validate_model_embeddings,
    _select_macro_f1_threshold,
    _split_grouped_calibration,
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


def test_extract_first_error_samples_can_exclude_rtseg_labels() -> None:
    result = extract_first_error_samples(
        dataset=_dataset(),
        correct_per_error=2,
        seed=7,
        step_token="[STEP]",
        trace_label_token="[LABEL]",
        correct_step_label="0",
        error_step_label="-1",
        include_rtseg_labels=False,
    )

    error_sample = result.filter(lambda row: row["labels"] == 1)[0]
    assert error_sample["text"] == (
        "[STEP] first [STEP] second [STEP] third"
    )
    assert "[LABEL]" not in error_sample["text"]
    assert "mistake" not in error_sample["text"]


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


def test_balance_classes_retains_all_errors_when_enough_correct_samples_exist() -> None:
    dataset = Dataset.from_list(
        [
            {
                "reasoning_trace": "a b error",
                "reasoning_steps": [[0, 1], [2, 3], [4, 9]],
                "step_ratings": ["0", "0", "-1"],
                "rtseg_labels": ["work", "work", "mistake"],
            },
            {
                "reasoning_trace": "a error",
                "reasoning_steps": [[0, 1], [2, 7]],
                "step_ratings": ["0", "-1"],
                "rtseg_labels": ["work", "mistake"],
            },
        ]
    )

    result = extract_first_error_samples(
        dataset=dataset,
        correct_per_error=10,
        seed=7,
        step_token="[STEP]",
        trace_label_token="[LABEL]",
        correct_step_label="0",
        error_step_label="-1",
        balance_classes=True,
    )

    assert result["labels"].count(1) == 2
    assert result["labels"].count(0) == 2
    assert len(result) == 4


def test_balance_classes_uses_largest_possible_balanced_subset() -> None:
    dataset = Dataset.from_list(
        [
            {
                "reasoning_trace": "error",
                "reasoning_steps": [[0, 5]],
                "step_ratings": ["-1"],
                "rtseg_labels": ["mistake"],
            },
            {
                "reasoning_trace": "error",
                "reasoning_steps": [[0, 5]],
                "step_ratings": ["-1"],
                "rtseg_labels": ["mistake"],
            },
            {
                "reasoning_trace": "a error",
                "reasoning_steps": [[0, 1], [2, 7]],
                "step_ratings": ["0", "-1"],
                "rtseg_labels": ["work", "mistake"],
            },
        ]
    )

    result = extract_first_error_samples(
        dataset=dataset,
        correct_per_error=3,
        seed=11,
        step_token="[STEP]",
        trace_label_token="[LABEL]",
        correct_step_label="0",
        error_step_label="-1",
        balance_classes=True,
    )

    assert result["labels"].count(1) == 1
    assert result["labels"].count(0) == 1
    assert len(result) == 2


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


def test_macro_f1_threshold_is_selected_from_probabilities() -> None:
    threshold = _select_macro_f1_threshold(
        probabilities=[0.10, 0.40, 0.35, 0.80],
        labels=[0, 0, 1, 1],
    )

    assert threshold == pytest.approx(0.35)


def test_macro_f1_threshold_does_not_optimize_only_positive_class() -> None:
    threshold = _select_macro_f1_threshold(
        probabilities=[0.90, 0.80, 0.70, 0.60, 0.20, 0.10],
        labels=[1, 0, 0, 0, 1, 1],
    )

    # Positive-class F1 is maximized by predicting every sample as positive
    # (threshold 0.10), whereas macro F1 is maximized at threshold 0.90.
    assert threshold == pytest.approx(0.90)


def test_positive_class_probabilities_use_binary_softmax() -> None:
    probabilities = _positive_class_probabilities(
        [[0.0, 0.0], [0.0, float(torch.log(torch.tensor(3.0)))]],
    )

    assert probabilities.tolist() == pytest.approx([0.5, 0.75])


def test_balanced_class_weights_upweight_the_minority_class() -> None:
    weights = _balanced_class_weights([0, 0, 0, 0, 0, 0, 1, 1])

    assert weights == pytest.approx([2 / 3, 2.0])


def test_grouped_calibration_split_is_disjoint_and_stratified() -> None:
    import numpy as np

    labels = np.tile([0, 1], 10)
    groups = np.repeat(np.arange(10), 2)
    outer_train_indices = np.arange(len(labels))

    fit_indices, calibration_indices = _split_grouped_calibration(
        outer_train_indices=outer_train_indices,
        labels=labels,
        groups=groups,
        calibration_fraction=0.2,
        seed=43,
    )

    assert set(groups[fit_indices]).isdisjoint(groups[calibration_indices])
    assert set(labels[fit_indices]) == {0, 1}
    assert set(labels[calibration_indices]) == {0, 1}
    assert len(calibration_indices) == 4


def test_weighted_trainer_uses_weighted_cross_entropy() -> None:
    class FakeTrainer:
        def __init__(self, **kwargs):
            del kwargs

    class FakeModel:
        def __call__(self, *, logits):
            return SimpleNamespace(logits=logits)

    trainer_class = _make_weighted_trainer_class(FakeTrainer)
    trainer = trainer_class(class_weights=[1.0, 3.0])
    logits = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    labels = torch.tensor([0, 1])
    expected = torch.nn.functional.cross_entropy(
        logits,
        labels,
        weight=torch.tensor([1.0, 3.0]),
    )

    loss = trainer.compute_loss(
        FakeModel(),
        {"logits": logits, "labels": labels},
    )

    assert loss == pytest.approx(expected)
