import pytest
from datasets import Dataset

from eval_utils.reasoning_step_nli_prm800k import (
    _format_step_for_model,
    extract_reasoning_step_pair_samples,
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


def test_extract_step_pairs_builds_balanced_same_trace_examples() -> None:
    result = extract_reasoning_step_pair_samples(
        dataset=_dataset(),
        fake_per_real=1,
        seed=7,
    )

    assert result["labels"].count(1) == 3
    assert result["labels"].count(0) == 3
    assert set(result["source_index"]) == {0}

    for row in result:
        first = row["premise_step_index"]
        second = row["hypothesis_step_index"]
        if row["labels"] == 1:
            assert second == first + 1
            assert row["pair_type"] == "following"
        else:
            assert second != first
            assert second != first + 1
            assert row["pair_type"] == "not_following"


def test_extract_step_pairs_retains_rtseg_and_correctness_metadata() -> None:
    result = extract_reasoning_step_pair_samples(
        dataset=_dataset(),
        fake_per_real=1,
        seed=11,
    )
    error_to_unannotated = next(
        row
        for row in result
        if row["premise_step_index"] == 2
        and row["hypothesis_step_index"] == 3
    )

    assert error_to_unannotated["labels"] == 1
    assert error_to_unannotated["premise"] == "third"
    assert error_to_unannotated["hypothesis"] == "fourth"
    assert error_to_unannotated["premise_rtseg_label"] == "mistake"
    assert error_to_unannotated["hypothesis_rtseg_label"] == "answer"
    assert error_to_unannotated["premise_step_rating"] == "-1"
    assert error_to_unannotated["hypothesis_step_rating"] == "x"


def test_two_step_trace_uses_reversed_pair_as_negative() -> None:
    dataset = Dataset.from_list(
        [
            {
                "reasoning_trace": "one two",
                "reasoning_steps": [[0, 3], [4, 7]],
                "step_ratings": ["-1", "x"],
                "rtseg_labels": ["work", "answer"],
            }
        ]
    )

    result = extract_reasoning_step_pair_samples(
        dataset=dataset,
        fake_per_real=1,
        seed=3,
    )

    real = next(row for row in result if row["labels"] == 1)
    fake = next(row for row in result if row["labels"] == 0)
    assert (real["premise_step_index"], real["hypothesis_step_index"]) == (0, 1)
    assert (fake["premise_step_index"], fake["hypothesis_step_index"]) == (1, 0)


def test_extract_step_pairs_is_seeded() -> None:
    kwargs = {"dataset": _dataset(), "fake_per_real": 1, "seed": 17}
    first = extract_reasoning_step_pair_samples(**kwargs)
    second = extract_reasoning_step_pair_samples(**kwargs)

    assert first.to_dict() == second.to_dict()


def test_extract_step_pairs_can_limit_dataset_size_deterministically() -> None:
    kwargs = {
        "dataset": _dataset(),
        "fake_per_real": 1,
        "seed": 17,
        "n": 4,
    }
    first = extract_reasoning_step_pair_samples(**kwargs)
    second = extract_reasoning_step_pair_samples(**kwargs)

    assert len(first) == 4
    assert first["labels"].count(0) == 2
    assert first["labels"].count(1) == 2
    assert first.to_dict() == second.to_dict()


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_extract_step_pairs_rejects_invalid_limit(n) -> None:
    with pytest.raises(ValueError, match="positive integer or None"):
        extract_reasoning_step_pair_samples(
            dataset=_dataset(),
            fake_per_real=1,
            seed=17,
            n=n,
        )


def test_extract_step_pairs_rejects_limit_larger_than_dataset() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        extract_reasoning_step_pair_samples(
            dataset=_dataset(),
            fake_per_real=1,
            seed=17,
            n=7,
        )


def test_extract_step_pairs_rejects_misaligned_metadata() -> None:
    dataset = Dataset.from_list(
        [
            {
                "reasoning_trace": "one two",
                "reasoning_steps": [[0, 3], [4, 7]],
                "step_ratings": ["0", "-1"],
                "rtseg_labels": ["work"],
            }
        ]
    )

    with pytest.raises(ValueError, match="must be aligned"):
        extract_reasoning_step_pair_samples(
            dataset=dataset,
            fake_per_real=1,
            seed=1,
        )


def test_model_format_can_include_or_ablate_rtseg_labels() -> None:
    with_labels = _format_step_for_model(
        step_text="Therefore x = 2.",
        rtseg_label="Result Consolidation",
        include_rtseg_labels=True,
        step_token="[STEP]",
        rtseg_label_token="[RTSEG_LABEL]",
    )
    without_labels = _format_step_for_model(
        step_text="Therefore x = 2.",
        rtseg_label="Result Consolidation",
        include_rtseg_labels=False,
        step_token="[STEP]",
        rtseg_label_token="[RTSEG_LABEL]",
    )

    assert with_labels == (
        "[STEP] Therefore x = 2. [RTSEG_LABEL] Result Consolidation"
    )
    assert without_labels == "[STEP] Therefore x = 2."
