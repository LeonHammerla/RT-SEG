from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict

from eval_utils.prm800k_utils import create_rtseg_dataset, load_prm800k


LOAD_KWARGS = {
    "dataset_name": "tasksource/PRM800K",
    "data_files": {"train": "phase2_train.jsonl", "test": "phase2_test.jsonl"},
    "raw_columns": [
        "labeler",
        "timestamp",
        "generation",
        "is_quality_control_question",
        "is_initial_screening_question",
        "question",
        "label",
    ],
    "finish_reason": "found_error",
    "error_rating": -1,
    "unannotated_label": "x",
    "streaming": False,
}


def _example(finish_reason: str = "found_error") -> dict:
    return {
        "labeler": "labeler-id",
        "timestamp": "2023-01-01T00:00:00",
        "generation": 1,
        "is_quality_control_question": False,
        "is_initial_screening_question": False,
        "question": {
            "problem": "What is 1 + 1?",
            "ground_truth_solution": "Add the numbers.",
            "ground_truth_answer": "2",
            "pre_generated_steps": [
                "Original first step.",
                "Original error step.",
                "Unannotated third step.",
                "Unannotated final answer.",
            ],
        },
        "label": {
            "finish_reason": finish_reason,
            "total_time": 100,
            "steps": [
                {
                    "completions": [
                        {"text": "First valid option.", "rating": 1, "flagged": False},
                        {"text": "Second valid option.", "rating": 1, "flagged": False},
                    ],
                    "human_completion": None,
                    "chosen_completion": 0,
                },
                {
                    "completions": [
                        {"text": "Valid option.", "rating": 1, "flagged": False},
                        {"text": "First error.", "rating": -1, "flagged": False},
                        {"text": "Second error.", "rating": -1, "flagged": False},
                    ],
                    "human_completion": None,
                    "chosen_completion": None,
                },
            ],
        },
    }


def test_load_prm800k_filters_and_selects_completions() -> None:
    raw = Dataset.from_list([_example(), _example("solution")])

    with patch("eval_utils.prm800k_utils.load_dataset", return_value=raw) as mocked:
        result = load_prm800k(**LOAD_KWARGS, split="test")

    mocked.assert_called_once_with(
        "tasksource/PRM800K",
        data_files={"train": "phase2_train.jsonl", "test": "phase2_test.jsonl"},
        split="test",
        streaming=False,
    )
    assert len(result) == 1
    expected_steps = [
        "First valid option.",
        "First error.",
        "Unannotated third step.",
        "Unannotated final answer.",
    ]
    assert result[0]["reasoning_trace"] == " ".join(expected_steps)
    assert [
        result[0]["reasoning_trace"][start:end]
        for start, end in result[0]["reasoning_steps"]
    ] == expected_steps
    assert result[0]["step_ratings"] == ["1", "-1", "x", "x"]
    assert len(result[0]["reasoning_steps"]) == len(result[0]["step_ratings"])
    assert result[0]["error_step_index"] == 1
    assert "label" not in result.column_names
    assert "question" not in result.column_names


def test_load_prm800k_concatenates_both_splits() -> None:
    raw = DatasetDict(
        {
            "train": Dataset.from_list([_example()]),
            "test": Dataset.from_list([_example()]),
        }
    )

    with patch("eval_utils.prm800k_utils.load_dataset", return_value=raw):
        result = load_prm800k(**LOAD_KWARGS, split=None)

    assert isinstance(result, Dataset)
    assert len(result) == 2
    for row in result:
        start, end = row["reasoning_steps"][-1]
        assert row["reasoning_trace"][start:end] == "Unannotated final answer."


def test_load_prm800k_rejects_steps_without_completions() -> None:
    example = _example()
    example["label"]["steps"][0]["completions"] = []
    raw = Dataset.from_list([example])

    with patch("eval_utils.prm800k_utils.load_dataset", return_value=raw):
        with pytest.raises(ValueError, match="contains no completions"):
            load_prm800k(**LOAD_KWARGS, split="train")


def test_create_rtseg_dataset_resegments_and_relabels(tmp_path) -> None:
    class FakeEngine:
        pass

    class FakeRTSeg:
        engines = [FakeEngine]
        label_fusion_type = "concat"
        seg_base_unit = "sent"

        def __call__(self, trace, **kwargs):
            old_error_start = trace.index("First error.")
            old_error_end = old_error_start + len("First error.")
            return [
                (0, old_error_start),
                (old_error_start, old_error_end + 5),
                (old_error_end + 5, len(trace)),
            ], ["setup", "computation", "answer"]

    raw = Dataset.from_list([_example()])
    with patch("eval_utils.prm800k_utils.load_dataset", return_value=raw):
        base_dataset = load_prm800k(**LOAD_KWARGS, split="train")

    result, output_directory = create_rtseg_dataset(
        base_dataset=base_dataset,
        rtseg=FakeRTSeg(),
        seed=17,
        top_k=1,
        output_root=tmp_path,
        correct_label="0",
        error_label="-1",
        unannotated_label="x",
    )

    assert result[0]["step_ratings"] == ["0", "-1", "x"]
    assert result[0]["rtseg_labels"] == ["setup", "computation", "answer"]
    assert result[0]["error_step_index"] == 1
    assert result[0]["rtseg_config"] == "FakeEngine_concat_sent"
    assert result[0]["sampling_seed"] == 17
    assert (tmp_path / "FakeEngine_concat_sent" / "dataset_info.json").exists()
