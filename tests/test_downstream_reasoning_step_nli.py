from pathlib import Path
from unittest.mock import patch

from datasets import Dataset

import downstream_reasoning_step_nli
from rt_segmentation import RTPlainSegmenter


def test_main_runs_segmentation_and_reasoning_step_nli_training(tmp_path) -> None:
    input_dataset = Dataset.from_dict({"reasoning_trace": ["A trace."]})
    sample_dataset = Dataset.from_dict(
        {
            "premise": ["first"],
            "hypothesis": ["second"],
            "labels": [1],
            "source_index": [0],
        }
    )
    dataset_path = tmp_path / "segmented"

    with (
        patch.object(
            downstream_reasoning_step_nli,
            "create_rtseg_dataset_main",
            return_value=(input_dataset, dataset_path),
        ) as mocked_create,
        patch.object(
            downstream_reasoning_step_nli,
            "load_from_disk",
            return_value=input_dataset,
        ) as mocked_load,
        patch.object(
            downstream_reasoning_step_nli,
            "extract_reasoning_step_pair_samples",
            return_value=sample_dataset,
        ) as mocked_extract,
        patch.object(
            downstream_reasoning_step_nli,
            "train_cross_validated_classifier",
        ) as mocked_train,
    ):
        result = downstream_reasoning_step_nli.main(
            rtseg_engines=[RTPlainSegmenter],
            rtseg_aligner=None,
            rtseg_label_fusion_type="concat",
            rtseg_base_unit="sent",
            rid="baseline",
            reuse_existing_dataset=False,
        )

    mocked_create.assert_called_once_with(
        rtseg_engines=[RTPlainSegmenter],
        rtseg_aligner=None,
        rtseg_label_fusion_type="concat",
        rtseg_base_unit="sent",
        rtseg_top_k=1000,
        rtseg_seed=42,
    )
    mocked_load.assert_called_once_with(dataset_path)
    mocked_extract.assert_called_once_with(
        dataset=input_dataset,
        fake_per_real=1,
        seed=42,
    )
    expected_output = (
        Path(downstream_reasoning_step_nli.__file__).resolve().parents[1]
        / "data"
        / "reasoning_step_nli_modernbert_baseline"
    )
    assert mocked_train.call_args.kwargs["sample_dataset"] is sample_dataset
    assert mocked_train.call_args.kwargs["output_directory"] == expected_output
    assert mocked_train.call_args.kwargs["include_rtseg_labels"] is True
    assert mocked_train.call_args.kwargs["max_length"] == 512
    assert mocked_train.call_args.kwargs["calibration_fraction"] == 0.2
    assert mocked_train.call_args.kwargs["use_class_weights"] is True
    assert result == {
        "rid": "baseline",
        "dataset_path": str(dataset_path),
        "output_directory": str(expected_output),
    }


def test_main_reuses_existing_dataset_for_requested_top_k(tmp_path) -> None:
    input_dataset = Dataset.from_dict({"reasoning_trace": ["A trace."]})
    sample_dataset = Dataset.from_dict(
        {
            "premise": ["first"],
            "hypothesis": ["second"],
            "labels": [1],
            "source_index": [0],
        }
    )
    dataset_path = tmp_path / "segmented"
    dataset_path.mkdir()

    with (
        patch.object(
            downstream_reasoning_step_nli,
            "get_rtseg_dataset_path",
            return_value=dataset_path,
        ) as mocked_get_path,
        patch.object(
            downstream_reasoning_step_nli,
            "create_rtseg_dataset_main",
        ) as mocked_create,
        patch.object(
            downstream_reasoning_step_nli,
            "load_from_disk",
            return_value=input_dataset,
        ) as mocked_load,
        patch.object(
            downstream_reasoning_step_nli,
            "extract_reasoning_step_pair_samples",
            return_value=sample_dataset,
        ),
        patch.object(
            downstream_reasoning_step_nli,
            "train_cross_validated_classifier",
        ),
    ):
        result = downstream_reasoning_step_nli.main(
            rtseg_engines=[RTPlainSegmenter],
            rtseg_aligner=None,
            rtseg_label_fusion_type="concat",
            rtseg_base_unit="sent",
            rid="baseline",
            rtseg_top_k=2500,
        )

    mocked_create.assert_not_called()
    mocked_get_path.assert_called_once_with(
        rtseg_engines=[RTPlainSegmenter],
        rtseg_label_fusion_type="concat",
        rtseg_base_unit="sent",
        rtseg_top_k=2500,
    )
    mocked_load.assert_called_once_with(dataset_path)
    assert result["dataset_path"] == str(dataset_path)


def test_multi_main_can_run_declared_configs_sequentially() -> None:
    configs = [
        {
            "rts_engines": [RTPlainSegmenter],
            "rts_aligner": None,
            "rts_label_fusion_type": "concat",
            "rts_base_unit": "sent",
            "rid": "sentence-baseline",
        },
        {
            "rts_engines": [RTPlainSegmenter],
            "rts_aligner": None,
            "rts_label_fusion_type": "concat",
            "rts_base_unit": "clause",
            "rid": "clause-baseline",
        },
    ]

    def fake_main(**kwargs):
        return {
            "rid": kwargs["rid"],
            "dataset_path": f"dataset/{kwargs['rid']}",
            "output_directory": f"output/{kwargs['rid']}",
        }

    with patch.object(
        downstream_reasoning_step_nli,
        "main",
        side_effect=fake_main,
    ) as mocked:
        results = downstream_reasoning_step_nli.multi_main(
            configs,
            use_multiprocessing=False,
            rtseg_top_k=2500,
        )

    assert [result["rid"] for result in results] == [
        "sentence-baseline",
        "clause-baseline",
    ]
    assert [call.kwargs["rid"] for call in mocked.call_args_list] == [
        "sentence-baseline",
        "clause-baseline",
    ]
    assert all(
        call.kwargs["reuse_existing_dataset"] is True
        for call in mocked.call_args_list
    )
    assert all(
        call.kwargs["rtseg_top_k"] == 2500
        for call in mocked.call_args_list
    )
