from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset

import downstream_firsterror
from rt_segmentation import OffsetFusionGraph, RTNewLine, RTPlainSegmenter, RTRuleRegex


def _configs() -> list[dict]:
    return [
        {
            "rts_engines": [RTNewLine, RTRuleRegex],
            "rts_aligner": OffsetFusionGraph,
            "rts_label_fusion_type": "concat",
            "rts_base_unit": "clause",
            "rid": "complex",
        },
        {
            "rts_engines": [RTPlainSegmenter],
            "rts_aligner": None,
            "rts_label_fusion_type": "concat",
            "rts_base_unit": "sent",
            "rid": "sentence-baseline",
        },
    ]


def test_main_runs_segmentation_and_training_in_repository_data_directory(
    tmp_path,
) -> None:
    input_dataset = Dataset.from_dict({"reasoning_trace": ["A trace."]})
    sample_dataset = Dataset.from_dict(
        {"text": ["sample"], "labels": [1], "source_index": [0]}
    )
    dataset_path = tmp_path / "segmented"

    with (
        patch.object(
            downstream_firsterror,
            "create_rtseg_dataset_main",
            return_value=(input_dataset, dataset_path),
        ) as mocked_create,
        patch.object(
            downstream_firsterror,
            "load_from_disk",
            return_value=input_dataset,
        ) as mocked_load,
        patch.object(
            downstream_firsterror,
            "extract_first_error_samples",
            return_value=sample_dataset,
        ) as mocked_extract,
        patch.object(
            downstream_firsterror,
            "train_cross_validated_classifier",
        ) as mocked_train,
    ):
        result = downstream_firsterror.main(
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
    assert mocked_extract.call_args.kwargs["balance_classes"] is False
    assert mocked_extract.call_args.kwargs["include_rtseg_labels"] is True
    expected_output = (
        Path(downstream_firsterror.__file__).resolve().parents[1]
        / "data"
        / "first_error_modernbert_baseline"
    )
    assert mocked_train.call_args.kwargs["output_directory"] == expected_output
    assert mocked_train.call_args.kwargs["calibration_fraction"] == 0.2
    assert mocked_train.call_args.kwargs["use_class_weights"] is True
    assert mocked_train.call_args.kwargs["include_rtseg_labels"] is True
    assert result == {
        "rid": "baseline",
        "dataset_path": str(dataset_path),
        "output_directory": str(expected_output),
    }


def test_main_forwards_rtseg_label_ablation(tmp_path) -> None:
    input_dataset = Dataset.from_dict({"reasoning_trace": ["A trace."]})
    sample_dataset = Dataset.from_dict(
        {"text": ["sample"], "labels": [1], "source_index": [0]}
    )
    dataset_path = tmp_path / "segmented"

    with (
        patch.object(
            downstream_firsterror,
            "create_rtseg_dataset_main",
            return_value=(input_dataset, dataset_path),
        ),
        patch.object(
            downstream_firsterror,
            "load_from_disk",
            return_value=input_dataset,
        ),
        patch.object(
            downstream_firsterror,
            "extract_first_error_samples",
            return_value=sample_dataset,
        ) as mocked_extract,
        patch.object(
            downstream_firsterror,
            "train_cross_validated_classifier",
        ) as mocked_train,
    ):
        downstream_firsterror.main(
            rtseg_engines=[RTPlainSegmenter],
            rtseg_aligner=None,
            rtseg_label_fusion_type="concat",
            rtseg_base_unit="sent",
            rid="without-labels",
            reuse_existing_dataset=False,
            include_rtseg_labels=False,
        )

    assert mocked_extract.call_args.kwargs["include_rtseg_labels"] is False
    assert mocked_train.call_args.kwargs["include_rtseg_labels"] is False


def test_main_reuses_existing_dataset_for_requested_top_k(tmp_path) -> None:
    input_dataset = Dataset.from_dict({"reasoning_trace": ["A trace."]})
    sample_dataset = Dataset.from_dict(
        {"text": ["sample"], "labels": [1], "source_index": [0]}
    )
    dataset_path = tmp_path / "segmented"
    dataset_path.mkdir()

    with (
        patch.object(
            downstream_firsterror,
            "get_rtseg_dataset_path",
            return_value=dataset_path,
        ) as mocked_get_path,
        patch.object(
            downstream_firsterror,
            "create_rtseg_dataset_main",
        ) as mocked_create,
        patch.object(
            downstream_firsterror,
            "load_from_disk",
            return_value=input_dataset,
        ) as mocked_load,
        patch.object(
            downstream_firsterror,
            "extract_first_error_samples",
            return_value=sample_dataset,
        ),
        patch.object(
            downstream_firsterror,
            "train_cross_validated_classifier",
        ),
    ):
        result = downstream_firsterror.main(
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
    configs = _configs()

    def fake_main(**kwargs):
        return {
            "rid": kwargs["rid"],
            "dataset_path": f"dataset/{kwargs['rid']}",
            "output_directory": f"output/{kwargs['rid']}",
        }

    with patch.object(downstream_firsterror, "main", side_effect=fake_main) as mocked:
        results = downstream_firsterror.multi_main(
            configs,
            use_multiprocessing=False,
            rtseg_top_k=2500,
            balance_classes=True,
            include_rtseg_labels=False,
        )

    assert [result["rid"] for result in results] == [
        "complex",
        "sentence-baseline",
    ]
    assert [call.kwargs["rid"] for call in mocked.call_args_list] == [
        "complex",
        "sentence-baseline",
    ]
    assert all(
        call.kwargs["reuse_existing_dataset"] is True
        for call in mocked.call_args_list
    )
    assert all(
        call.kwargs["rtseg_top_k"] == 2500
        for call in mocked.call_args_list
    )
    assert all(
        call.kwargs["balance_classes"] is True
        for call in mocked.call_args_list
    )
    assert all(
        call.kwargs["include_rtseg_labels"] is False
        for call in mocked.call_args_list
    )


def test_multi_main_uses_spawned_processes_and_preserves_config_order() -> None:
    configs = _configs()
    submitted = []
    executor_arguments = {}
    spawn_context = object()

    class FakeFuture:
        def __init__(
            self,
            function,
            config,
            reuse_existing_dataset,
            rtseg_top_k,
            balance_classes,
            include_rtseg_labels,
        ):
            self.function = function
            self.config = config
            self.reuse_existing_dataset = reuse_existing_dataset
            self.rtseg_top_k = rtseg_top_k
            self.balance_classes = balance_classes
            self.include_rtseg_labels = include_rtseg_labels

        def result(self):
            return self.function(
                self.config,
                self.reuse_existing_dataset,
                self.rtseg_top_k,
                self.balance_classes,
                self.include_rtseg_labels,
            )

    class FakeExecutor:
        def __init__(self, **kwargs):
            executor_arguments.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(
            self,
            function,
            config,
            reuse_existing_dataset,
            rtseg_top_k,
            balance_classes,
            include_rtseg_labels,
        ):
            future = FakeFuture(
                function,
                config,
                reuse_existing_dataset,
                rtseg_top_k,
                balance_classes,
                include_rtseg_labels,
            )
            submitted.append(future)
            return future

    def fake_main(**kwargs):
        return {
            "rid": kwargs["rid"],
            "dataset_path": f"dataset/{kwargs['rid']}",
            "output_directory": f"output/{kwargs['rid']}",
        }

    with (
        patch.object(downstream_firsterror, "ProcessPoolExecutor", FakeExecutor),
        patch.object(
            downstream_firsterror.multiprocessing,
            "get_context",
            return_value=spawn_context,
        ) as mocked_context,
        patch.object(
            downstream_firsterror,
            "as_completed",
            side_effect=lambda futures: reversed(list(futures)),
        ),
        patch.object(
            downstream_firsterror,
            "main",
            side_effect=fake_main,
        ) as mocked_main,
    ):
        results = downstream_firsterror.multi_main(
            configs,
            use_multiprocessing=True,
            reuse_existing_dataset=False,
            rtseg_top_k=2500,
            balance_classes=True,
            include_rtseg_labels=False,
        )

    mocked_context.assert_called_once_with("spawn")
    assert executor_arguments == {
        "max_workers": len(configs),
        "mp_context": spawn_context,
    }
    assert len(submitted) == len(configs)
    assert [result["rid"] for result in results] == [
        "complex",
        "sentence-baseline",
    ]
    assert all(
        call.kwargs["reuse_existing_dataset"] is False
        for call in mocked_main.call_args_list
    )
    assert all(
        call.kwargs["rtseg_top_k"] == 2500
        for call in mocked_main.call_args_list
    )
    assert all(
        call.kwargs["balance_classes"] is True
        for call in mocked_main.call_args_list
    )
    assert all(
        call.kwargs["include_rtseg_labels"] is False
        for call in mocked_main.call_args_list
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda configs: configs + [{**configs[1], "rid": "complex"}], "Duplicate rid"),
        (lambda configs: configs + [{**configs[1], "rid": "../unsafe"}], "unsafe rid"),
        (
            lambda configs: configs
            + [{**configs[1], "rid": "same-segmentation"}],
            "duplicates another segmentation setup",
        ),
    ],
)
def test_parallel_config_validation_prevents_output_collisions(
    mutate,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        downstream_firsterror.multi_main(
            mutate(_configs()),
            use_multiprocessing=True,
        )


def test_multi_main_rejects_invalid_worker_count() -> None:
    with pytest.raises(ValueError, match="max_workers"):
        downstream_firsterror.multi_main(
            _configs(),
            use_multiprocessing=True,
            max_workers=0,
        )
