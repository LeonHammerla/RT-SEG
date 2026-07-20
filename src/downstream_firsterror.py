import multiprocessing
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk

from eval_utils.first_error_detect_prm800k import (
    extract_first_error_samples,
    train_cross_validated_classifier,
)
from eval_utils.prm800k_utils import (
    create_rtseg_dataset_main,
    get_rtseg_dataset_path,
)
from rt_segmentation import (
    OffsetFusionGraph,
    RTEmbeddingBasedSemanticShift,
    RTNewLine,
    RTRuleRegex,
    RTZeroShotSeqClassificationTA, RTPlainSegmenter, RTEntailmentBasedSegmentation, RTBERTopicSegmentation,
    OffsetFusionFuzzy,
)


def main(
    rtseg_engines,
    rtseg_aligner,
    rtseg_label_fusion_type,
    rtseg_base_unit,
    rid,
    *,
    rtseg_top_k: int = 1000,
    reuse_existing_dataset: bool = True,
    balance_classes: bool = False,
) -> dict[str, str]:
    dataset_path = get_rtseg_dataset_path(
        rtseg_engines=rtseg_engines,
        rtseg_label_fusion_type=rtseg_label_fusion_type,
        rtseg_base_unit=rtseg_base_unit,
        rtseg_top_k=rtseg_top_k,
    )
    if reuse_existing_dataset and dataset_path.exists():
        print(f"[{rid}] Reusing existing RT-SEG dataset: {dataset_path}", flush=True)
    else:
        _, dataset_path = create_rtseg_dataset_main(
            rtseg_engines=rtseg_engines,
            rtseg_aligner=rtseg_aligner,
            rtseg_label_fusion_type=rtseg_label_fusion_type,
            rtseg_base_unit=rtseg_base_unit,
            rtseg_top_k=rtseg_top_k,
            rtseg_seed=42,
        )

    repository_root = Path(__file__).resolve().parents[1]
    output_directory = repository_root / "data" / f"first_error_modernbert_{rid}"
    correct_per_error = 3
    random_seed = 42
    number_of_folds = 5
    model_name = "answerdotai/ModernBERT-base"
    maximum_sequence_length = 8192
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    number_of_epochs = 6.0
    train_batch_size = 2
    evaluation_batch_size = 8
    gradient_accumulation_steps = 4
    use_bf16 = True
    use_fp16 = False
    attention_implementation = "flash_attention_2"
    deterministic_flash_attention = True
    use_gradient_checkpointing = True
    calibration_fraction = 0.2
    use_class_weights = True
    step_special_token = "[STEP]"
    trace_label_special_token = "[LABEL]"
    correct_rating = "0"
    first_error_rating = "-1"

    input_dataset = load_from_disk(dataset_path)
    if not isinstance(input_dataset, Dataset):
        raise TypeError("dataset_path must contain a Hugging Face Dataset.")

    extracted_samples = extract_first_error_samples(
        dataset=input_dataset,
        correct_per_error=correct_per_error,
        seed=random_seed,
        step_token=step_special_token,
        trace_label_token=trace_label_special_token,
        correct_step_label=correct_rating,
        error_step_label=first_error_rating,
        balance_classes=balance_classes,
    )
    train_cross_validated_classifier(
        sample_dataset=extracted_samples,
        model_name=model_name,
        output_directory=output_directory,
        n_folds=number_of_folds,
        seed=random_seed,
        step_token=step_special_token,
        trace_label_token=trace_label_special_token,
        max_length=maximum_sequence_length,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        num_train_epochs=number_of_epochs,
        train_batch_size=train_batch_size,
        eval_batch_size=evaluation_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        attention_implementation=attention_implementation,
        deterministic_flash_attention=deterministic_flash_attention,
        use_gradient_checkpointing=use_gradient_checkpointing,
        calibration_fraction=calibration_fraction,
        use_class_weights=use_class_weights,
    )
    return {
        "rid": rid,
        "dataset_path": str(dataset_path),
        "output_directory": str(output_directory),
    }


def _segmentation_config_key(config: Mapping[str, Any]) -> tuple[Any, ...]:
    engines = tuple(
        f"{engine.__module__}.{engine.__qualname__}"
        for engine in config["rts_engines"]
    )
    aligner = config["rts_aligner"]
    aligner_name = (
        None if aligner is None else f"{aligner.__module__}.{aligner.__qualname__}"
    )
    return (
        engines,
        aligner_name,
        config["rts_label_fusion_type"],
        config["rts_base_unit"],
    )


def _validate_configs(
    configs: Sequence[Mapping[str, Any]],
    *,
    require_unique_segmentations: bool,
) -> list[dict[str, Any]]:
    required_keys = {
        "rts_engines",
        "rts_aligner",
        "rts_label_fusion_type",
        "rts_base_unit",
        "rid",
    }
    normalized_configs = []
    seen_rids = set()
    seen_segmentations = set()

    if not configs:
        raise ValueError("At least one RT-SEG configuration must be declared.")

    for index, raw_config in enumerate(configs):
        missing_keys = required_keys.difference(raw_config)
        if missing_keys:
            raise ValueError(
                f"Configuration {index} is missing keys: "
                f"{', '.join(sorted(missing_keys))}."
            )

        config = dict(raw_config)
        config["rts_engines"] = list(config["rts_engines"])
        if not config["rts_engines"]:
            raise ValueError(f"Configuration {index} must select at least one engine.")
        if len(config["rts_engines"]) > 1 and config["rts_aligner"] is None:
            raise ValueError(
                f"Configuration {index} requires an aligner for multiple engines."
            )
        if config["rts_label_fusion_type"] not in {"concat", "majority"}:
            raise ValueError(
                f"Configuration {index} has an invalid label fusion type."
            )
        if config["rts_base_unit"] not in {"clause", "sent"}:
            raise ValueError(f"Configuration {index} has an invalid base unit.")

        rid = config["rid"]
        if not isinstance(rid, str) or not rid.strip():
            raise ValueError(f"Configuration {index} must have a nonempty rid.")
        if Path(rid).name != rid or rid in {".", ".."}:
            raise ValueError(
                f"Configuration {index} has an unsafe rid {rid!r}; "
                "use a filename-like identifier."
            )
        if rid in seen_rids:
            raise ValueError(f"Duplicate rid {rid!r} would overwrite classifier output.")
        seen_rids.add(rid)

        segmentation_key = _segmentation_config_key(config)
        if require_unique_segmentations and segmentation_key in seen_segmentations:
            raise ValueError(
                f"Configuration {index} duplicates another segmentation setup "
                "and would overwrite its RT-SEG dataset."
            )
        seen_segmentations.add(segmentation_key)
        normalized_configs.append(config)

    return normalized_configs


def _run_config(
    config: Mapping[str, Any],
    reuse_existing_dataset: bool = True,
    rtseg_top_k: int = 1000,
    balance_classes: bool = False,
) -> dict[str, str]:
    rid = config["rid"]
    print(f"[{rid}] Starting downstream first-error run.", flush=True)
    result = main(
        rtseg_engines=config["rts_engines"],
        rtseg_aligner=config["rts_aligner"],
        rtseg_label_fusion_type=config["rts_label_fusion_type"],
        rtseg_base_unit=config["rts_base_unit"],
        rid=rid,
        rtseg_top_k=rtseg_top_k,
        reuse_existing_dataset=reuse_existing_dataset,
        balance_classes=balance_classes,
    )
    print(f"[{rid}] Completed downstream first-error run.", flush=True)
    return result


def multi_main(
    configs: Sequence[Mapping[str, Any]],
    *,
    use_multiprocessing: bool = True,
    max_workers: int | None = 8,
    reuse_existing_dataset: bool = True,
    rtseg_top_k: int = 1000,
    balance_classes: bool = False,
) -> list[dict[str, str]]:
    """Run every declared RT-SEG configuration sequentially or in parallel."""
    normalized_configs = _validate_configs(
        configs,
        require_unique_segmentations=use_multiprocessing,
    )

    if not use_multiprocessing:
        return [
            _run_config(
                config,
                reuse_existing_dataset,
                rtseg_top_k,
                balance_classes,
            )
            for config in normalized_configs
        ]

    worker_count = len(normalized_configs) if max_workers is None else max_workers
    if worker_count <= 0:
        raise ValueError("max_workers must be greater than zero.")
    worker_count = min(worker_count, len(normalized_configs))

    results_by_rid = {}
    first_exception = None
    spawn_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=spawn_context,
    ) as executor:
        future_to_rid = {
            executor.submit(
                _run_config,
                config,
                reuse_existing_dataset,
                rtseg_top_k,
                balance_classes,
            ): config["rid"]
            for config in normalized_configs
        }
        for future in as_completed(future_to_rid):
            rid = future_to_rid[future]
            try:
                results_by_rid[rid] = future.result()
            except Exception as exc:
                print(f"[{rid}] Failed: {exc}", flush=True)
                if first_exception is None:
                    first_exception = exc

    if first_exception is not None:
        raise RuntimeError(
            "At least one downstream first-error configuration failed."
        ) from first_exception

    # Preserve declaration order even though processes finish out of order.
    return [results_by_rid[config["rid"]] for config in normalized_configs]


if __name__ == "__main__":
    RTSEG_CONFIGS: list[dict[str, Any]] = [
        {
            "rts_engines": [
                RTPlainSegmenter
            ],
            "rts_aligner": None,
            "rts_label_fusion_type": "concat",
            "rts_base_unit": "sent",
            "rid": "sentbase",
        },
        {
            "rts_engines": [
                RTPlainSegmenter
            ],
            "rts_aligner": None,
            "rts_label_fusion_type": "concat",
            "rts_base_unit": "clause",
            "rid": "clausebase",
        },
        {
            "rts_engines": [
                RTNewLine,
                RTRuleRegex,
                RTZeroShotSeqClassificationTA,
            ],
            "rts_aligner": OffsetFusionGraph,
            "rts_label_fusion_type": "concat",
            "rts_base_unit": "clause",
            "rid": "complex1",
        },
        {
            "rts_engines": [
                RTNewLine,
                RTRuleRegex,
                RTZeroShotSeqClassificationTA,
                RTEmbeddingBasedSemanticShift,
            ],
            "rts_aligner": OffsetFusionGraph,
            "rts_label_fusion_type": "concat",
            "rts_base_unit": "clause",
            "rid": "complex2",
        }
    ]

    multi_main(
        configs=RTSEG_CONFIGS,
        use_multiprocessing=True,
        max_workers=4,
        rtseg_top_k=5000,
        balance_classes=True,
    )
