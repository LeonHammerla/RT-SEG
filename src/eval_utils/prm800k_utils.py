"""Loading and preprocessing utilities for the PRM800K dataset."""

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from pprint import pprint
from typing import Any, Literal, cast

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
    load_dataset,
)
from tqdm import tqdm
import sys

source_directory = Path(__file__).resolve().parents[1]
if str(source_directory) not in sys.path:
    sys.path.insert(0, str(source_directory))

from rt_segmentation import (
    OffsetFusionGraph,
    RTNewLine,
    RTNewLineVerbose,
    RTPlainSegmenter,
    RTRuleRegex,
    RTSeg,
    RTZeroShotSeqClassificationTA, RTEmbeddingBasedSemanticShift,
)


def _select_completion(
    *,
    step: Mapping[str, Any],
    error_rating: int,
) -> Mapping[str, Any]:
    """Select the first erroneous completion, or the first completion."""
    completions = step.get("completions")
    if not isinstance(completions, Sequence) or isinstance(completions, (str, bytes)):
        raise ValueError("A PRM800K step must contain a sequence of completions.")
    if not completions:
        raise ValueError("A PRM800K step contains no completions.")

    for completion in completions:
        if completion.get("rating") == error_rating:
            return completion
    return completions[0]


def _has_finish_reason(
    example: Mapping[str, Any],
    *,
    finish_reason: str,
) -> bool:
    label = example.get("label")
    return isinstance(label, Mapping) and label.get("finish_reason") == finish_reason


def _join_reasoning_steps(steps: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:
    offsets = []
    cursor = 0
    for index, step in enumerate(steps):
        if index > 0:
            cursor += 1
        start = cursor
        cursor += len(step)
        offsets.append((start, cursor))
    return " ".join(steps), offsets


def _preprocess_annotation(
    example: Mapping[str, Any],
    *,
    error_rating: int,
    unannotated_label: str,
) -> dict[str, Any]:
    question = example["question"]
    label = example["label"]
    selected_completions = [
        _select_completion(step=step, error_rating=error_rating)
        for step in label["steps"]
    ]
    ratings = [completion["rating"] for completion in selected_completions]
    pre_generated_steps = question.get("pre_generated_steps")
    if not isinstance(pre_generated_steps, Sequence) or isinstance(
        pre_generated_steps, (str, bytes)
    ):
        raise ValueError("A PRM800K question must contain pre_generated_steps.")
    if len(pre_generated_steps) < len(selected_completions):
        raise ValueError(
            "A PRM800K question has fewer pre-generated steps than annotations."
        )

    unannotated_steps = pre_generated_steps[len(selected_completions) :]
    reasoning_step_texts = [
        completion["text"] for completion in selected_completions
    ]
    reasoning_step_texts.extend(unannotated_steps)
    reasoning_trace, reasoning_steps = _join_reasoning_steps(reasoning_step_texts)
    step_ratings = [str(rating) for rating in ratings]
    step_ratings.extend([unannotated_label] * len(unannotated_steps))

    return {
        "problem": question["problem"],
        "ground_truth_solution": question.get("ground_truth_solution"),
        "ground_truth_answer": question.get("ground_truth_answer"),
        "reasoning_trace": reasoning_trace,
        "reasoning_steps": reasoning_steps,
        "step_ratings": step_ratings,
        "error_step_index": ratings.index(error_rating),
        "labeler": example.get("labeler"),
        "timestamp": example.get("timestamp"),
        "generation": example.get("generation"),
        "is_quality_control_question": example.get("is_quality_control_question"),
        "is_initial_screening_question": example.get("is_initial_screening_question"),
    }


def _preprocess_split(
    *,
    dataset: Dataset | IterableDataset,
    raw_columns: Sequence[str],
    finish_reason: str,
    error_rating: int,
    unannotated_label: str,
) -> Dataset | IterableDataset:
    filtered = dataset.filter(
        _has_finish_reason,
        fn_kwargs={"finish_reason": finish_reason},
    )
    return filtered.map(
        _preprocess_annotation,
        fn_kwargs={
            "error_rating": error_rating,
            "unannotated_label": unannotated_label,
        },
        remove_columns=list(raw_columns),
    )


def load_prm800k(
    *,
    dataset_name: str,
    data_files: Mapping[str, str],
    raw_columns: Sequence[str],
    finish_reason: str,
    error_rating: int,
    unannotated_label: str,
    split: Literal["train", "test"] | None,
    streaming: bool,
) -> Dataset | IterableDataset:
    """Load PRM800K and select one completion for every reasoning step."""
    raw_dataset = load_dataset(
        dataset_name,
        data_files=dict(data_files),
        split=split,
        streaming=streaming,
    )

    if split is not None:
        return _preprocess_split(
            dataset=cast(Dataset | IterableDataset, raw_dataset),
            raw_columns=raw_columns,
            finish_reason=finish_reason,
            error_rating=error_rating,
            unannotated_label=unannotated_label,
        )

    dataset_dict = cast(DatasetDict | IterableDatasetDict, raw_dataset)
    return concatenate_datasets(
        [
            _preprocess_split(
                dataset=dataset,
                raw_columns=raw_columns,
                finish_reason=finish_reason,
                error_rating=error_rating,
                unannotated_label=unannotated_label,
            )
            for dataset in dataset_dict.values()
        ]
    )


def _get_rtseg_config(*, rtseg: Any) -> str:
    engine_names = "_".join(engine.__name__ for engine in rtseg.engines)
    return f"{engine_names}_{rtseg.label_fusion_type}_{rtseg.seg_base_unit}"


def _offsets_overlap(
    *,
    first: Sequence[int],
    second: Sequence[int],
) -> bool:
    """Return whether two half-open offset intervals overlap."""
    return first[0] < second[1] and second[0] < first[1]


def _label_resegmented_steps(
    *,
    new_offsets: Sequence[Sequence[int]],
    old_error_offset: Sequence[int],
    reasoning_trace: str,
    correct_label: str,
    error_label: str,
    unannotated_label: str,
) -> tuple[list[list[int]], list[str], int]:
    if len(old_error_offset) != 2:
        raise ValueError("The old error offset must contain exactly two positions.")
    old_error_start, old_error_end = old_error_offset
    trace_length = len(reasoning_trace)
    if not 0 <= old_error_start < old_error_end <= trace_length:
        raise ValueError("The old error offset is invalid for the reasoning trace.")
    if not new_offsets:
        raise ValueError("RT-SEG must return at least one segment offset.")
    normalized_offsets = []
    new_error_index = None

    for index, offset in enumerate(new_offsets):
        if len(offset) != 2:
            raise ValueError("Every RT-SEG offset must contain exactly two positions.")
        start, end = int(offset[0]), int(offset[1])
        if not 0 <= start < end <= trace_length:
            raise ValueError(f"Invalid RT-SEG offset: [{start}, {end}].")
        expected_start = normalized_offsets[-1][1] if normalized_offsets else 0
        if start != expected_start:
            raise ValueError(
                "RT-SEG offsets must be contiguous and start at zero: "
                f"expected segment {index} to start at {expected_start}, got {start}."
            )
        if not reasoning_trace[start:end].strip():
            raise ValueError(
                f"RT-SEG segment {index} contains only whitespace."
            )
        normalized_offsets.append([start, end])
        if new_error_index is None and _offsets_overlap(
            first=(start, end),
            second=(old_error_start, old_error_end),
        ):
            new_error_index = index

    if normalized_offsets[-1][1] != trace_length:
        raise ValueError(
            "RT-SEG offsets must cover the complete reasoning trace: "
            f"expected final end {trace_length}, got {normalized_offsets[-1][1]}."
        )

    if new_error_index is None:
        raise ValueError("No RT-SEG segment overlaps the original error segment.")

    labels = [correct_label] * new_error_index
    labels.append(error_label)
    labels.extend(
        [unannotated_label] * (len(normalized_offsets) - new_error_index - 1)
    )
    return normalized_offsets, labels, new_error_index


def create_rtseg_dataset(
    *,
    base_dataset: Dataset,
    rtseg: Any,
    seed: int,
    top_k: int,
    output_root: str | Path,
    correct_label: str,
    error_label: str,
    unannotated_label: str,
) -> tuple[Dataset, Path]:
    """Shuffle, sample, resegment, relabel, and store a PRM800K dataset."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if top_k > len(base_dataset):
        raise ValueError(
            f"top_k ({top_k}) exceeds the dataset size ({len(base_dataset)})."
        )

    sampled_dataset = base_dataset.shuffle(seed=seed).select(range(top_k))
    rtseg_config = _get_rtseg_config(rtseg=rtseg)
    processed_examples = []

    for example in tqdm(sampled_dataset, desc=f"Segmenting with {rtseg_config}"):
        reasoning_trace = example["reasoning_trace"]
        old_error_offset = example["reasoning_steps"][example["error_step_index"]]
        new_offsets, rtseg_labels = rtseg(
            trace=reasoning_trace,
            problem=example["problem"],
        )
        if not isinstance(rtseg_labels, Sequence) or isinstance(
            rtseg_labels, (str, bytes)
        ):
            raise ValueError("RT-SEG labels must be a sequence.")
        if len(rtseg_labels) != len(new_offsets):
            raise ValueError(
                "RT-SEG must return exactly one label for every segment offset."
            )
        normalized_offsets, labels, new_error_index = _label_resegmented_steps(
            new_offsets=new_offsets,
            old_error_offset=old_error_offset,
            reasoning_trace=reasoning_trace,
            correct_label=correct_label,
            error_label=error_label,
            unannotated_label=unannotated_label,
        )

        processed_example = dict(example)
        processed_example["reasoning_steps"] = normalized_offsets
        processed_example["step_ratings"] = labels
        processed_example["rtseg_labels"] = list(rtseg_labels)
        processed_example["error_step_index"] = new_error_index
        processed_example["rtseg_config"] = rtseg_config
        processed_example["sampling_seed"] = seed
        processed_examples.append(processed_example)

    processed_dataset = Dataset.from_list(processed_examples)
    output_directory = Path(output_root) / rtseg_config
    if output_directory.exists():
        shutil.rmtree(output_directory)
    processed_dataset.save_to_disk(output_directory)

    print(f"Saved RT-SEG dataset to: {output_directory}")
    print(processed_dataset)
    print("Example:")
    pprint(processed_dataset[0])
    return processed_dataset, output_directory


def load_base_dataset():
    prm800k_dataset_name = "tasksource/PRM800K"
    prm800k_data_files = {
        "train": "phase2_train.jsonl",
        "test": "phase2_test.jsonl",
    }
    prm800k_raw_columns = [
        "labeler",
        "timestamp",
        "generation",
        "is_quality_control_question",
        "is_initial_screening_question",
        "question",
        "label",
    ]
    prm800k_finish_reason = "found_error"
    prm800k_error_rating = -1
    prm800k_unannotated_label = "x"
    prm800k_split = None
    prm800k_streaming = False
    output_directory = Path(__file__).resolve().parents[2] / "data" / "fe_dataset"

    processed_dataset = load_prm800k(
        dataset_name=prm800k_dataset_name,
        data_files=prm800k_data_files,
        raw_columns=prm800k_raw_columns,
        finish_reason=prm800k_finish_reason,
        error_rating=prm800k_error_rating,
        unannotated_label=prm800k_unannotated_label,
        split=prm800k_split,
        streaming=prm800k_streaming,
    )
    if output_directory.exists():
        shutil.rmtree(output_directory)
    processed_dataset.save_to_disk(output_directory)

    print(f"Saved processed PRM800K dataset to: {output_directory}")
    print(processed_dataset)
    print(f"Rows: {processed_dataset.num_rows}")
    print("Features:")
    pprint(processed_dataset.features)
    print("Example:")
    pprint(processed_dataset[0])
    print("Example:")
    pprint(processed_dataset[1])


def create_rtseg_dataset_main(
        rtseg_engines,
        rtseg_aligner,
        rtseg_label_fusion_type,
        rtseg_base_unit,
        rtseg_top_k,
        rtseg_seed
    ):
    processed_dataset = Dataset.load_from_disk(Path(__file__).resolve().parents[2] / "data" / "fe_dataset")
    rtseg_correct_label = "0"
    rtseg_error_label = "-1"
    rtseg_unannotated_label = "x"
    rtseg_output_root = (
            Path(__file__).resolve().parents[2] / "data" / "fe_rtseg_datasets"
    )

    segmenter = RTSeg(
        engines=rtseg_engines,
        aligner=rtseg_aligner,
        label_fusion_type=rtseg_label_fusion_type,
        seg_base_unit=rtseg_base_unit,
    )
    res, dpath = create_rtseg_dataset(
        base_dataset=processed_dataset,
        rtseg=segmenter,
        seed=rtseg_seed,
        top_k=rtseg_top_k,
        output_root=rtseg_output_root,
        correct_label=rtseg_correct_label,
        error_label=rtseg_error_label,
        unannotated_label=rtseg_unannotated_label,
    )
    return res, dpath



def create_base():
    rts_engines = [RTPlainSegmenter]
    rts_aligner = None
    rts_label_fusion_type = "concat"
    rts_base_unit = "sent"

    create_rtseg_dataset_main(
        rtseg_engines=rts_engines,
        rtseg_aligner=rts_aligner,
        rtseg_label_fusion_type=rts_label_fusion_type,
        rtseg_base_unit=rts_base_unit,
        rtseg_top_k=1000,
        rtseg_seed=42
    )

def create_complex1():
    rts_engines = [RTNewLine, RTRuleRegex, RTZeroShotSeqClassificationTA]
    rts_aligner = OffsetFusionGraph
    rts_label_fusion_type = "concat"
    rts_base_unit = "clause"

    create_rtseg_dataset_main(
        rtseg_engines=rts_engines,
        rtseg_aligner=rts_aligner,
        rtseg_label_fusion_type=rts_label_fusion_type,
        rtseg_base_unit=rts_base_unit,
        rtseg_top_k=1000,
        rtseg_seed=42
    )


def create_complex2():
    rts_engines = [RTNewLine, RTRuleRegex, RTZeroShotSeqClassificationTA, RTEmbeddingBasedSemanticShift]
    rts_aligner = OffsetFusionGraph
    rts_label_fusion_type = "concat"
    rts_base_unit = "clause"

    create_rtseg_dataset_main(
        rtseg_engines=rts_engines,
        rtseg_aligner=rts_aligner,
        rtseg_label_fusion_type=rts_label_fusion_type,
        rtseg_base_unit=rts_base_unit,
        rtseg_top_k=1000,
        rtseg_seed=42
    )

if __name__ == "__main__":
    create_complex2()
