"""Train a first-error detector on resegmented PRM800K reasoning traces."""

import importlib.util
import json
import random
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, load_from_disk
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold


def _extract_step_texts(
    *,
    reasoning_trace: str,
    reasoning_steps: Sequence[Sequence[int]],
) -> list[str]:
    step_texts = []
    for offset in reasoning_steps:
        if len(offset) != 2:
            raise ValueError("Every reasoning step offset must contain two positions.")
        start, end = int(offset[0]), int(offset[1])
        if not 0 <= start <= end <= len(reasoning_trace):
            raise ValueError(f"Invalid reasoning step offset: [{start}, {end}].")
        step_texts.append(reasoning_trace[start:end])
    return step_texts


def _build_prefix_sequence(
    *,
    step_texts: Sequence[str],
    trace_labels: Sequence[str],
    target_index: int,
    step_token: str,
    trace_label_token: str,
) -> str:
    parts = []
    for step_text, trace_label in zip(
        step_texts[: target_index + 1],
        trace_labels[: target_index + 1],
        strict=True,
    ):
        parts.extend((step_token, step_text, trace_label_token, str(trace_label)))
    return " ".join(parts)


def extract_first_error_samples(
    *,
    dataset: Dataset,
    correct_per_error: int,
    seed: int,
    step_token: str,
    trace_label_token: str,
    correct_step_label: str,
    error_step_label: str,
    balance_classes: bool = False,
) -> Dataset:
    """Create error prefixes and select correct prefixes globally.

    Correct prefixes are selected round-robin across source traces up to the
    requested global ratio. Every available prefix is used at most once, so the
    result may contain fewer correct samples than requested when the dataset does
    not contain enough unique pre-error prefixes. When ``balance_classes`` is
    enabled, retain the largest possible equally sized sets of first-error and
    correct samples; ``correct_per_error`` is ignored for target sizing.
    """
    if correct_per_error <= 0:
        raise ValueError("correct_per_error must be greater than zero.")

    random_generator = random.Random(seed)
    error_samples = []
    correct_samples_by_trace = []

    for source_index, example in enumerate(dataset):
        reasoning_steps = example["reasoning_steps"]
        step_ratings = example["step_ratings"]
        trace_labels = example["rtseg_labels"]
        if not (
            len(reasoning_steps) == len(step_ratings) == len(trace_labels)
        ):
            raise ValueError(
                "reasoning_steps, step_ratings, and rtseg_labels must be aligned."
            )

        error_indices = [
            index for index, label in enumerate(step_ratings) if label == error_step_label
        ]
        if len(error_indices) != 1:
            raise ValueError("Every trace must contain exactly one first-error label.")
        error_index = error_indices[0]
        correct_indices = [
            index
            for index, label in enumerate(step_ratings[:error_index])
            if label == correct_step_label
        ]
        step_texts = _extract_step_texts(
            reasoning_trace=example["reasoning_trace"],
            reasoning_steps=reasoning_steps,
        )

        def make_sample(target_index: int, binary_label: int) -> dict[str, Any]:
            return {
                "text": _build_prefix_sequence(
                    step_texts=step_texts,
                    trace_labels=trace_labels,
                    target_index=target_index,
                    step_token=step_token,
                    trace_label_token=trace_label_token,
                ),
                "labels": binary_label,
                "source_index": source_index,
                "target_step_index": target_index,
                "target_type": "first_error" if binary_label else "correct",
            }

        error_samples.append(make_sample(error_index, 1))
        trace_correct_samples = [make_sample(index, 0) for index in correct_indices]
        random_generator.shuffle(trace_correct_samples)
        if trace_correct_samples:
            correct_samples_by_trace.append(trace_correct_samples)

    if not error_samples:
        raise ValueError("The dataset contains no first-error samples.")

    available_correct_count = sum(map(len, correct_samples_by_trace))
    if not available_correct_count:
        raise ValueError(
            "The dataset contains no correct prefixes before its first errors."
        )

    if balance_classes:
        samples_per_class = min(len(error_samples), available_correct_count)
        if samples_per_class < len(error_samples):
            random_generator.shuffle(error_samples)
            error_samples = error_samples[:samples_per_class]
        correct_target = samples_per_class
    else:
        correct_target = correct_per_error * len(error_samples)

    # Take one prefix per eligible trace per pass for the most balanced unique
    # allocation possible, only allowing longer traces to contribute more once
    # shorter traces have run out of candidates.
    selected_correct_samples = []
    maximum_correct_count = max(map(len, correct_samples_by_trace), default=0)
    for correct_sample_index in range(maximum_correct_count):
        for trace_samples in correct_samples_by_trace:
            if correct_sample_index < len(trace_samples):
                selected_correct_samples.append(trace_samples[correct_sample_index])
                if len(selected_correct_samples) == correct_target:
                    break
        if len(selected_correct_samples) == correct_target:
            break

    samples = error_samples + selected_correct_samples
    random_generator.shuffle(samples)

    sample_dataset = Dataset.from_list(samples)
    print(
        f"Prepared {len(sample_dataset)} samples from {len(dataset)} traces; "
        f"retained {len(error_samples)} first-error samples."
    )
    print(
        f"Correct samples: {sample_dataset['labels'].count(0)}, "
        f"first-error samples: {sample_dataset['labels'].count(1)}; "
        f"requested correct samples: {correct_target}; "
        f"balanced classes: {balance_classes}."
    )
    return sample_dataset


def _positive_class_probabilities(logits: Any) -> np.ndarray:
    logits_array = np.asarray(logits, dtype=np.float64)
    if logits_array.ndim != 2 or logits_array.shape[1] != 2:
        raise ValueError("Binary classification logits must have shape (n, 2).")
    shifted_logits = logits_array - np.max(logits_array, axis=1, keepdims=True)
    probabilities = np.exp(shifted_logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities[:, 1]


def _select_binary_f1_threshold(
    *,
    probabilities: Any,
    labels: Any,
) -> float:
    """Choose a positive-class threshold using calibration data only."""
    probability_array = np.asarray(probabilities, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    if probability_array.ndim != 1 or label_array.ndim != 1:
        raise ValueError("probabilities and labels must be one-dimensional.")
    if len(probability_array) != len(label_array) or not len(label_array):
        raise ValueError("probabilities and labels must have equal nonzero length.")
    if not np.isfinite(probability_array).all():
        raise ValueError("probabilities must contain only finite values.")
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise ValueError("probabilities must be between zero and one.")
    if set(np.unique(label_array)).difference({0, 1}):
        raise ValueError("labels must contain only zero and one.")

    precision, recall, thresholds = precision_recall_curve(
        label_array,
        probability_array,
    )
    if not len(thresholds):
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f_scores = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    best_f1 = float(np.max(f_scores))
    best_indices = np.flatnonzero(np.isclose(f_scores, best_f1))
    # A stable tie-break near 0.5 avoids needlessly extreme thresholds.
    best_index = int(
        best_indices[np.argmin(np.abs(thresholds[best_indices] - 0.5))]
    )
    return float(thresholds[best_index])


def _balanced_class_weights(labels: Any) -> list[float]:
    """Return sklearn-style balanced weights for binary cross-entropy."""
    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.ndim != 1 or not len(label_array):
        raise ValueError("labels must be a nonempty one-dimensional array.")
    counts = np.bincount(label_array, minlength=2)
    if len(counts) != 2 or np.any(counts == 0):
        raise ValueError("Both classes are required to compute class weights.")
    sample_count = len(label_array)
    return [float(sample_count / (2 * count)) for count in counts]


def _split_grouped_calibration(
    *,
    outer_train_indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    calibration_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an outer training fold into grouped fit and calibration sets."""
    if not 0.0 < calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be between zero and 0.5.")
    outer_train_indices = np.asarray(outer_train_indices, dtype=np.int64)
    local_labels = labels[outer_train_indices]
    local_groups = groups[outer_train_indices]
    n_splits = max(2, round(1.0 / calibration_fraction))
    if n_splits > len(np.unique(local_groups)):
        raise ValueError("Not enough training groups for calibration splitting.")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    candidates = []
    for fit_local, calibration_local in splitter.split(
        np.zeros(len(local_labels)),
        local_labels,
        local_groups,
    ):
        if (
            len(np.unique(local_labels[fit_local])) == 2
            and len(np.unique(local_labels[calibration_local])) == 2
        ):
            observed_fraction = len(calibration_local) / len(local_labels)
            candidates.append(
                (
                    abs(observed_fraction - calibration_fraction),
                    fit_local,
                    calibration_local,
                )
            )
    if not candidates:
        raise ValueError(
            "Could not create fit and calibration sets containing both classes."
        )
    _, fit_local, calibration_local = min(candidates, key=lambda item: item[0])
    return outer_train_indices[fit_local], outer_train_indices[calibration_local]


def _make_weighted_trainer_class(trainer_class: Any) -> type:
    class WeightedLossTrainer(trainer_class):
        def __init__(self, *args: Any, class_weights: Sequence[float], **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.class_weights = tuple(float(weight) for weight in class_weights)

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            import torch

            del num_items_in_batch
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            weights = torch.tensor(
                self.class_weights,
                dtype=torch.float32,
                device=outputs.logits.device,
            )
            loss = torch.nn.functional.cross_entropy(
                outputs.logits.float(),
                labels,
                weight=weights,
            )
            return (loss, outputs) if return_outputs else loss

    return WeightedLossTrainer


def _compute_metrics_at_threshold(
    *,
    logits: Any,
    labels: Any,
    threshold: float,
) -> dict[str, float]:
    probabilities = _positive_class_probabilities(logits)
    predictions = (probabilities >= threshold).astype(np.int64)
    labels = np.asarray(labels)
    precision, recall, f_scores, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_first_error": float(precision[1]),
        "recall_first_error": float(recall[1]),
        "f1_correct": float(f_scores[0]),
        "f1_first_error": float(f_scores[1]),
        "f1_macro": float(np.mean(f_scores)),
    }


def _compute_metrics(prediction: Any) -> dict[str, float]:
    return _compute_metrics_at_threshold(
        logits=prediction.predictions,
        labels=prediction.label_ids,
        threshold=0.5,
    )


def _compute_calibrated_metrics(prediction: Any) -> dict[str, float]:
    threshold = _select_binary_f1_threshold(
        probabilities=_positive_class_probabilities(prediction.predictions),
        labels=prediction.label_ids,
    )
    metrics = _compute_metrics_at_threshold(
        logits=prediction.predictions,
        labels=prediction.label_ids,
        threshold=threshold,
    )
    metrics["decision_threshold"] = threshold
    return metrics


def _add_and_validate_special_tokens(
    *,
    tokenizer: Any,
    special_tokens: Sequence[str],
) -> dict[str, Any]:
    vocabulary_before = tokenizer.get_vocab()
    vocabulary_size_before = len(tokenizer)
    expected_new_tokens = [
        token for token in special_tokens if token not in vocabulary_before
    ]
    added_token_count = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(special_tokens)}
    )
    vocabulary_after = tokenizer.get_vocab()

    if added_token_count != len(expected_new_tokens):
        raise RuntimeError(
            "Tokenizer reported an unexpected number of added special tokens: "
            f"expected {len(expected_new_tokens)}, got {added_token_count}."
        )
    if len(tokenizer) != vocabulary_size_before + added_token_count:
        raise RuntimeError("Tokenizer vocabulary size did not grow as expected.")

    token_ids = {}
    for token in special_tokens:
        if token not in vocabulary_after:
            raise RuntimeError(f"Special token {token!r} is missing from the vocabulary.")
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            raise RuntimeError(f"Special token {token!r} resolves to the unknown token.")
        if tokenizer.tokenize(token) != [token]:
            raise RuntimeError(f"Special token {token!r} is split by the tokenizer.")
        token_ids[token] = int(token_id)

    result = {
        "vocabulary_size_before": vocabulary_size_before,
        "vocabulary_size_after": len(tokenizer),
        "added_token_count": added_token_count,
        "token_ids": token_ids,
    }
    print(f"Tokenizer special-token validation: {result}")
    return result


def _resize_and_validate_model_embeddings(
    *,
    model: Any,
    tokenizer: Any,
    expected_label_count: int,
) -> dict[str, Any]:
    input_embeddings_before = model.get_input_embeddings()
    if input_embeddings_before is None:
        raise RuntimeError("The model does not expose an input embedding layer.")
    vocabulary_size_before = int(input_embeddings_before.weight.shape[0])

    model.resize_token_embeddings(len(tokenizer))
    input_embeddings_after = model.get_input_embeddings()
    vocabulary_size_after = int(input_embeddings_after.weight.shape[0])
    if vocabulary_size_after != len(tokenizer):
        raise RuntimeError(
            "Input embedding rows do not match the tokenizer vocabulary size."
        )
    if model.config.vocab_size != len(tokenizer):
        raise RuntimeError(
            "Model config vocab_size does not match the tokenizer vocabulary size."
        )
    if model.config.num_labels != expected_label_count:
        raise RuntimeError(
            f"Expected {expected_label_count} classification labels, got "
            f"{model.config.num_labels}."
        )
    classifier = getattr(model, "classifier", None)
    classifier_output_size = getattr(classifier, "out_features", None)
    if (
        classifier_output_size is not None
        and classifier_output_size != expected_label_count
    ):
        raise RuntimeError(
            "Classification head output size does not match the label count."
        )

    output_embeddings = model.get_output_embeddings()
    output_vocabulary_size = None
    if output_embeddings is not None:
        output_vocabulary_size = int(output_embeddings.weight.shape[0])
        if output_vocabulary_size != len(tokenizer):
            raise RuntimeError(
                "Output embedding rows do not match the tokenizer vocabulary size."
            )

    result = {
        "input_vocabulary_size_before": vocabulary_size_before,
        "input_vocabulary_size_after": vocabulary_size_after,
        "output_vocabulary_size_after": output_vocabulary_size,
        "config_vocabulary_size_after": int(model.config.vocab_size),
        "classification_label_count": int(model.config.num_labels),
        "classifier_output_size": classifier_output_size,
    }
    print(f"Model embedding validation: {result}")
    return result


def _validate_flash_attention_environment(
    *,
    attention_implementation: str,
    use_bf16: bool,
    use_fp16: bool,
) -> Any:
    import torch

    if attention_implementation != "flash_attention_2":
        raise ValueError("attention_implementation must be 'flash_attention_2'.")
    if use_bf16 == use_fp16:
        raise ValueError("Enable exactly one of BF16 or FP16 for FlashAttention-2.")
    if not torch.cuda.is_available():
        raise RuntimeError("FlashAttention-2 training requires a CUDA GPU.")
    if importlib.util.find_spec("flash_attn") is None:
        raise RuntimeError(
            "FlashAttention-2 is required but flash_attn is not installed. "
            "Install it with `pip install flash-attn --no-build-isolation`."
        )
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is enabled, but the current CUDA GPU lacks support.")
    return torch.bfloat16 if use_bf16 else torch.float16


def train_cross_validated_classifier(
    *,
    sample_dataset: Dataset,
    model_name: str,
    output_directory: str | Path,
    n_folds: int,
    seed: int,
    step_token: str,
    trace_label_token: str,
    max_length: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    num_train_epochs: float,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    use_bf16: bool,
    use_fp16: bool,
    attention_implementation: str,
    deterministic_flash_attention: bool,
    use_gradient_checkpointing: bool,
    calibration_fraction: float = 0.2,
    use_class_weights: bool = True,
) -> dict[str, Any]:
    """Fine-tune with nested grouped calibration inside grouped n-fold CV."""
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if n_folds < 2:
        raise ValueError("n_folds must be at least two.")
    unique_groups = len(set(sample_dataset["source_index"]))
    if n_folds > unique_groups:
        raise ValueError(
            f"n_folds ({n_folds}) exceeds the number of source traces "
            f"({unique_groups})."
        )
    model_dtype = _validate_flash_attention_environment(
        attention_implementation=attention_implementation,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
    )

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    prepared_dataset_path = output_path / "prepared_dataset"
    if prepared_dataset_path.exists():
        shutil.rmtree(prepared_dataset_path)
    sample_dataset.save_to_disk(prepared_dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_validation = _add_and_validate_special_tokens(
        tokenizer=tokenizer,
        special_tokens=[step_token, trace_label_token],
    )
    tokenizer.truncation_side = "left"
    model_config = AutoConfig.from_pretrained(model_name)
    context_limit = min(
        int(tokenizer.model_max_length),
        int(model_config.max_position_embeddings),
    )
    if not 0 < max_length <= context_limit:
        raise ValueError(
            f"max_length must be between 1 and the model limit ({context_limit})."
        )
    print(
        f"Sequences use dynamic per-batch padding with a maximum of "
        f"{max_length} tokens."
    )

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = sample_dataset.map(tokenize_batch, batched=True)
    splitter = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )
    labels = np.asarray(sample_dataset["labels"])
    groups = np.asarray(sample_dataset["source_index"])
    fold_metrics = []
    embedding_validations = []
    trainer_class = _make_weighted_trainer_class(Trainer)

    for fold_index, (outer_train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels, groups),
        start=1,
    ):
        fold_seed = seed + fold_index
        set_seed(fold_seed)
        train_indices, calibration_indices = _split_grouped_calibration(
            outer_train_indices=outer_train_indices,
            labels=labels,
            groups=groups,
            calibration_fraction=calibration_fraction,
            seed=fold_seed,
        )
        class_weights = (
            _balanced_class_weights(labels[train_indices])
            if use_class_weights
            else [1.0, 1.0]
        )
        fold_directory = output_path / f"fold_{fold_index}"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label={0: "correct", 1: "first_error"},
            label2id={"correct": 0, "first_error": 1},
            attn_implementation=attention_implementation,
            dtype=model_dtype,
            deterministic_flash_attn=deterministic_flash_attention,
        )
        configured_attention = getattr(
            model.config, "_attn_implementation", None
        )
        if configured_attention != attention_implementation:
            raise RuntimeError(
                "ModernBERT did not activate FlashAttention-2: "
                f"configured backend is {configured_attention!r}."
            )
        print(
            f"Fold {fold_index} attention backend: {configured_attention}; "
            f"dtype: {model.dtype}."
        )
        embedding_validations.append(
            _resize_and_validate_model_embeddings(
                model=model,
                tokenizer=tokenizer,
                expected_label_count=2,
            )
        )
        training_arguments = TrainingArguments(
            output_dir=str(fold_directory),
            overwrite_output_dir=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_first_error",
            greater_is_better=True,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            bf16=use_bf16,
            fp16=use_fp16,
            seed=fold_seed,
            data_seed=fold_seed,
            save_total_limit=1,
            report_to="none",
            full_determinism=deterministic_flash_attention,
            gradient_checkpointing=use_gradient_checkpointing,
            group_by_length=True,
        )
        trainer = trainer_class(
            model=model,
            args=training_arguments,
            train_dataset=tokenized_dataset.select(train_indices.tolist()),
            eval_dataset=tokenized_dataset.select(calibration_indices.tolist()),
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(
                tokenizer=tokenizer,
                padding="longest",
                pad_to_multiple_of=8,
            ),
            compute_metrics=_compute_calibrated_metrics,
            class_weights=class_weights,
        )
        trainer.train()
        calibration_output = trainer.predict(
            tokenized_dataset.select(calibration_indices.tolist())
        )
        calibration_probabilities = _positive_class_probabilities(
            calibration_output.predictions
        )
        decision_threshold = _select_binary_f1_threshold(
            probabilities=calibration_probabilities,
            labels=calibration_output.label_ids,
        )
        calibration_metrics = _compute_metrics_at_threshold(
            logits=calibration_output.predictions,
            labels=calibration_output.label_ids,
            threshold=decision_threshold,
        )
        validation_output = trainer.predict(
            tokenized_dataset.select(validation_indices.tolist())
        )
        metrics = _compute_metrics_at_threshold(
            logits=validation_output.predictions,
            labels=validation_output.label_ids,
            threshold=decision_threshold,
        )
        metrics["fold"] = fold_index
        metrics["outer_train_samples"] = len(outer_train_indices)
        metrics["train_samples"] = len(train_indices)
        metrics["calibration_samples"] = len(calibration_indices)
        metrics["validation_samples"] = len(validation_indices)
        metrics["decision_threshold"] = decision_threshold
        metrics["class_weights"] = class_weights
        metrics["calibration_metrics"] = calibration_metrics
        fold_metrics.append(metrics)
        trainer.model.config.decision_threshold = decision_threshold
        trainer.model.config.positive_label = "first_error"
        trainer.model.config.threshold_calibration_metric = "f1_first_error"
        best_model_directory = fold_directory / "best_model"
        trainer.save_model(best_model_directory)
        tokenizer.save_pretrained(best_model_directory)

    metric_names = [
        "accuracy",
        "precision_first_error",
        "recall_first_error",
        "f1_correct",
        "f1_first_error",
        "f1_macro",
    ]
    averages = {
        metric_name: {
            "mean": float(np.mean([fold[metric_name] for fold in fold_metrics])),
            "std": float(np.std([fold[metric_name] for fold in fold_metrics])),
        }
        for metric_name in metric_names
    }
    report = {
        "model_name": model_name,
        "n_folds": n_folds,
        "seed": seed,
        "sample_count": len(sample_dataset),
        "source_trace_count": unique_groups,
        "class_counts": {
            "correct": sample_dataset["labels"].count(0),
            "first_error": sample_dataset["labels"].count(1),
        },
        "tokenizer_validation": tokenizer_validation,
        "embedding_validations": embedding_validations,
        "training_config": {
            "max_length": max_length,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "num_train_epochs": num_train_epochs,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "bf16": use_bf16,
            "fp16": use_fp16,
            "attention_implementation": attention_implementation,
            "deterministic_flash_attention": deterministic_flash_attention,
            "gradient_checkpointing": use_gradient_checkpointing,
            "calibration_fraction": calibration_fraction,
            "calibration_metric": "f1_first_error",
            "use_class_weights": use_class_weights,
            "class_weight_method": "balanced",
            "dynamic_batch_padding": True,
            "step_token": step_token,
            "trace_label_token": trace_label_token,
        },
        "folds": fold_metrics,
        "averages": averages,
    }
    report_path = output_path / "cross_validation_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Cross-validation results:")
    for fold in fold_metrics:
        print(
            f"Fold {fold['fold']}: first-error F1={fold['f1_first_error']:.4f}, "
            f"macro F1={fold['f1_macro']:.4f}, "
            f"threshold={fold['decision_threshold']:.4f}"
        )
    print(
        "Average first-error F1: "
        f"{averages['f1_first_error']['mean']:.4f} +/- "
        f"{averages['f1_first_error']['std']:.4f}"
    )
    print(
        f"Average macro F1: {averages['f1_macro']['mean']:.4f} +/- "
        f"{averages['f1_macro']['std']:.4f}"
    )
    print(f"Saved metrics to: {report_path}")
    return report


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[2]
    dataset_path = (
        repository_root
        / "data"
        / "fe_rtseg_datasets"
        / "RTNewLine_RTRuleRegex_RTZeroShotSeqClassificationTA_concat_clause"
        # / "RTNewLineVerbose_concat_sent"
    )
    output_directory = repository_root / "data" / "first_error_modernbert_complex"
    correct_per_error = 3
    balance_classes = False
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
    )
