"""Train a classifier that predicts whether two PRM800K steps are consecutive.

Positive examples are the adjacent, ordered step pairs in each reasoning trace.
Negative examples are sampled from ordered, non-consecutive pairs in the same
trace.  Keeping both members of a pair in one trace makes the negatives harder
than pairs drawn from unrelated math problems and makes grouped cross-validation
leakage-free.
"""

import json
import random
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

if __package__:
    from .first_error_detect_prm800k import (
        _add_and_validate_special_tokens,
        _balanced_class_weights,
        _extract_step_texts,
        _make_weighted_trainer_class,
        _positive_class_probabilities,
        _resize_and_validate_model_embeddings,
        _select_binary_f1_threshold,
        _split_grouped_calibration,
        _validate_flash_attention_environment,
    )
else:
    from first_error_detect_prm800k import (
        _add_and_validate_special_tokens,
        _balanced_class_weights,
        _extract_step_texts,
        _make_weighted_trainer_class,
        _positive_class_probabilities,
        _resize_and_validate_model_embeddings,
        _select_binary_f1_threshold,
        _split_grouped_calibration,
        _validate_flash_attention_environment,
    )


def extract_reasoning_step_pair_samples(
    *,
    dataset: Dataset,
    fake_per_real: int,
    seed: int,
) -> Dataset:
    """Create adjacent real pairs and same-trace, non-consecutive fake pairs.

    Every adjacent pair is retained, including pairs containing correct, error,
    or unannotated steps.  Fake pairs are unique within each source trace.  When
    ``fake_per_real`` requests more unique fake pairs than a trace can provide,
    all available fake pairs are retained instead of duplicating examples.
    """
    if fake_per_real <= 0:
        raise ValueError("fake_per_real must be greater than zero.")

    random_generator = random.Random(seed)
    samples: list[dict[str, Any]] = []
    skipped_short_traces = 0
    requested_fake_count = 0
    retained_fake_count = 0

    for source_index, example in enumerate(dataset):
        reasoning_steps = example["reasoning_steps"]
        step_ratings = example["step_ratings"]
        rtseg_labels = example["rtseg_labels"]
        if not (
            len(reasoning_steps) == len(step_ratings) == len(rtseg_labels)
        ):
            raise ValueError(
                "reasoning_steps, step_ratings, and rtseg_labels must be aligned."
            )
        if len(reasoning_steps) < 2:
            skipped_short_traces += 1
            continue

        step_texts = _extract_step_texts(
            reasoning_trace=example["reasoning_trace"],
            reasoning_steps=reasoning_steps,
        )

        def make_sample(
            first_index: int,
            second_index: int,
            binary_label: int,
        ) -> dict[str, Any]:
            return {
                "premise": step_texts[first_index],
                "hypothesis": step_texts[second_index],
                "labels": binary_label,
                "source_index": source_index,
                "premise_step_index": first_index,
                "hypothesis_step_index": second_index,
                "pair_type": "following" if binary_label else "not_following",
                "premise_rtseg_label": str(rtseg_labels[first_index]),
                "hypothesis_rtseg_label": str(rtseg_labels[second_index]),
                "premise_step_rating": str(step_ratings[first_index]),
                "hypothesis_step_rating": str(step_ratings[second_index]),
            }

        real_pairs = [
            make_sample(index, index + 1, 1)
            for index in range(len(step_texts) - 1)
        ]
        samples.extend(real_pairs)

        # A fake pair may be reversed or skip one or more steps, but it may not
        # repeat a step or accidentally be an adjacent pair in the true order.
        fake_indices = [
            (first_index, second_index)
            for first_index in range(len(step_texts))
            for second_index in range(len(step_texts))
            if first_index != second_index and second_index != first_index + 1
        ]
        random_generator.shuffle(fake_indices)
        trace_fake_target = fake_per_real * len(real_pairs)
        requested_fake_count += trace_fake_target
        selected_fake_indices = fake_indices[:trace_fake_target]
        retained_fake_count += len(selected_fake_indices)
        samples.extend(
            make_sample(first_index, second_index, 0)
            for first_index, second_index in selected_fake_indices
        )

    if not samples:
        raise ValueError("The dataset contains no traces with at least two steps.")
    if retained_fake_count == 0:
        raise ValueError("The dataset contains no valid fake step pairs.")

    random_generator.shuffle(samples)
    sample_dataset = Dataset.from_list(samples)
    print(
        f"Prepared {len(sample_dataset)} pairs from {len(dataset)} traces; "
        f"skipped {skipped_short_traces} traces with fewer than two steps."
    )
    print(
        f"Following pairs: {sample_dataset['labels'].count(1)}, "
        f"not-following pairs: {sample_dataset['labels'].count(0)}; "
        f"requested not-following pairs: {requested_fake_count}."
    )
    return sample_dataset


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
        "precision_following": float(precision[1]),
        "recall_following": float(recall[1]),
        "f1_not_following": float(f_scores[0]),
        "f1_following": float(f_scores[1]),
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


def _format_step_for_model(
    *,
    step_text: str,
    rtseg_label: str,
    include_rtseg_labels: bool,
    step_token: str,
    rtseg_label_token: str,
) -> str:
    if include_rtseg_labels:
        return (
            f"{step_token} {step_text} "
            f"{rtseg_label_token} {rtseg_label}"
        )
    return f"{step_token} {step_text}"


def train_cross_validated_classifier(
    *,
    sample_dataset: Dataset,
    model_name: str,
    output_directory: str | Path,
    n_folds: int,
    seed: int,
    include_rtseg_labels: bool,
    step_token: str,
    rtseg_label_token: str,
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
    special_tokens = [step_token]
    if include_rtseg_labels:
        special_tokens.append(rtseg_label_token)
    tokenizer_validation = _add_and_validate_special_tokens(
        tokenizer=tokenizer,
        special_tokens=special_tokens,
    )
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
        "Step pairs use the tokenizer's native sequence-pair separators, "
        f"dynamic per-batch padding, and a maximum of {max_length} tokens."
    )

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        premises = [
            _format_step_for_model(
                step_text=step_text,
                rtseg_label=rtseg_label,
                include_rtseg_labels=include_rtseg_labels,
                step_token=step_token,
                rtseg_label_token=rtseg_label_token,
            )
            for step_text, rtseg_label in zip(
                batch["premise"], batch["premise_rtseg_label"], strict=True
            )
        ]
        hypotheses = [
            _format_step_for_model(
                step_text=step_text,
                rtseg_label=rtseg_label,
                include_rtseg_labels=include_rtseg_labels,
                step_token=step_token,
                rtseg_label_token=rtseg_label_token,
            )
            for step_text, rtseg_label in zip(
                batch["hypothesis"], batch["hypothesis_rtseg_label"], strict=True
            )
        ]
        return tokenizer(
            premises,
            hypotheses,
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
            id2label={0: "not_following", 1: "following"},
            label2id={"not_following": 0, "following": 1},
            attn_implementation=attention_implementation,
            dtype=model_dtype,
            deterministic_flash_attn=deterministic_flash_attention,
        )
        configured_attention = getattr(model.config, "_attn_implementation", None)
        if configured_attention != attention_implementation:
            raise RuntimeError(
                f"The model did not activate {attention_implementation!r}: "
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
            metric_for_best_model="f1_following",
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
        trainer.model.config.positive_label = "following"
        trainer.model.config.threshold_calibration_metric = "f1_following"
        best_model_directory = fold_directory / "best_model"
        trainer.save_model(best_model_directory)
        tokenizer.save_pretrained(best_model_directory)

    metric_names = [
        "accuracy",
        "precision_following",
        "recall_following",
        "f1_not_following",
        "f1_following",
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
            "not_following": sample_dataset["labels"].count(0),
            "following": sample_dataset["labels"].count(1),
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
            "calibration_metric": "f1_following",
            "use_class_weights": use_class_weights,
            "class_weight_method": "balanced",
            "dynamic_batch_padding": True,
            "include_rtseg_labels": include_rtseg_labels,
            "step_token": step_token,
            "rtseg_label_token": rtseg_label_token,
        },
        "folds": fold_metrics,
        "averages": averages,
    }
    report_path = output_path / "cross_validation_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Cross-validation results:")
    for fold in fold_metrics:
        print(
            f"Fold {fold['fold']}: following F1={fold['f1_following']:.4f}, "
            f"macro F1={fold['f1_macro']:.4f}, "
            f"threshold={fold['decision_threshold']:.4f}"
        )
    print(
        f"Average following F1: {averages['f1_following']['mean']:.4f} +/- "
        f"{averages['f1_following']['std']:.4f}"
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
    )
    output_directory = repository_root / "data" / "reasoning_step_nli_modernbert"
    fake_per_real = 1
    random_seed = 42
    number_of_folds = 5
    model_name = "answerdotai/ModernBERT-base"
    maximum_sequence_length = 512
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    number_of_epochs = 6.0
    train_batch_size = 8
    evaluation_batch_size = 16
    gradient_accumulation_steps = 2
    use_bf16 = True
    use_fp16 = False
    attention_implementation = "flash_attention_2"
    deterministic_flash_attention = True
    use_gradient_checkpointing = True
    include_rtseg_labels = True
    step_special_token = "[STEP]"
    rtseg_label_special_token = "[RTSEG_LABEL]"

    input_dataset = load_from_disk(dataset_path)
    if not isinstance(input_dataset, Dataset):
        raise TypeError("dataset_path must contain a Hugging Face Dataset.")

    extracted_samples = extract_reasoning_step_pair_samples(
        dataset=input_dataset,
        fake_per_real=fake_per_real,
        seed=random_seed,
    )
    train_cross_validated_classifier(
        sample_dataset=extracted_samples,
        model_name=model_name,
        output_directory=output_directory,
        n_folds=number_of_folds,
        seed=random_seed,
        include_rtseg_labels=include_rtseg_labels,
        step_token=step_special_token,
        rtseg_label_token=rtseg_label_special_token,
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
