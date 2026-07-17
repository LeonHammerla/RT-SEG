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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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
) -> Dataset:
    """Create one error prefix per trace and globally balance correct prefixes.

    Correct prefixes are selected round-robin across source traces up to the
    requested global ratio. Every available prefix is used at most once, so the
    result may contain fewer correct samples than requested when the dataset does
    not contain enough unique pre-error prefixes.
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

    correct_target = correct_per_error * len(error_samples)
    if correct_target and not correct_samples_by_trace:
        raise ValueError(
            "The dataset contains no correct prefixes before its first errors."
        )

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
        "retained one first-error sample per trace."
    )
    print(
        f"Correct samples: {sample_dataset['labels'].count(0)}, "
        f"first-error samples: {sample_dataset['labels'].count(1)}; "
        f"requested correct samples: {correct_target}."
    )
    return sample_dataset


def _compute_metrics(prediction: Any) -> dict[str, float]:
    predictions = np.argmax(prediction.predictions, axis=-1)
    labels = prediction.label_ids
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
) -> dict[str, Any]:
    """Fine-tune a binary sequence classifier using grouped n-fold CV."""
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

    for fold_index, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels, groups),
        start=1,
    ):
        fold_seed = seed + fold_index
        set_seed(fold_seed)
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
        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=tokenized_dataset.select(train_indices.tolist()),
            eval_dataset=tokenized_dataset.select(validation_indices.tolist()),
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(
                tokenizer=tokenizer,
                padding="longest",
                pad_to_multiple_of=8,
            ),
            compute_metrics=_compute_metrics,
        )
        trainer.train()
        raw_metrics = trainer.evaluate()
        metrics = {
            key.removeprefix("eval_"): float(value)
            for key, value in raw_metrics.items()
            if key.startswith("eval_")
            and key
            not in {
                "eval_loss",
                "eval_runtime",
                "eval_samples_per_second",
                "eval_steps_per_second",
            }
        }
        metrics["fold"] = fold_index
        metrics["train_samples"] = len(train_indices)
        metrics["validation_samples"] = len(validation_indices)
        fold_metrics.append(metrics)
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
            f"macro F1={fold['f1_macro']:.4f}"
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
