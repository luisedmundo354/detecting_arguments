"""Span classification routines restricted to Linear SVC baselines."""
from __future__ import annotations

import os
from numbers import Integral
from pathlib import Path
from typing import Dict, Optional, Sequence

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

from helpers import (
    SENTENCE_TRANSFORMER_MODELS,
    assign_stratified_folds,
    build_fold_audit,
    build_tfidf_vectorizer,
    encode_sentences,
    filter_implicit_conclusions as filter_implicit_conclusions_df,
    load_annotation_spans,
)
from gpt5_classifier import run_gpt5_classification

if TYPE_CHECKING:  # pragma: no cover
    from openai import OpenAI
    from .gpt5_classifier import GPT5Config

DEFAULT_EMBEDDINGS: Sequence[str] = ("tfidf", "sbert", "legal-bert", "modern-bert")


def _merge_analysis_and_conclusion_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where any Conclusion labels are remapped to Analysis."""

    def _normalize_label(value: object) -> object:
        if isinstance(value, str) and value.strip().lower() == "conclusion":
            return "Analysis"
        return value

    updated = df.copy()
    updated["label"] = updated["label"].apply(_normalize_label)
    return updated


def run_linear_svc_classification(
    annotation_dir: Path | str,
    *,
    embeddings: Sequence[str] = DEFAULT_EMBEDDINGS,
    n_splits: int = 5,
    random_state: int = 42,
    batch_size: int = 16,
    model_overrides: Optional[Dict[str, str]] = None,
    verbose: bool = True,
    include_baselines: bool = True,
    gpt5: bool = False,
    gpt5_config: "GPT5Config | None" = None,
    gpt5_client: "OpenAI | None" = None,
    use_modern_bert: bool = False,
    test_mode: bool = False,
    gpt5_checkpoint_path: Optional[Path | str] = None,
    gpt5_workers: int = 1,
    combine_analysis_conclusion: bool = False,
    filter_implicit_conclusions: bool = False,
) -> Dict[str, object]:
    """Execute strict grouped k-fold evaluation with a Linear SVC back-end.

    Parameters
    ----------
    annotation_dir:
        Directory containing Label Studio JSON exports.
    embeddings:
        Iterable of embedding identifiers; supported values are ``tfidf``,
        ``sbert``, ``legal-bert``, and ``modern-bert``.
    n_splits:
        Number of folds for cross-validation. The value must be feasible for the
        grouped label distribution or the function raises a ``ValueError``.
    random_state:
        Seed forwarded to the fold splitter and the Linear SVC estimator.
    batch_size:
        Mini-batch size used by SentenceTransformer encoders.
    model_overrides:
        Optional mapping to replace the default SentenceTransformer checkpoints
        on a per-embedding basis.
    verbose:
        When ``True`` the function prints a scikit-learn style report per
        embedding along with baseline summaries.
    include_baselines:
        If ``True``, compute and display random and majority-class baselines.
    combine_analysis_conclusion:
        When ``True`` treat Conclusion spans as Analysis for training, evaluation,
        and downstream GPT-5 prompting.
    filter_implicit_conclusions:
        When ``True`` drop spans whose text corresponds to implicit intermediate
        conclusion placeholders before training.
    use_modern_bert:
        Toggle Modern-BERT embeddings. When ``False`` the embedding is skipped
        without attempting to reach SageMaker.
    gpt5_checkpoint_path:
        Optional JSONL path used to persist GPT-5 predictions between runs. Ignored
        in test mode.
    gpt5_workers:
        Number of concurrent GPT-5 workers. Values greater than one only affect
        the GPT-5 branch.

    Returns
    -------
    Dict[str, object]
        Reports, prediction tables, fold audit metadata, and run configuration.
    """

    _validate_run_inputs(
        annotation_dir=annotation_dir,
        embeddings=embeddings,
        n_splits=n_splits,
        random_state=random_state,
        batch_size=batch_size,
        model_overrides=model_overrides,
        verbose=verbose,
        include_baselines=include_baselines,
        gpt5=gpt5,
        gpt5_config=gpt5_config,
        gpt5_client=gpt5_client,
        use_modern_bert=use_modern_bert,
        test_mode=test_mode,
        gpt5_checkpoint_path=gpt5_checkpoint_path,
        gpt5_workers=gpt5_workers,
        combine_analysis_conclusion=combine_analysis_conclusion,
        filter_implicit_conclusions=filter_implicit_conclusions,
    )

    if test_mode:
        verbose = True

    source_dir = Path(annotation_dir).expanduser().resolve()
    df = load_annotation_spans(source_dir)
    if filter_implicit_conclusions:
        df = filter_implicit_conclusions_df(df)
    if combine_analysis_conclusion:
        df = _merge_analysis_and_conclusion_labels(df)
    df = assign_stratified_folds(df, n_splits=n_splits, random_state=random_state)
    fold_audit = build_fold_audit(df)

    labels_sorted = sorted(df["label"].unique())
    fold_ids = sorted(df["fold"].unique())

    predictions_frame = df[["span_id", "document_id", "start", "end", "text", "label", "fold"]].copy()
    reports: Dict[str, Dict[str, object]] = {}

    for embedding in embeddings:
        key = embedding.lower()
        if key not in DEFAULT_EMBEDDINGS:
            raise ValueError(f"Unsupported embedding '{embedding}'.")

        if key == "modern-bert" and not use_modern_bert:
            if verbose:
                print("Skipping modern-bert embedding (set use_modern_bert=True to enable).")
            continue

        fold_predictions = np.empty(len(df), dtype=object)

        if key == "tfidf":
            for fold in fold_ids:
                train_mask = df["fold"] != fold
                test_mask = ~train_mask

                vectorizer = build_tfidf_vectorizer()
                X_train = vectorizer.fit_transform(df.loc[train_mask, "text"])
                X_test = vectorizer.transform(df.loc[test_mask, "text"])

                y_train = df.loc[train_mask, "label"].to_numpy()

                classifier = LinearSVC(
                    random_state=random_state,
                    class_weight="balanced",
                    max_iter=5000,
                )
                classifier.fit(X_train, y_train)
                fold_predictions[test_mask.to_numpy()] = classifier.predict(X_test)
        else:
            embeddings_matrix = encode_sentences(
                df["text"].tolist(),
                model_key=key,
                model_overrides=model_overrides,
                batch_size=batch_size,
            )

            for fold in fold_ids:
                train_mask = df["fold"] != fold
                test_mask = ~train_mask

                X_train = embeddings_matrix[train_mask.to_numpy()]
                X_test = embeddings_matrix[test_mask.to_numpy()]

                y_train = df.loc[train_mask, "label"].to_numpy()

                classifier = LinearSVC(
                    random_state=random_state,
                    class_weight="balanced",
                    max_iter=5000,
                )
                classifier.fit(X_train, y_train)
                fold_predictions[test_mask.to_numpy()] = classifier.predict(X_test)

        report_dict = classification_report(
            df["label"].to_numpy(),
            fold_predictions,
            labels=labels_sorted,
            target_names=labels_sorted,
            output_dict=True,
            zero_division=0,
        )

        if verbose:
            print(f"\nEmbedding: {embedding}")
            print(
                classification_report(
                    df["label"].to_numpy(),
                    fold_predictions,
                    labels=labels_sorted,
                    target_names=labels_sorted,
                    zero_division=0,
                )
            )

        column_name = f"prediction_{key}"
        predictions_frame[column_name] = fold_predictions
        reports[key] = {
            "report": report_dict,
            "prediction_column": column_name,
        }

    baselines: Dict[str, Dict[str, object]] = {}
    if include_baselines:
        baselines = _evaluate_random_and_majority(
            df,
            fold_ids,
            labels_sorted,
            random_state=random_state,
            verbose=verbose,
        )
        for baseline_payload in baselines.values():
            predictions_frame[baseline_payload["prediction_column"]] = baseline_payload["predictions"]

    gpt5_output: Dict[str, object] = {}
    if gpt5:
        gpt5_results = run_gpt5_classification(
            source_dir,
            client=gpt5_client,
            config=gpt5_config,
            test_mode=test_mode,
            checkpoint_path=gpt5_checkpoint_path,
            combine_analysis_conclusion=combine_analysis_conclusion,
            filter_implicit_conclusions=filter_implicit_conclusions,
            gpt5_workers=gpt5_workers,
        )
        gpt_pred_frame = gpt5_results["predictions"][["span_id", "prediction_gpt5"]]
        predictions_frame = predictions_frame.merge(
            gpt_pred_frame,
            on="span_id",
            how="left",
        )
        gpt5_output = {
            "report": gpt5_results["report"],
            "prediction_column": "prediction_gpt5",
            "processed_rows": gpt5_results["processed_rows"],
            "checkpoint_path": gpt5_results["checkpoint_path"],
        }

    model_metadata = {
        key: SENTENCE_TRANSFORMER_MODELS[key]
        for key in SENTENCE_TRANSFORMER_MODELS
        if key != "modern-bert"
    }
    if use_modern_bert:
        model_metadata["modern-bert"] = os.getenv("MODERN_BERT_ENDPOINT", "sagemaker-endpoint")
    else:
        model_metadata["modern-bert"] = "disabled"

    return {
        "reports": reports,
        "predictions": predictions_frame,
        "model_defaults": model_metadata,
        "baselines": baselines,
        "gpt5": gpt5_output,
        "fold_audit": fold_audit,
        "labels": labels_sorted,
        "fold_ids": fold_ids,
        "source_dir": str(source_dir),
        "config": {
            "annotation_dir": str(source_dir),
            "embeddings": list(embeddings),
            "n_splits": int(n_splits),
            "random_state": int(random_state),
            "batch_size": int(batch_size),
            "include_baselines": include_baselines,
            "gpt5": gpt5,
            "use_modern_bert": use_modern_bert,
            "test_mode": test_mode,
            "combine_analysis_conclusion": combine_analysis_conclusion,
            "filter_implicit_conclusions": filter_implicit_conclusions,
            "gpt5_checkpoint_path": str(gpt5_checkpoint_path) if gpt5_checkpoint_path is not None else None,
            "gpt5_workers": int(gpt5_workers),
            "model_overrides": dict(model_overrides) if model_overrides is not None else None,
        },
    }


def _evaluate_random_and_majority(
    df: pd.DataFrame,
    fold_ids: Sequence[int],
    labels_sorted: Sequence[str],
    *,
    random_state: int,
    verbose: bool,
) -> Dict[str, Dict[str, object]]:
    """Compute random and majority baselines and optionally print summaries."""

    baseline_reports: Dict[str, Dict[str, object]] = {}
    rng = np.random.default_rng(random_state)

    for baseline_name in ("random", "majority"):
        predictions = np.empty(len(df), dtype=object)

        for fold in fold_ids:
            train_mask = df["fold"] != fold
            test_mask = ~train_mask

            y_train = df.loc[train_mask, "label"].to_numpy()
            test_indices = test_mask.to_numpy()

            if baseline_name == "random":
                choices = np.unique(y_train)
                predictions[test_indices] = rng.choice(choices, size=test_indices.sum())
            else:  # majority
                values, counts = np.unique(y_train, return_counts=True)
                majority_label = values[np.argmax(counts)]
                predictions[test_indices] = majority_label

        report = classification_report(
            df["label"].to_numpy(),
            predictions,
            labels=labels_sorted,
            target_names=labels_sorted,
            output_dict=True,
            zero_division=0,
        )

        baseline_reports[baseline_name] = {
            "report": report,
            "prediction_column": f"prediction_{baseline_name}",
            "predictions": predictions.copy(),
        }

    if verbose:
        _print_baseline_summary(baseline_reports, labels_sorted)

    return baseline_reports


def _print_baseline_summary(
    baseline_reports: Dict[str, Dict[str, object]],
    labels_sorted: Sequence[str],
) -> None:
    """Pretty-print F1-score summaries for baseline classifiers."""

    header = ["Classifier", "Avg"] + list(labels_sorted)
    print("\nBaselines (F1-score)")
    print(" | ".join(header))

    for name in ("random", "majority"):
        report = baseline_reports.get(name, {}).get("report", {})
        if not report:
            continue

        macro = report.get("macro avg", {}).get("f1-score", 0.0)
        row = [name.title(), f"{macro:.2f}"]
        for label in labels_sorted:
            score = report.get(label, {}).get("f1-score", 0.0)
            row.append(f"{score:.2f}")
        print(" - " + " | ".join(row))


def _validate_run_inputs(
    *,
    annotation_dir: Path | str,
    embeddings: Sequence[str],
    n_splits: int,
    random_state: int,
    batch_size: int,
    model_overrides: Optional[Dict[str, str]],
    verbose: bool,
    include_baselines: bool,
    gpt5: bool,
    gpt5_config: "GPT5Config | None",
    gpt5_client: "OpenAI | None",
    use_modern_bert: bool,
    test_mode: bool,
    gpt5_checkpoint_path: Optional[Path | str],
    gpt5_workers: int,
    combine_analysis_conclusion: bool,
    filter_implicit_conclusions: bool,
) -> None:
    source_dir = _coerce_path(annotation_dir, name="annotation_dir")
    if not source_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"annotation_dir must be a directory: {source_dir}")

    if isinstance(embeddings, str) or not isinstance(embeddings, Sequence):
        raise TypeError("embeddings must be a non-string sequence of embedding names.")
    if not embeddings:
        raise ValueError("embeddings must not be empty.")
    for embedding in embeddings:
        if not isinstance(embedding, str):
            raise TypeError(
                f"Each embedding name must be a string, got {type(embedding).__name__}."
            )
        if embedding.lower() not in DEFAULT_EMBEDDINGS:
            raise ValueError(f"Unsupported embedding '{embedding}'.")

    _require_integral(n_splits, name="n_splits", minimum=2)
    _require_integral(random_state, name="random_state")
    _require_integral(batch_size, name="batch_size", minimum=1)
    _require_bool(verbose, name="verbose")
    _require_bool(include_baselines, name="include_baselines")
    _require_bool(gpt5, name="gpt5")
    _require_bool(use_modern_bert, name="use_modern_bert")
    _require_bool(test_mode, name="test_mode")
    _require_bool(combine_analysis_conclusion, name="combine_analysis_conclusion")
    _require_bool(filter_implicit_conclusions, name="filter_implicit_conclusions")
    _require_integral(gpt5_workers, name="gpt5_workers", minimum=1)

    if model_overrides is not None:
        if not isinstance(model_overrides, dict):
            raise TypeError("model_overrides must be a dictionary when provided.")
        for key, value in model_overrides.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("model_overrides keys and values must both be strings.")

    if gpt5_config is not None:
        config_type = getattr(gpt5_config, "__class__", None)
        if config_type is None or config_type.__name__ != "GPT5Config":
            raise TypeError("gpt5_config must be a GPT5Config instance when provided.")
    if gpt5_client is not None and not hasattr(gpt5_client, "responses"):
        raise TypeError("gpt5_client must expose a 'responses' attribute when provided.")
    if gpt5_checkpoint_path is not None:
        _coerce_path(gpt5_checkpoint_path, name="gpt5_checkpoint_path")


def _require_bool(value: object, *, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}.")


def _require_integral(value: object, *, name: str, minimum: Optional[int] = None) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if minimum is not None and int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")


def _coerce_path(value: Path | str, *, name: str) -> Path:
    if not isinstance(value, (Path, str)):
        raise TypeError(f"{name} must be a path-like value, got {type(value).__name__}.")
    return Path(value).expanduser()
