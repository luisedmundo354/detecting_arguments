"""Span classification routines restricted to Linear SVC baselines."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Sequence

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

from .helpers import (
    SENTENCE_TRANSFORMER_MODELS,
    assign_stratified_folds,
    build_tfidf_vectorizer,
    encode_sentences,
    load_annotation_spans,
)
from .gpt5_classifier import run_gpt5_classification

if TYPE_CHECKING:  # pragma: no cover
    from openai import OpenAI
    from .gpt5_classifier import GPT5Config

DEFAULT_EMBEDDINGS: Sequence[str] = ("tfidf", "sbert", "legal-bert", "modern-bert")


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
) -> Dict[str, Dict[str, object]]:
    """Execute stratified k-fold evaluation with a Linear SVC back-end.

    Parameters
    ----------
    annotation_dir:
        Directory containing Label Studio JSON exports.
    embeddings:
        Iterable of embedding identifiers; supported values are ``tfidf``,
        ``sbert``, ``legal-bert``, and ``modern-bert``.
    n_splits:
        Number of folds for cross-validation. The value is clipped if any class
        has fewer examples than the requested number of folds.
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
    use_modern_bert:
        Toggle Modern-BERT embeddings. When ``False`` the embedding is skipped
        without attempting to reach SageMaker.
    gpt5_checkpoint_path:
        Optional JSONL path used to persist GPT-5 predictions between runs. Ignored
        in test mode.

    Returns
    -------
    Dict[str, Dict[str, object]]
        Reports for each embedding, the combined predictions dataframe, model
        lookup metadata, and (optionally) baseline outputs.
    """

    if test_mode:
        verbose = True

    df = load_annotation_spans(annotation_dir)
    df = assign_stratified_folds(df, n_splits=n_splits, random_state=random_state)

    labels_sorted = sorted(df["label"].unique())
    fold_ids = sorted(df["fold"].unique())

    predictions_frame = df[["span_id", "label", "fold"]].copy()
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

    gpt5_output: Dict[str, object] = {}
    if gpt5:
        gpt5_results = run_gpt5_classification(
            annotation_dir,
            client=gpt5_client,
            config=gpt5_config,
            test_mode=test_mode,
            checkpoint_path=gpt5_checkpoint_path,
        )
        gpt_pred_frame = gpt5_results["predictions"][["span_id", "prediction_gpt5"]]
        predictions_frame = predictions_frame.merge(
            gpt_pred_frame,
            on="span_id",
            how="left",
        )
        gpt5_output = {"report": gpt5_results["report"], "prediction_column": "prediction_gpt5"}

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

        baseline_reports[baseline_name] = {"report": report}

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
