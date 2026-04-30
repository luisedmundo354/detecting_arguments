"""Reporting helpers for classification experiment artifacts."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import pandas as pd


def export_run_artifacts(
    *,
    results: Mapping[str, Any],
    output_root: Path | str,
    run_name: str,
    overwrite: bool = False,
    max_examples_per_pair: int = 10,
) -> Dict[str, str]:
    """Export fold, prediction, confusion, and summary artifacts for one run."""

    _validate_results_payload(results)
    _require_non_empty_string(run_name, name="run_name")
    _require_bool(overwrite, name="overwrite")
    _require_positive_int(max_examples_per_pair, name="max_examples_per_pair")

    root_path = _coerce_path(output_root, name="output_root").resolve()
    root_path.mkdir(parents=True, exist_ok=True)

    run_dir = root_path / run_name
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {run_dir}. "
                "Pass overwrite=True or choose a different run_name."
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    predictions = results["predictions"].copy()
    fold_manifest = results["fold_audit"]
    summary = build_classification_summary(results)
    confusion = build_confusion_analysis(
        predictions,
        labels=results["labels"],
        max_examples_per_pair=max_examples_per_pair,
    )

    predictions_path = run_dir / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    fold_manifest_path = run_dir / "fold_manifest.json"
    fold_manifest_path.write_text(json.dumps(fold_manifest, indent=2), encoding="utf8")

    fold_summary_path = run_dir / "fold_summary.txt"
    fold_summary_path.write_text(format_fold_summary(fold_manifest), encoding="utf8")

    confusion_json_path = run_dir / "confusion_analysis.json"
    confusion_json_path.write_text(json.dumps(confusion, indent=2), encoding="utf8")

    confusion_txt_path = run_dir / "confusion_analysis.txt"
    confusion_txt_path.write_text(format_confusion_analysis(confusion), encoding="utf8")

    summary_json_path = run_dir / "classification_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf8")

    summary_txt_path = run_dir / "classification_summary.txt"
    summary_txt_path.write_text(format_classification_summary(summary), encoding="utf8")

    return {
        "run_dir": str(run_dir),
        "predictions_csv": str(predictions_path),
        "fold_manifest_json": str(fold_manifest_path),
        "fold_summary_txt": str(fold_summary_path),
        "confusion_analysis_json": str(confusion_json_path),
        "confusion_analysis_txt": str(confusion_txt_path),
        "classification_summary_json": str(summary_json_path),
        "classification_summary_txt": str(summary_txt_path),
    }


def build_classification_summary(results: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a summary payload for all model reports in one run."""

    _validate_results_payload(results)
    labels = list(results["labels"])
    model_entries = []

    for model_name, payload in _iter_model_reports(results):
        report = payload["report"]
        row = {
            "model": model_name,
            "prediction_column": payload["prediction_column"],
            "macro_f1": float(report["macro avg"]["f1-score"]),
        }
        for label in labels:
            row[label] = float(report.get(label, {}).get("f1-score", 0.0))
        model_entries.append(row)

    return {
        "source_dir": results["source_dir"],
        "config": results["config"],
        "labels": labels,
        "models": model_entries,
    }


def build_summary_table_frame(results: Mapping[str, Any]) -> pd.DataFrame:
    """Return a DataFrame shaped for notebook display and manuscript checks."""

    summary = build_classification_summary(results)
    labels = summary["labels"]
    columns = ["model", "macro_f1"] + labels
    frame = pd.DataFrame(summary["models"])
    return frame.loc[:, columns]


def build_confusion_analysis(
    predictions: pd.DataFrame,
    *,
    labels: Sequence[str],
    gold_column: str = "label",
    max_examples_per_pair: int = 10,
) -> Dict[str, Any]:
    """Build confusion matrices and focused Rule/Analysis diagnostics."""

    _require_dataframe(predictions, name="predictions")
    _require_non_empty_string(gold_column, name="gold_column")
    _require_positive_int(max_examples_per_pair, name="max_examples_per_pair")
    if gold_column not in predictions.columns:
        raise KeyError(f"Missing gold column '{gold_column}' in predictions dataframe.")
    if not labels:
        raise ValueError("labels must not be empty.")

    label_list = list(labels)
    prediction_columns = _prediction_columns(predictions)
    if not prediction_columns:
        raise ValueError("No prediction columns were found in the predictions dataframe.")

    models: Dict[str, Any] = {}
    for prediction_column in prediction_columns:
        gold = predictions[gold_column]
        pred = predictions[prediction_column]
        counts = pd.crosstab(gold, pred, dropna=False)
        counts = counts.reindex(index=label_list, columns=label_list, fill_value=0)
        row_totals = counts.sum(axis=1).replace(0, 1)
        row_normalized = counts.div(row_totals, axis=0)

        models[prediction_column] = {
            "labels": label_list,
            "counts": counts.astype(int).to_dict(orient="index"),
            "row_normalized": row_normalized.round(6).to_dict(orient="index"),
            "rule_analysis": _build_rule_analysis_section(
                predictions,
                prediction_column=prediction_column,
                gold_column=gold_column,
                max_examples_per_pair=max_examples_per_pair,
            ),
        }

    return {
        "gold_column": gold_column,
        "labels": label_list,
        "prediction_columns": prediction_columns,
        "models": models,
    }


def format_fold_summary(fold_manifest: Mapping[str, Any]) -> str:
    """Render a plain-text summary of grouped fold assignments."""

    _require_mapping(fold_manifest, name="fold_manifest")
    lines = [
        f"Splitter: {fold_manifest['splitter']}",
        f"Rows: {fold_manifest['row_count']}",
        f"Groups: {fold_manifest['group_count']}",
        f"Folds: {fold_manifest['fold_count']}",
        f"All groups covered by test folds: {fold_manifest['all_test_groups_covered']}",
        f"Total overlap group count: {fold_manifest['total_overlap_group_count']}",
        "",
        "Global label counts:",
    ]
    for label, count in fold_manifest["label_counts"].items():
        lines.append(f"  {label}: {count}")

    for fold in fold_manifest["folds"]:
        lines.extend(
            [
                "",
                f"Fold {fold['fold']}",
                f"  Train rows: {fold['train_row_count']}",
                f"  Test rows: {fold['test_row_count']}",
                f"  Train groups: {fold['train_group_count']}",
                f"  Test groups: {fold['test_group_count']}",
                f"  Overlap groups: {fold['overlap_group_count']}",
                f"  Train group IDs: {_format_value_list(fold['train_groups'])}",
                f"  Test group IDs: {_format_value_list(fold['test_groups'])}",
                "  Train label counts:",
            ]
        )
        for label, count in fold["train_label_counts"].items():
            lines.append(f"    {label}: {count}")
        lines.append("  Test label counts:")
        for label, count in fold["test_label_counts"].items():
            lines.append(f"    {label}: {count}")

    return "\n".join(lines) + "\n"


def format_confusion_analysis(confusion: Mapping[str, Any]) -> str:
    """Render model confusion matrices and Rule/Analysis diagnostics."""

    _require_mapping(confusion, name="confusion")
    lines = [
        f"Gold column: {confusion['gold_column']}",
        f"Labels: {', '.join(confusion['labels'])}",
    ]

    for prediction_column in confusion["prediction_columns"]:
        payload = confusion["models"][prediction_column]
        lines.extend(["", f"Model: {prediction_column}", "Counts:"])
        lines.extend(_format_matrix(payload["counts"], payload["labels"]))
        lines.append("Row-normalized:")
        lines.extend(_format_matrix(payload["row_normalized"], payload["labels"], decimals=4))

        rule_analysis = payload["rule_analysis"]
        lines.extend(
            [
                "Rule vs Analysis:",
                f"  Analysis -> Rule count: {rule_analysis['analysis_to_rule_count']}",
                f"  Rule -> Analysis count: {rule_analysis['rule_to_analysis_count']}",
                f"  Analysis -> Rule rate: {rule_analysis['analysis_to_rule_rate']:.4f}",
                f"  Rule -> Analysis rate: {rule_analysis['rule_to_analysis_rate']:.4f}",
                "  Analysis -> Rule examples:",
            ]
        )
        lines.extend(_format_example_block(rule_analysis["analysis_to_rule_examples"]))
        lines.append("  Rule -> Analysis examples:")
        lines.extend(_format_example_block(rule_analysis["rule_to_analysis_examples"]))

    return "\n".join(lines) + "\n"


def format_classification_summary(summary: Mapping[str, Any]) -> str:
    """Render macro and per-label F1 scores as plain text."""

    _require_mapping(summary, name="summary")
    labels = summary["labels"]
    header = ["Model", "Macro F1"] + list(labels)
    lines = [
        f"Source directory: {summary['source_dir']}",
        f"Config: {json.dumps(summary['config'], sort_keys=True)}",
        "",
        " | ".join(header),
    ]

    for row in summary["models"]:
        values = [row["model"], f"{row['macro_f1']:.4f}"]
        for label in labels:
            values.append(f"{row[label]:.4f}")
        lines.append(" | ".join(values))

    return "\n".join(lines) + "\n"


def format_rule_analysis_section(
    confusion: Mapping[str, Any],
    *,
    prediction_column: str,
) -> str:
    """Render just the Rule/Analysis section for one model."""

    _require_mapping(confusion, name="confusion")
    _require_non_empty_string(prediction_column, name="prediction_column")
    if prediction_column not in confusion["models"]:
        raise KeyError(f"Unknown prediction column '{prediction_column}'.")

    payload = confusion["models"][prediction_column]["rule_analysis"]
    lines = [
        f"Model: {prediction_column}",
        f"Analysis -> Rule count: {payload['analysis_to_rule_count']}",
        f"Rule -> Analysis count: {payload['rule_to_analysis_count']}",
        f"Analysis -> Rule rate: {payload['analysis_to_rule_rate']:.4f}",
        f"Rule -> Analysis rate: {payload['rule_to_analysis_rate']:.4f}",
        "Analysis -> Rule examples:",
    ]
    lines.extend(_format_example_block(payload["analysis_to_rule_examples"]))
    lines.append("Rule -> Analysis examples:")
    lines.extend(_format_example_block(payload["rule_to_analysis_examples"]))
    return "\n".join(lines) + "\n"


def _build_rule_analysis_section(
    predictions: pd.DataFrame,
    *,
    prediction_column: str,
    gold_column: str,
    max_examples_per_pair: int,
) -> Dict[str, Any]:
    if prediction_column not in predictions.columns:
        raise KeyError(f"Missing prediction column '{prediction_column}'.")

    gold = predictions[gold_column]
    pred = predictions[prediction_column]
    analysis_total = int((gold == "Analysis").sum())
    rule_total = int((gold == "Rule").sum())

    analysis_to_rule_mask = (gold == "Analysis") & (pred == "Rule")
    rule_to_analysis_mask = (gold == "Rule") & (pred == "Analysis")

    analysis_to_rule_count = int(analysis_to_rule_mask.sum())
    rule_to_analysis_count = int(rule_to_analysis_mask.sum())

    return {
        "analysis_total": analysis_total,
        "rule_total": rule_total,
        "analysis_to_rule_count": analysis_to_rule_count,
        "rule_to_analysis_count": rule_to_analysis_count,
        "analysis_to_rule_rate": analysis_to_rule_count / analysis_total if analysis_total else 0.0,
        "rule_to_analysis_rate": rule_to_analysis_count / rule_total if rule_total else 0.0,
        "analysis_to_rule_examples": _extract_examples(
            predictions.loc[analysis_to_rule_mask],
            prediction_column=prediction_column,
            max_examples=max_examples_per_pair,
        ),
        "rule_to_analysis_examples": _extract_examples(
            predictions.loc[rule_to_analysis_mask],
            prediction_column=prediction_column,
            max_examples=max_examples_per_pair,
        ),
    }


def _extract_examples(
    df: pd.DataFrame,
    *,
    prediction_column: str,
    max_examples: int,
) -> list[Dict[str, Any]]:
    if df.empty:
        return []

    ordered = df.sort_values(["fold", "document_id", "start", "span_id"]).head(max_examples)
    examples = []
    for _, row in ordered.iterrows():
        examples.append(
            {
                "span_id": row["span_id"],
                "document_id": row["document_id"],
                "fold": int(row["fold"]),
                "gold": row["label"],
                "prediction": row[prediction_column],
                "text_excerpt": _truncate_text(row["text"]),
            }
        )
    return examples


def _iter_model_reports(results: Mapping[str, Any]) -> Iterable[tuple[str, Dict[str, Any]]]:
    reports = results["reports"]
    for key in reports:
        yield key, reports[key]

    if results["gpt5"]:
        yield "gpt5", results["gpt5"]

    baselines = results["baselines"]
    for key in ("random", "majority"):
        if key in baselines:
            yield key, baselines[key]


def _prediction_columns(predictions: pd.DataFrame) -> list[str]:
    return [column for column in predictions.columns if column.startswith("prediction_")]


def _format_matrix(
    matrix: Mapping[str, Mapping[str, Any]],
    labels: Sequence[str],
    *,
    decimals: int | None = None,
) -> list[str]:
    header = "  true\\pred | " + " | ".join(labels)
    lines = [header]
    for row_label in labels:
        row = matrix[row_label]
        rendered = []
        for column_label in labels:
            value = row[column_label]
            if decimals is None:
                rendered.append(str(value))
            else:
                rendered.append(f"{float(value):.{decimals}f}")
        lines.append("  " + row_label + " | " + " | ".join(rendered))
    return lines


def _format_example_block(examples: Sequence[Mapping[str, Any]]) -> list[str]:
    if not examples:
        return ["    <none>"]
    lines = []
    for example in examples:
        lines.append(
            "    "
            + f"{example['span_id']} | doc={example['document_id']} | fold={example['fold']} | "
            + f"gold={example['gold']} | pred={example['prediction']}"
        )
        lines.append("      " + example["text_excerpt"])
    return lines


def _format_value_list(values: Sequence[Any]) -> str:
    return ", ".join(str(value) for value in values)


def _truncate_text(value: Any, *, limit: int = 240) -> str:
    if not isinstance(value, str):
        raise TypeError(f"text values must be strings, got {type(value).__name__}.")
    collapsed = " ".join(value.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _validate_results_payload(results: Mapping[str, Any]) -> None:
    _require_mapping(results, name="results")
    required_keys = {
        "reports",
        "predictions",
        "baselines",
        "gpt5",
        "fold_audit",
        "labels",
        "source_dir",
        "config",
    }
    missing = sorted(required_keys.difference(results.keys()))
    if missing:
        raise KeyError(f"Missing required results keys: {', '.join(missing)}.")
    _require_dataframe(results["predictions"], name="results['predictions']")
    _require_mapping(results["fold_audit"], name="results['fold_audit']")
    if not isinstance(results["labels"], Sequence) or isinstance(results["labels"], str):
        raise TypeError("results['labels'] must be a non-string sequence.")


def _require_mapping(value: object, *, name: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")


def _require_dataframe(value: object, *, name: str) -> None:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(value).__name__}.")


def _require_non_empty_string(value: object, *, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}.")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_bool(value: object, *, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}.")


def _require_positive_int(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")


def _coerce_path(value: Path | str, *, name: str) -> Path:
    if not isinstance(value, (Path, str)):
        raise TypeError(f"{name} must be a path-like value, got {type(value).__name__}.")
    return Path(value).expanduser()
