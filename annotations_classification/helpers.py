"""Utility functions for span-level classification on Label Studio annotations."""
from __future__ import annotations

import json
from functools import lru_cache
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:  # pragma: no cover - optional import for newer scikit-learn versions
    from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None  # type: ignore

from sentence_transformers import SentenceTransformer

# Default SentenceTransformer models for each embedding option.
SENTENCE_TRANSFORMER_MODELS: Dict[str, str] = {
    "sbert": "sentence-transformers/bert-base-nli-mean-tokens",
    "legal-bert": "nlpaueb/legal-bert-small-uncased",
    "modern-bert": "sagemaker-endpoint",
}


def load_annotation_spans(annotation_dir: Path | str) -> pd.DataFrame:
    """Return a dataframe with one row per labeled span.

    The function expects Label Studio JSON exports where each file contains a
    single task. Entries without textual content or class labels are skipped.
    """

    annotation_path = Path(annotation_dir)
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_path}")

    records = []
    json_files = sorted(annotation_path.glob("*.json"), key=lambda p: p.stem)
    if not json_files:
        raise FileNotFoundError(f"No JSON annotation files found in {annotation_path}")

    for json_file in json_files:
        with json_file.open("r", encoding="utf8") as handle:
            payload = json.load(handle)

        task = payload.get("task", {})
        task_id = task.get("id")
        data = task.get("data", {})
        document_id = data.get("ref_id")

        for entry in payload.get("result", []):
            if entry.get("type") != "labels":
                continue

            value = entry.get("value") or {}
            labels = value.get("labels") or []
            text = value.get("text") or ""

            text = text.strip()
            if not labels or not text:
                continue

            label = labels[0]
            records.append(
                {
                    "span_id": f"{json_file.stem}:{entry.get('id')}",
                    "task_id": task_id,
                    "document_id": document_id,
                    "start": value.get("start"),
                    "end": value.get("end"),
                    "text": text,
                    "label": label,
                }
            )

    if not records:
        raise ValueError("No labeled spans were found in the provided annotations.")

    df = pd.DataFrame.from_records(records)
    df.reset_index(drop=True, inplace=True)
    return df


def load_annotation_contents(annotation_dir: Path | str) -> Dict[object, str]:
    """Return a mapping of ``document_id`` (``ref_id``) to the raw ``case_content`` text.

    The function expects Label Studio JSON exports where each file contains a
    single task. If multiple exports share the same ``ref_id``, the longest
    non-empty ``case_content`` is retained.
    """

    annotation_path = Path(annotation_dir)
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_path}")

    contents: Dict[object, str] = {}
    json_files = sorted(annotation_path.glob("*.json"), key=lambda p: p.stem)
    if not json_files:
        raise FileNotFoundError(f"No JSON annotation files found in {annotation_path}")

    for json_file in json_files:
        with json_file.open("r", encoding="utf8") as handle:
            payload = json.load(handle)

        task = payload.get("task", {})
        data = task.get("data", {})
        document_id = data.get("ref_id")
        case_content = data.get("case_content")

        if document_id is None or not isinstance(case_content, str) or not case_content:
            continue

        existing = contents.get(document_id)
        if existing is None or len(case_content) > len(existing):
            contents[document_id] = case_content

    return contents


def filter_implicit_conclusions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop spans whose text is only an implicit intermediate conclusion code."""

    if df.empty:
        return df.copy()

    text_series = df["text"].astype(str).str.strip().str.lower()
    mask = text_series.str.startswith("implicit intermediate conclusion")
    filtered = df.loc[~mask].copy()
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def assign_stratified_folds(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    group_column: Optional[str] = "document_id",
) -> pd.DataFrame:
    """Annotate the dataframe with strict case-disjoint stratified folds."""

    _require_dataframe(df, name="df")
    if df.empty:
        raise ValueError("Cannot assign folds to an empty dataframe.")
    _require_integral(n_splits, name="n_splits", minimum=2)
    _require_integral(random_state, name="random_state")
    _require_non_empty_string(group_column, name="group_column")
    _require_required_columns(df, required_columns=("text", "label", group_column))

    if StratifiedGroupKFold is None:
        raise ImportError(
            "StratifiedGroupKFold is required for case-disjoint evaluation. "
            "Install a scikit-learn version that provides it."
        )

    if df["label"].isna().any():
        raise ValueError("Column 'label' contains null values.")
    if df["text"].isna().any():
        raise ValueError("Column 'text' contains null values.")

    groups = df[group_column]
    if groups.isna().any():
        raise ValueError(f"Column '{group_column}' contains null group identifiers.")

    unique_groups = groups.nunique(dropna=False)
    if unique_groups < n_splits:
        raise ValueError(
            f"Requested {n_splits} folds but only {unique_groups} unique groups are available "
            f"in column '{group_column}'."
        )

    label_counts = df["label"].value_counts()
    min_class = int(label_counts.min())
    if min_class < n_splits:
        raise ValueError(
            f"Requested {n_splits} folds but the smallest class has only {min_class} samples."
        )

    folds = np.zeros(len(df), dtype=int)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    split_iterator: Iterable = splitter.split(df["text"], df["label"], groups=groups.to_numpy())
    for fold_idx, (_, test_index) in enumerate(split_iterator, start=1):
        folds[test_index] = fold_idx

    df_with_folds = df.copy()
    df_with_folds["fold"] = folds
    return df_with_folds


def build_fold_audit(
    df: pd.DataFrame,
    *,
    group_column: str = "document_id",
    label_column: str = "label",
    fold_column: str = "fold",
    splitter_name: str = "StratifiedGroupKFold",
) -> Dict[str, Any]:
    """Build a JSON-serializable summary of grouped fold assignments."""

    _require_dataframe(df, name="df")
    if df.empty:
        raise ValueError("Cannot audit folds for an empty dataframe.")
    _require_non_empty_string(group_column, name="group_column")
    _require_non_empty_string(label_column, name="label_column")
    _require_non_empty_string(fold_column, name="fold_column")
    _require_non_empty_string(splitter_name, name="splitter_name")
    _require_required_columns(df, required_columns=(group_column, label_column, fold_column))

    if df[group_column].isna().any():
        raise ValueError(f"Column '{group_column}' contains null group identifiers.")
    if df[label_column].isna().any():
        raise ValueError(f"Column '{label_column}' contains null labels.")
    if df[fold_column].isna().any():
        raise ValueError(f"Column '{fold_column}' contains null fold assignments.")

    fold_ids = sorted(int(value) for value in df[fold_column].unique())
    label_counts = _series_to_python_dict(df[label_column].value_counts().sort_index())
    folds_payload: list[Dict[str, Any]] = []
    test_group_union: set[Any] = set()
    total_overlap_count = 0

    for fold_id in fold_ids:
        test_df = df.loc[df[fold_column] == fold_id].copy()
        train_df = df.loc[df[fold_column] != fold_id].copy()
        test_groups = set(test_df[group_column].tolist())
        train_groups = set(train_df[group_column].tolist())
        overlap_groups = train_groups.intersection(test_groups)
        total_overlap_count += len(overlap_groups)
        test_group_union.update(test_groups)

        folds_payload.append(
            {
                "fold": fold_id,
                "train_row_count": int(len(train_df)),
                "test_row_count": int(len(test_df)),
                "train_group_count": int(len(train_groups)),
                "test_group_count": int(len(test_groups)),
                "train_groups": _sorted_python_list(train_groups),
                "test_groups": _sorted_python_list(test_groups),
                "overlap_group_count": int(len(overlap_groups)),
                "overlap_groups": _sorted_python_list(overlap_groups),
                "train_label_counts": _series_to_python_dict(
                    train_df[label_column].value_counts().sort_index()
                ),
                "test_label_counts": _series_to_python_dict(
                    test_df[label_column].value_counts().sort_index()
                ),
            }
        )

    all_groups = set(df[group_column].tolist())
    return {
        "splitter": splitter_name,
        "group_column": group_column,
        "label_column": label_column,
        "fold_column": fold_column,
        "row_count": int(len(df)),
        "group_count": int(len(all_groups)),
        "fold_count": int(len(fold_ids)),
        "label_counts": label_counts,
        "all_group_ids": _sorted_python_list(all_groups),
        "all_test_groups_covered": test_group_union == all_groups,
        "total_overlap_group_count": int(total_overlap_count),
        "folds": folds_payload,
    }


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str) -> SentenceTransformer:
    """Cache SentenceTransformer instances to avoid redundant downloads."""

    return SentenceTransformer(model_name)


def encode_sentences(
    texts: Sequence[str],
    *,
    model_key: str,
    model_overrides: Optional[Dict[str, str]] = None,
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """Embed sentences using the requested SentenceTransformer model."""

    key = model_key.lower()
    if key == "modern-bert":
        from modern_bert_sagemaker import get_default_client

        client = get_default_client()
        return client.embeddings(texts, batch_size=batch_size, normalize=normalize)

    model_lookup = {**SENTENCE_TRANSFORMER_MODELS}
    if model_overrides:
        model_lookup.update({k.lower(): v for k, v in model_overrides.items()})

    if key not in model_lookup:
        raise KeyError(f"Unknown embedding key '{model_key}'.")

    model = _load_sentence_transformer(model_lookup[key])
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=normalize,
    )
    return np.asarray(embeddings)


def build_tfidf_vectorizer(**kwargs) -> TfidfVectorizer:
    """Return a TF-IDF vectorizer with sane defaults for legal text."""

    defaults = {
        "lowercase": True,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.9,
    }
    defaults.update(kwargs)
    return TfidfVectorizer(**defaults)


def _require_dataframe(value: object, *, name: str) -> None:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(value).__name__}.")


def _require_integral(value: object, *, name: str, minimum: Optional[int] = None) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if minimum is not None and int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")


def _require_non_empty_string(value: object, *, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}.")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_required_columns(df: pd.DataFrame, *, required_columns: Sequence[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        missing_display = ", ".join(sorted(missing))
        raise KeyError(f"Missing required dataframe columns: {missing_display}.")


def _series_to_python_dict(series: pd.Series) -> Dict[Any, Any]:
    return {key: _to_python_scalar(value) for key, value in series.to_dict().items()}


def _sorted_python_list(values: Iterable[Any]) -> list[Any]:
    normalized = [_to_python_scalar(value) for value in values]
    return sorted(normalized, key=lambda item: (str(type(item)), str(item)))


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
