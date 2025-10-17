"""Utility functions for span-level classification on Label Studio annotations."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

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


def assign_stratified_folds(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    group_column: Optional[str] = "document_id",
) -> pd.DataFrame:
    """Annotate the dataframe with stratified CV folds.

    If grouping information is available and the installed scikit-learn version
    exposes :class:`StratifiedGroupKFold`, the function keeps spans from the
    same document in the same fold.
    """

    if df.empty:
        raise ValueError("Cannot assign folds to an empty dataframe.")

    label_counts = df["label"].value_counts()
    min_class = int(label_counts.min())
    if min_class < 2:
        raise ValueError("At least two samples per class are required for CV.")

    effective_splits = min(n_splits, min_class)
    if effective_splits < 2:
        raise ValueError("Unable to create more than one fold with the data provided.")

    folds = np.zeros(len(df), dtype=int)
    groups: Optional[Iterable] = None
    if group_column and group_column in df.columns:
        groups = df[group_column].to_numpy()

    splitter: StratifiedKFold
    split_iterator: Iterable

    if groups is not None and StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
        split_iterator = splitter.split(df["text"], df["label"], groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
        split_iterator = splitter.split(df["text"], df["label"])

    for fold_idx, (_, test_index) in enumerate(split_iterator, start=1):
        folds[test_index] = fold_idx

    df_with_folds = df.copy()
    df_with_folds["fold"] = folds
    return df_with_folds


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
        from .modern_bert_sagemaker import get_default_client

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
