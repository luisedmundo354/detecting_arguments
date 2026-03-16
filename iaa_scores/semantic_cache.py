"""Semantic embedding cache for explicit span texts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Mapping

import numpy as np

from .annotation_graphs import load_annotation_graphs
from .cache_utils import (
    cache_root,
    compute_dataset_fingerprint,
    dump_json,
    ensure_parent_dir,
    load_json,
    repo_relative_path,
)
from .cohere_api import (
    DEFAULT_INPUT_TYPE,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIMENSION,
    embed_texts,
    load_repo_cohere_api_key,
)


def build_or_load_semantic_embedding_cache(
    *,
    repo_root: Path,
    input_dir: Path,
    overwrite: bool = False,
    model: str = DEFAULT_MODEL,
    input_type: str = DEFAULT_INPUT_TYPE,
    output_dimension: int = DEFAULT_OUTPUT_DIMENSION,
    batch_size: int = 96,
    embedder: Callable[..., list[list[float]]] | None = None,
) -> dict:
    """Build or load a strict semantic embedding cache for explicit span texts."""

    input_dir = input_dir.resolve()
    dataset_fingerprint = compute_dataset_fingerprint(input_dir)
    npz_path, manifest_path = semantic_cache_paths(
        repo_root=repo_root,
        input_dir=input_dir,
        model=model,
        input_type=input_type,
        output_dimension=output_dimension,
    )

    if npz_path.exists() or manifest_path.exists():
        if not npz_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(
                f"Semantic cache is incomplete: expected both {npz_path} and {manifest_path}."
            )
        if overwrite:
            pass
        else:
            return load_semantic_embedding_cache(
                repo_root=repo_root,
                input_dir=input_dir,
                model=model,
                input_type=input_type,
                output_dimension=output_dimension,
            )

    graphs_by_doc = load_annotation_graphs(
        input_dir,
        min_annotators=1,
        require_repaired_implicit_offsets=True,
    )
    explicit_texts, node_row_index = _collect_explicit_texts(graphs_by_doc)
    if not explicit_texts:
        raise ValueError("No explicit span texts found for semantic embedding.")

    if embedder is None:
        api_key = load_repo_cohere_api_key(repo_root)
        embedder = embed_texts
        raw_embeddings = embedder(
            explicit_texts,
            api_key=api_key,
            model=model,
            input_type=input_type,
            output_dimension=output_dimension,
            batch_size=batch_size,
        )
    else:
        raw_embeddings = embedder(
            explicit_texts,
            model=model,
            input_type=input_type,
            output_dimension=output_dimension,
            batch_size=batch_size,
        )

    embeddings = np.asarray(raw_embeddings, dtype=np.float32)
    if embeddings.shape != (len(explicit_texts), output_dimension):
        raise ValueError(
            f"Semantic embedding matrix shape {embeddings.shape} does not match "
            f"({len(explicit_texts)}, {output_dimension})."
        )

    ensure_parent_dir(npz_path)
    np.savez_compressed(npz_path, embeddings=embeddings)

    manifest = {
        "cache_version": 1,
        "dataset_path": repo_relative_path(input_dir, repo_root),
        "dataset_fingerprint": dataset_fingerprint,
        "model": model,
        "input_type": input_type,
        "output_dimension": output_dimension,
        "embedding_type": "float",
        "truncate": "NONE",
        "batch_size": batch_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(explicit_texts),
        "row_to_text": explicit_texts,
        "text_to_row": {text: index for index, text in enumerate(explicit_texts)},
        "node_row_index": node_row_index,
    }
    dump_json(manifest, manifest_path)
    return _semantic_cache_payload(npz_path, manifest_path, manifest, embeddings)


def load_semantic_embedding_cache(
    *,
    repo_root: Path,
    input_dir: Path,
    model: str = DEFAULT_MODEL,
    input_type: str = DEFAULT_INPUT_TYPE,
    output_dimension: int = DEFAULT_OUTPUT_DIMENSION,
) -> dict:
    """Load and validate a semantic embedding cache."""

    input_dir = input_dir.resolve()
    dataset_fingerprint = compute_dataset_fingerprint(input_dir)
    npz_path, manifest_path = semantic_cache_paths(
        repo_root=repo_root,
        input_dir=input_dir,
        model=model,
        input_type=input_type,
        output_dimension=output_dimension,
    )
    if not npz_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"Semantic cache not found. Expected {npz_path} and {manifest_path}."
        )

    manifest = load_json(manifest_path)
    required_keys = {
        "dataset_path",
        "dataset_fingerprint",
        "model",
        "input_type",
        "output_dimension",
        "row_count",
        "row_to_text",
        "text_to_row",
        "node_row_index",
    }
    missing = sorted(required_keys - set(manifest))
    if missing:
        raise ValueError(f"Semantic embedding manifest is missing keys: {missing}")

    if manifest["dataset_fingerprint"] != dataset_fingerprint:
        raise ValueError(
            "Semantic embedding cache fingerprint does not match the current dataset. "
            "Use overwrite=True to rebuild it."
        )
    if manifest["model"] != model or manifest["input_type"] != input_type:
        raise ValueError(
            "Semantic embedding cache configuration does not match the requested model/input_type."
        )
    if int(manifest["output_dimension"]) != int(output_dimension):
        raise ValueError(
            "Semantic embedding cache output dimension does not match the requested value."
        )

    npz = np.load(npz_path)
    if "embeddings" not in npz:
        raise ValueError(f"Semantic embedding cache at {npz_path} is missing the 'embeddings' array.")
    embeddings = np.asarray(npz["embeddings"], dtype=np.float32)
    if embeddings.shape != (int(manifest["row_count"]), int(output_dimension)):
        raise ValueError(
            f"Semantic embedding matrix shape {embeddings.shape} does not match manifest row_count/output_dimension."
        )
    return _semantic_cache_payload(npz_path, manifest_path, manifest, embeddings)


def semantic_cache_paths(
    *,
    repo_root: Path,
    input_dir: Path,
    model: str,
    input_type: str,
    output_dimension: int,
) -> tuple[Path, Path]:
    dataset_name = input_dir.name
    file_stem = (
        f"{dataset_name}__cohere_{model}__{input_type}__dim{int(output_dimension)}__float"
    )
    root = cache_root(repo_root) / "embeddings"
    return root / f"{file_stem}.npz", root / f"{file_stem}.manifest.json"


def _collect_explicit_texts(graphs_by_doc) -> tuple[list[str], dict[str, int]]:
    unique_texts: list[str] = []
    text_to_row: Dict[str, int] = {}
    node_row_index: dict[str, int] = {}

    for ann_by_annotator in graphs_by_doc.values():
        for record in ann_by_annotator.values():
            annotation = record.annotation
            for span in annotation.spans:
                if span.is_implicit:
                    continue
                text = span.text.strip()
                if not text:
                    raise ValueError(
                        f"Explicit span {span.node_id} in {record.path} has empty text."
                    )
                row_index = text_to_row.get(text)
                if row_index is None:
                    row_index = len(unique_texts)
                    unique_texts.append(text)
                    text_to_row[text] = row_index
                node_key = f"{annotation.source_file}::{span.node_id}"
                node_row_index[node_key] = row_index

    return unique_texts, node_row_index


def _semantic_cache_payload(
    npz_path: Path,
    manifest_path: Path,
    manifest: Mapping[str, object],
    embeddings: np.ndarray,
) -> dict:
    text_to_row = {str(text): int(index) for text, index in dict(manifest["text_to_row"]).items()}
    text_to_embedding = {
        text: embeddings[row_index]
        for text, row_index in text_to_row.items()
    }
    return {
        "npz_path": npz_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "embeddings": embeddings,
        "text_to_embedding": text_to_embedding,
    }
