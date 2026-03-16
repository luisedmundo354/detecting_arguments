"""Persistent explicit-span pair caches for edit and semantic matching."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

from .alignment import align_explicit_spans, align_explicit_spans_semantic
from .annotation_graphs import load_annotation_graphs
from .cache_utils import cache_root, compute_dataset_fingerprint, dump_json, float_slug, load_json, repo_relative_path


def build_or_load_pair_cache(
    *,
    repo_root: Path,
    input_dir: Path,
    backend: str,
    overwrite: bool = False,
    metric: str | None = None,
    min_sim: float = 0.0,
    semantic_cache: Mapping[str, object] | None = None,
) -> dict:
    """Build or load a strict pair cache for the repaired overlap set."""

    input_dir = input_dir.resolve()
    dataset_fingerprint = compute_dataset_fingerprint(input_dir)
    cache_path = pair_cache_path(
        repo_root=repo_root,
        input_dir=input_dir,
        backend=backend,
        metric=metric,
        min_sim=min_sim,
        semantic_cache=semantic_cache,
    )

    if cache_path.exists() and not overwrite:
        return load_pair_cache(
            repo_root=repo_root,
            input_dir=input_dir,
            backend=backend,
            metric=metric,
            min_sim=min_sim,
            semantic_cache=semantic_cache,
        )

    graphs_by_doc = load_annotation_graphs(
        input_dir,
        min_annotators=2,
        require_repaired_implicit_offsets=True,
    )
    config = _pair_backend_config(
        backend=backend,
        metric=metric,
        min_sim=min_sim,
        semantic_cache=semantic_cache,
    )

    documents = []
    for ref_id, records_by_annotator in sorted(graphs_by_doc.items()):
        annotators = tuple(sorted(records_by_annotator))
        if len(annotators) != 2:
            raise ValueError(
                f"Pair cache generation requires exactly two annotators per document; "
                f"ref_id {ref_id} has {len(annotators)} annotators."
            )
        record_a = records_by_annotator[annotators[0]]
        record_b = records_by_annotator[annotators[1]]

        if backend == "edit":
            alignment = align_explicit_spans(
                record_a.annotation.spans,
                record_b.annotation.spans,
                metric=str(metric),
                min_sim=min_sim,
            )
        elif backend == "semantic":
            if semantic_cache is None:
                raise ValueError("semantic_cache is required for semantic pair cache generation.")
            alignment = align_explicit_spans_semantic(
                record_a.annotation.spans,
                record_b.annotation.spans,
                text_to_embedding=semantic_cache["text_to_embedding"],
                min_sim=min_sim,
            )
        else:
            raise ValueError(f"Unsupported pair-cache backend: {backend}")

        documents.append(
            {
                "ref_id": ref_id,
                "annotators": list(annotators),
                "files": [record_a.path.name, record_b.path.name],
                "span_counts_by_annotator": {
                    annotators[0]: _explicit_span_counts(record_a.annotation),
                    annotators[1]: _explicit_span_counts(record_b.annotation),
                },
                "matches": [
                    {
                        "label": match.label,
                        "node_id_a": match.span_a.node_id,
                        "node_id_b": match.span_b.node_id,
                        "text_a": match.span_a.text,
                        "text_b": match.span_b.text,
                        "similarity": match.similarity,
                        "source_file_a": record_a.annotation.source_file,
                        "source_file_b": record_b.annotation.source_file,
                    }
                    for match in alignment.matches
                ],
                "unmatched_a": [
                    {
                        "label": span.label,
                        "node_id": span.node_id,
                        "text": span.text,
                        "source_file": record_a.annotation.source_file,
                    }
                    for span in alignment.unmatched_a
                ],
                "unmatched_b": [
                    {
                        "label": span.label,
                        "node_id": span.node_id,
                        "text": span.text,
                        "source_file": record_b.annotation.source_file,
                    }
                    for span in alignment.unmatched_b
                ],
            }
        )

    pair_cache = {
        "cache_version": 1,
        "dataset_path": repo_relative_path(input_dir, repo_root),
        "dataset_fingerprint": dataset_fingerprint,
        "backend": backend,
        "label_restricted": True,
        "config": config,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "documents": documents,
    }
    dump_json(pair_cache, cache_path)
    return pair_cache


def load_pair_cache(
    *,
    repo_root: Path,
    input_dir: Path,
    backend: str,
    metric: str | None = None,
    min_sim: float = 0.0,
    semantic_cache: Mapping[str, object] | None = None,
) -> dict:
    """Load and validate a pair cache for the requested backend/config."""

    input_dir = input_dir.resolve()
    dataset_fingerprint = compute_dataset_fingerprint(input_dir)
    cache_path = pair_cache_path(
        repo_root=repo_root,
        input_dir=input_dir,
        backend=backend,
        metric=metric,
        min_sim=min_sim,
        semantic_cache=semantic_cache,
    )
    if not cache_path.exists():
        raise FileNotFoundError(f"Pair cache not found at {cache_path}")

    pair_cache = load_json(cache_path)
    required_keys = {"dataset_path", "dataset_fingerprint", "backend", "config", "documents"}
    missing = sorted(required_keys - set(pair_cache))
    if missing:
        raise ValueError(f"Pair cache {cache_path} is missing keys: {missing}")
    if pair_cache["dataset_fingerprint"] != dataset_fingerprint:
        raise ValueError(
            "Pair cache fingerprint does not match the current dataset. Use overwrite=True to rebuild it."
        )
    if pair_cache["backend"] != backend:
        raise ValueError(
            f"Pair cache backend mismatch: expected {backend}, found {pair_cache['backend']}"
        )

    expected_config = _pair_backend_config(
        backend=backend,
        metric=metric,
        min_sim=min_sim,
        semantic_cache=semantic_cache,
    )
    if pair_cache["config"] != expected_config:
        raise ValueError(
            f"Pair cache configuration mismatch for {cache_path}. "
            f"Expected {expected_config}, found {pair_cache['config']}."
        )
    if pair_cache.get("label_restricted") is not True:
        raise ValueError("Pair cache must be label_restricted=True for this pipeline.")
    return pair_cache


def pair_cache_path(
    *,
    repo_root: Path,
    input_dir: Path,
    backend: str,
    metric: str | None,
    min_sim: float,
    semantic_cache: Mapping[str, object] | None,
) -> Path:
    dataset_name = input_dir.name
    if backend == "edit":
        if not metric:
            raise ValueError("metric is required for the edit pair cache path.")
        file_name = (
            f"{dataset_name}__pairs__edit_{metric}__min{float_slug(min_sim)}__label_restricted.json"
        )
    elif backend == "semantic":
        if semantic_cache is None:
            raise ValueError("semantic_cache is required for the semantic pair cache path.")
        manifest = dict(semantic_cache["manifest"])
        model = manifest["model"]
        input_type = manifest["input_type"]
        output_dimension = int(manifest["output_dimension"])
        file_name = (
            f"{dataset_name}__pairs__semantic_cohere_{model}__{input_type}"
            f"__dim{output_dimension}__cosine__min{float_slug(min_sim)}__label_restricted.json"
        )
    else:
        raise ValueError(f"Unsupported pair-cache backend: {backend}")
    return cache_root(repo_root) / "pairs" / file_name


def _pair_backend_config(
    *,
    backend: str,
    metric: str | None,
    min_sim: float,
    semantic_cache: Mapping[str, object] | None,
) -> dict:
    if backend == "edit":
        if not metric:
            raise ValueError("metric is required for edit pair cache generation.")
        return {
            "metric_family": "edit_distance",
            "metric": metric,
            "min_sim": float(min_sim),
            "label_restricted": True,
        }
    if backend == "semantic":
        if semantic_cache is None:
            raise ValueError("semantic_cache is required for semantic pair cache generation.")
        manifest = dict(semantic_cache["manifest"])
        return {
            "metric_family": "semantic_cosine",
            "provider": "cohere",
            "model": manifest["model"],
            "input_type": manifest["input_type"],
            "output_dimension": int(manifest["output_dimension"]),
            "min_sim": float(min_sim),
            "label_restricted": True,
        }
    raise ValueError(f"Unsupported pair-cache backend: {backend}")


def _explicit_span_counts(annotation) -> dict[str, int]:
    counts = Counter(
        span.label
        for span in annotation.spans
        if not span.is_implicit
    )
    return dict(sorted(counts.items()))
