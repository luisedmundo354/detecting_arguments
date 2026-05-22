"""Experimental label-blind explicit-span pairing for IAA diagnostics."""

from __future__ import annotations

if __package__ in {None, ""}:  # pragma: no cover - enables direct script execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "iaa_scores"

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from dataset_statistics.modules import AnnotationSpan
from span_iaa_ned import compute_optimal_matching, compute_optimal_matching_from_similarity_matrix

from . import DEFAULT_ANNOTATION_DIR
from .annotation_graphs import load_annotation_graphs
from .cache_utils import (
    cache_root,
    compute_dataset_fingerprint,
    dump_json,
    float_slug,
    load_json,
    repo_relative_path,
)
from .cohere_api import DEFAULT_INPUT_TYPE, DEFAULT_MODEL, DEFAULT_OUTPUT_DIMENSION
from .pair_cache import build_or_load_pair_cache, pair_cache_path
from .semantic_cache import load_semantic_embedding_cache, semantic_cache_paths


def build_or_load_label_blind_pair_cache(
    *,
    repo_root: Path,
    input_dir: Path,
    backend: str,
    overwrite: bool = False,
    metric: str | None = None,
    min_sim: float = 0.0,
    semantic_cache: Mapping[str, object] | None = None,
) -> dict:
    """Build or load a pair cache that can match spans across different labels."""

    input_dir = input_dir.resolve()
    dataset_fingerprint = compute_dataset_fingerprint(input_dir)
    cache_path = label_blind_pair_cache_path(
        repo_root=repo_root,
        input_dir=input_dir,
        backend=backend,
        metric=metric,
        min_sim=min_sim,
        semantic_cache=semantic_cache,
    )

    if cache_path.exists() and not overwrite:
        return load_label_blind_pair_cache(
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
    config = _backend_config(
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
                "Label-blind pair cache generation requires exactly two annotators "
                f"per document; ref_id {ref_id} has {len(annotators)} annotators."
            )

        record_a = records_by_annotator[annotators[0]]
        record_b = records_by_annotator[annotators[1]]
        spans_a = _explicit_spans(record_a.annotation.spans)
        spans_b = _explicit_spans(record_b.annotation.spans)

        if backend == "edit":
            matching = compute_optimal_matching(
                [span.text for span in spans_a],
                [span.text for span in spans_b],
                metric=str(metric),
                min_similarity=min_sim,
            )
        elif backend == "semantic":
            if semantic_cache is None:
                raise ValueError("semantic_cache is required for semantic label-blind pairs.")
            matching = _compute_semantic_matching(
                spans_a,
                spans_b,
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
                    annotators[0]: _explicit_span_counts(spans_a),
                    annotators[1]: _explicit_span_counts(spans_b),
                },
                "matches": [
                    _match_record(
                        spans_a[index_a],
                        spans_b[index_b],
                        similarity,
                        source_file_a=record_a.annotation.source_file,
                        source_file_b=record_b.annotation.source_file,
                    )
                    for index_a, index_b, similarity in matching.matches
                ],
                "unmatched_a": [
                    _unmatched_record(
                        spans_a[index],
                        source_file=record_a.annotation.source_file,
                    )
                    for index in matching.unmatched_indices_a
                ],
                "unmatched_b": [
                    _unmatched_record(
                        spans_b[index],
                        source_file=record_b.annotation.source_file,
                    )
                    for index in matching.unmatched_indices_b
                ],
            }
        )

    pair_cache = {
        "cache_version": 1,
        "dataset_path": repo_relative_path(input_dir, repo_root),
        "dataset_fingerprint": dataset_fingerprint,
        "backend": backend,
        "label_restricted": False,
        "config": config,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "documents": documents,
    }
    dump_json(pair_cache, cache_path)
    return pair_cache


def load_label_blind_pair_cache(
    *,
    repo_root: Path,
    input_dir: Path,
    backend: str,
    metric: str | None = None,
    min_sim: float = 0.0,
    semantic_cache: Mapping[str, object] | None = None,
) -> dict:
    """Load and validate a label-blind pair cache."""

    input_dir = input_dir.resolve()
    dataset_fingerprint = compute_dataset_fingerprint(input_dir)
    cache_path = label_blind_pair_cache_path(
        repo_root=repo_root,
        input_dir=input_dir,
        backend=backend,
        metric=metric,
        min_sim=min_sim,
        semantic_cache=semantic_cache,
    )
    if not cache_path.exists():
        raise FileNotFoundError(f"Label-blind pair cache not found at {cache_path}")

    pair_cache = load_json(cache_path)
    required_keys = {"dataset_path", "dataset_fingerprint", "backend", "config", "documents"}
    missing = sorted(required_keys - set(pair_cache))
    if missing:
        raise ValueError(f"Label-blind pair cache {cache_path} is missing keys: {missing}")
    if pair_cache["dataset_fingerprint"] != dataset_fingerprint:
        raise ValueError(
            "Label-blind pair cache fingerprint does not match the current dataset. "
            "Use --overwrite to rebuild it."
        )
    if pair_cache["backend"] != backend:
        raise ValueError(
            f"Label-blind pair cache backend mismatch: expected {backend}, "
            f"found {pair_cache['backend']}"
        )

    expected_config = _backend_config(
        backend=backend,
        metric=metric,
        min_sim=min_sim,
        semantic_cache=semantic_cache,
    )
    if pair_cache["config"] != expected_config:
        raise ValueError(
            f"Label-blind pair cache configuration mismatch for {cache_path}. "
            f"Expected {expected_config}, found {pair_cache['config']}."
        )
    if pair_cache.get("label_restricted") is not False:
        raise ValueError("Pair cache must be label_restricted=False for this experiment.")
    return pair_cache


def label_blind_pair_cache_path(
    *,
    repo_root: Path,
    input_dir: Path,
    backend: str,
    metric: str | None,
    min_sim: float,
    semantic_cache: Mapping[str, object] | None,
) -> Path:
    """Return the deterministic cache path for label-blind pair caches."""

    dataset_name = input_dir.name
    if backend == "edit":
        if not metric:
            raise ValueError("metric is required for the edit label-blind pair cache path.")
        file_name = (
            f"{dataset_name}__pairs__edit_{metric}__min{float_slug(min_sim)}"
            "__label_blind.json"
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
            f"__dim{output_dimension}__cosine__min{float_slug(min_sim)}"
            "__label_blind.json"
        )
    else:
        raise ValueError(f"Unsupported pair-cache backend: {backend}")
    return cache_root(repo_root) / "pairs" / file_name


def summarize_pair_cache_boundaries(pair_cache: Mapping[str, object]) -> dict:
    """Summarize aggregate soft-F1 from cached similarities while ignoring labels."""

    per_doc = {}
    totals = {
        "span_count_a": 0,
        "span_count_b": 0,
        "matched_count": 0,
        "soft_true_positive": 0.0,
    }
    for document_record in pair_cache.get("documents", []):
        doc_summary = _boundary_summary_for_document(document_record)
        per_doc[str(document_record.get("ref_id"))] = doc_summary
        totals["span_count_a"] += doc_summary["span_count_a"]
        totals["span_count_b"] += doc_summary["span_count_b"]
        totals["matched_count"] += doc_summary["matched_count"]
        totals["soft_true_positive"] += doc_summary["soft_true_positive"]

    return {
        "per_doc": per_doc,
        "overall": _finish_boundary_summary(totals),
    }


def summarize_label_blind_pair_cache(pair_cache: Mapping[str, object]) -> dict:
    """Return boundary-F1 plus cross-label diagnostics for a label-blind cache."""

    boundary = summarize_pair_cache_boundaries(pair_cache)
    cross_label_count = 0
    match_count = 0
    confusion: dict[str, dict[str, int]] = {}
    for document_record in pair_cache.get("documents", []):
        for match in document_record.get("matches", []):
            label_a = str(match.get("label_a"))
            label_b = str(match.get("label_b"))
            confusion.setdefault(label_a, {})
            confusion[label_a][label_b] = confusion[label_a].get(label_b, 0) + 1
            match_count += 1
            if label_a != label_b:
                cross_label_count += 1

    cross_label_rate = (cross_label_count / match_count) if match_count else float("nan")
    return {
        "boundary": boundary,
        "cross_label": {
            "match_count": match_count,
            "cross_label_count": cross_label_count,
            "cross_label_rate": cross_label_rate,
        },
        "label_confusion": {
            label_a: dict(sorted(label_counts.items()))
            for label_a, label_counts in sorted(confusion.items())
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run experimental label-blind explicit-span pairing diagnostics.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR,
        help="Directory with repaired IAA Label Studio JSON exports.",
    )
    parser.add_argument(
        "--backend",
        choices=("edit", "semantic", "both"),
        default="both",
        help="Pairing backend to run.",
    )
    parser.add_argument(
        "--metric",
        default="yujianbo",
        choices=("yujianbo", "higueramico"),
        help="Normalized edit-distance metric for edit pairing.",
    )
    parser.add_argument(
        "--edit-min-sim",
        type=float,
        default=0.1,
        help="Minimum edit similarity for a label-blind edit match.",
    )
    parser.add_argument(
        "--semantic-min-sim",
        type=float,
        default=0.0,
        help="Minimum cosine similarity for a label-blind semantic match.",
    )
    parser.add_argument(
        "--semantic-model",
        default=DEFAULT_MODEL,
        help="Cohere embedding model used by the existing semantic cache.",
    )
    parser.add_argument(
        "--semantic-input-type",
        default=DEFAULT_INPUT_TYPE,
        help="Cohere embedding input_type used by the existing semantic cache.",
    )
    parser.add_argument(
        "--semantic-output-dimension",
        type=int,
        default=DEFAULT_OUTPUT_DIMENSION,
        help="Cohere embedding dimension used by the existing semantic cache.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild label-blind pair caches instead of loading existing ones.",
    )
    parser.add_argument(
        "--allow-stale-semantic-cache",
        action="store_true",
        help=(
            "Load an existing semantic embedding cache even if its dataset fingerprint "
            "does not match the current input directory. Exact missing span texts will "
            "still fail during semantic matching."
        ),
    )
    parser.add_argument(
        "--no-restricted-comparison",
        action="store_true",
        help="Only report label-blind scores; skip existing label-restricted comparison.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"--input-dir {input_dir} is not a directory")

    backends = ("edit", "semantic") if args.backend == "both" else (args.backend,)
    semantic_cache = (
        _load_required_semantic_cache(repo_root, input_dir, args)
        if "semantic" in backends
        else None
    )
    for backend in backends:
        if backend == "edit":
            _run_edit_backend(repo_root, input_dir, args)
        elif backend == "semantic":
            _run_semantic_backend(repo_root, input_dir, args, semantic_cache=semantic_cache)
        else:  # pragma: no cover - argparse constrains this
            raise SystemExit(f"Unsupported backend: {backend}")


def _run_edit_backend(repo_root: Path, input_dir: Path, args: argparse.Namespace) -> None:
    label_blind_cache = build_or_load_label_blind_pair_cache(
        repo_root=repo_root,
        input_dir=input_dir,
        backend="edit",
        overwrite=args.overwrite,
        metric=args.metric,
        min_sim=args.edit_min_sim,
    )
    label_blind_path = label_blind_pair_cache_path(
        repo_root=repo_root,
        input_dir=input_dir,
        backend="edit",
        metric=args.metric,
        min_sim=args.edit_min_sim,
        semantic_cache=None,
    )
    restricted_cache = None
    restricted_path = None
    if not args.no_restricted_comparison:
        restricted_cache = build_or_load_pair_cache(
            repo_root=repo_root,
            input_dir=input_dir,
            backend="edit",
            overwrite=args.overwrite,
            metric=args.metric,
            min_sim=args.edit_min_sim,
        )
        restricted_path = pair_cache_path(
            repo_root=repo_root,
            input_dir=input_dir,
            backend="edit",
            metric=args.metric,
            min_sim=args.edit_min_sim,
            semantic_cache=None,
        )
    _print_report(
        title="Edit-distance pairs",
        score_label="Boundary/text soft-F1",
        label_blind_cache=label_blind_cache,
        label_blind_path=label_blind_path,
        restricted_cache=restricted_cache,
        restricted_path=restricted_path,
    )


def _run_semantic_backend(
    repo_root: Path,
    input_dir: Path,
    args: argparse.Namespace,
    *,
    semantic_cache: Mapping[str, object] | None,
) -> None:
    if semantic_cache is None:
        semantic_cache = _load_required_semantic_cache(repo_root, input_dir, args)
    label_blind_cache = build_or_load_label_blind_pair_cache(
        repo_root=repo_root,
        input_dir=input_dir,
        backend="semantic",
        overwrite=args.overwrite,
        min_sim=args.semantic_min_sim,
        semantic_cache=semantic_cache,
    )
    label_blind_path = label_blind_pair_cache_path(
        repo_root=repo_root,
        input_dir=input_dir,
        backend="semantic",
        metric=None,
        min_sim=args.semantic_min_sim,
        semantic_cache=semantic_cache,
    )
    restricted_cache = None
    restricted_path = None
    if not args.no_restricted_comparison:
        restricted_cache = build_or_load_pair_cache(
            repo_root=repo_root,
            input_dir=input_dir,
            backend="semantic",
            overwrite=args.overwrite,
            min_sim=args.semantic_min_sim,
            semantic_cache=semantic_cache,
        )
        restricted_path = pair_cache_path(
            repo_root=repo_root,
            input_dir=input_dir,
            backend="semantic",
            metric=None,
            min_sim=args.semantic_min_sim,
            semantic_cache=semantic_cache,
        )
    _print_report(
        title="Semantic cosine pairs",
        score_label="Semantic cosine soft-F1",
        label_blind_cache=label_blind_cache,
        label_blind_path=label_blind_path,
        restricted_cache=restricted_cache,
        restricted_path=restricted_path,
    )


def _load_required_semantic_cache(
    repo_root: Path,
    input_dir: Path,
    args: argparse.Namespace,
) -> dict:
    try:
        return load_semantic_embedding_cache(
            repo_root=repo_root,
            input_dir=input_dir,
            model=args.semantic_model,
            input_type=args.semantic_input_type,
            output_dimension=args.semantic_output_dimension,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "Semantic embedding cache was not found. This experimental script does not "
            "rebuild embeddings by default; run with --backend edit or create the semantic "
            "cache through the existing IAA workflow first."
        ) from exc
    except ValueError as exc:
        if "fingerprint does not match" not in str(exc) or not args.allow_stale_semantic_cache:
            raise
        return _load_stale_semantic_embedding_cache(repo_root, input_dir, args)


def _load_stale_semantic_embedding_cache(
    repo_root: Path,
    input_dir: Path,
    args: argparse.Namespace,
) -> dict:
    npz_path, manifest_path = semantic_cache_paths(
        repo_root=repo_root,
        input_dir=input_dir,
        model=args.semantic_model,
        input_type=args.semantic_input_type,
        output_dimension=args.semantic_output_dimension,
    )
    if not npz_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"Semantic cache not found. Expected {npz_path} and {manifest_path}."
        )

    manifest = load_json(manifest_path)
    required_keys = {"model", "input_type", "output_dimension", "row_count", "text_to_row"}
    missing = sorted(required_keys - set(manifest))
    if missing:
        raise ValueError(f"Semantic embedding manifest is missing keys: {missing}")
    if manifest["model"] != args.semantic_model or manifest["input_type"] != args.semantic_input_type:
        raise ValueError(
            "Semantic embedding cache configuration does not match the requested model/input_type."
        )
    if int(manifest["output_dimension"]) != int(args.semantic_output_dimension):
        raise ValueError(
            "Semantic embedding cache output dimension does not match the requested value."
        )

    npz = np.load(npz_path)
    if "embeddings" not in npz:
        raise ValueError(f"Semantic embedding cache at {npz_path} is missing the 'embeddings' array.")
    embeddings = np.asarray(npz["embeddings"], dtype=np.float32)
    if embeddings.shape != (int(manifest["row_count"]), int(args.semantic_output_dimension)):
        raise ValueError(
            "Semantic embedding matrix shape does not match manifest row_count/output_dimension."
        )

    text_to_row = {
        str(text): int(index)
        for text, index in dict(manifest["text_to_row"]).items()
    }
    text_to_embedding = {
        text: embeddings[row_index]
        for text, row_index in text_to_row.items()
    }
    current_fingerprint = compute_dataset_fingerprint(input_dir)
    print(
        "WARNING: using a stale semantic embedding cache. "
        f"manifest fingerprint={manifest.get('dataset_fingerprint')}; "
        f"current fingerprint={current_fingerprint}. "
        "Only exact span texts already present in the cache can be matched."
    )
    return {
        "npz_path": npz_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "embeddings": embeddings,
        "text_to_embedding": text_to_embedding,
    }


def _print_report(
    *,
    title: str,
    score_label: str,
    label_blind_cache: Mapping[str, object],
    label_blind_path: Path,
    restricted_cache: Mapping[str, object] | None,
    restricted_path: Path | None,
) -> None:
    label_blind_summary = summarize_label_blind_pair_cache(label_blind_cache)
    label_blind_boundary = label_blind_summary["boundary"]["overall"]
    cross_label = label_blind_summary["cross_label"]

    print(f"\n=== {title}: label-blind diagnostic ===")
    print(f"Label-blind cache: {label_blind_path}")
    print(
        f"Label-blind {score_label}: "
        f"{_format_float(label_blind_boundary['f1'])} "
        f"(P={_format_float(label_blind_boundary['precision'])}, "
        f"R={_format_float(label_blind_boundary['recall'])}, "
        f"matches={label_blind_boundary['matched_count']})"
    )
    print(
        "Cross-label matches: "
        f"{cross_label['cross_label_count']} / {cross_label['match_count']} "
        f"({_format_float(cross_label['cross_label_rate'])})"
    )

    if restricted_cache is not None and restricted_path is not None:
        restricted_boundary = summarize_pair_cache_boundaries(restricted_cache)["overall"]
        print(f"Label-restricted cache: {restricted_path}")
        print(
            f"Label-restricted {score_label}: "
            f"{_format_float(restricted_boundary['f1'])} "
            f"(P={_format_float(restricted_boundary['precision'])}, "
            f"R={_format_float(restricted_boundary['recall'])}, "
            f"matches={restricted_boundary['matched_count']})"
        )

    print("Label confusion from matched pairs:")
    _print_confusion(label_blind_summary["label_confusion"])


def _compute_semantic_matching(
    spans_a: Sequence[AnnotationSpan],
    spans_b: Sequence[AnnotationSpan],
    *,
    text_to_embedding: Mapping[str, np.ndarray],
    min_sim: float,
):
    similarity_matrix = np.zeros((len(spans_a), len(spans_b)), dtype=float)
    for index_a, span_a in enumerate(spans_a):
        embedding_a = _embedding_for_text(span_a.text, text_to_embedding)
        for index_b, span_b in enumerate(spans_b):
            embedding_b = _embedding_for_text(span_b.text, text_to_embedding)
            similarity = _cosine_similarity(embedding_a, embedding_b)
            similarity_matrix[index_a, index_b] = similarity if similarity >= min_sim else 0.0
    return compute_optimal_matching_from_similarity_matrix(similarity_matrix)


def _boundary_summary_for_document(document_record: Mapping[str, object]) -> dict:
    annotators = document_record.get("annotators")
    if not isinstance(annotators, list) or len(annotators) != 2:
        raise ValueError("Pair cache document record must contain exactly two annotators.")

    counts_by_annotator = document_record.get("span_counts_by_annotator")
    if not isinstance(counts_by_annotator, dict):
        raise ValueError("Pair cache document record is missing span_counts_by_annotator.")

    annotator_a, annotator_b = annotators
    label_counts_a = counts_by_annotator.get(annotator_a)
    label_counts_b = counts_by_annotator.get(annotator_b)
    if not isinstance(label_counts_a, dict) or not isinstance(label_counts_b, dict):
        raise ValueError("Pair cache document record has malformed span counts.")

    totals = {
        "span_count_a": sum(int(count) for count in label_counts_a.values()),
        "span_count_b": sum(int(count) for count in label_counts_b.values()),
        "matched_count": 0,
        "soft_true_positive": 0.0,
    }
    for match in document_record.get("matches", []):
        similarity = float(match.get("similarity", 0.0))
        if similarity < 0.0:
            raise ValueError("Pair cache match similarity cannot be negative.")
        totals["matched_count"] += 1
        totals["soft_true_positive"] += similarity
    return _finish_boundary_summary(totals)


def _finish_boundary_summary(totals: Mapping[str, float]) -> dict:
    span_count_a = int(totals["span_count_a"])
    span_count_b = int(totals["span_count_b"])
    soft_tp = float(totals["soft_true_positive"])
    precision = soft_tp / span_count_a if span_count_a else float("nan")
    recall = soft_tp / span_count_b if span_count_b else float("nan")
    f1 = (
        (2.0 * precision * recall) / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "span_count_a": span_count_a,
        "span_count_b": span_count_b,
        "matched_count": int(totals["matched_count"]),
        "soft_true_positive": soft_tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _backend_config(
    *,
    backend: str,
    metric: str | None,
    min_sim: float,
    semantic_cache: Mapping[str, object] | None,
) -> dict:
    if backend == "edit":
        if not metric:
            raise ValueError("metric is required for edit label-blind pair generation.")
        return {
            "metric_family": "edit_distance",
            "metric": metric,
            "min_sim": float(min_sim),
            "label_restricted": False,
        }
    if backend == "semantic":
        if semantic_cache is None:
            raise ValueError("semantic_cache is required for semantic label-blind pairs.")
        manifest = dict(semantic_cache["manifest"])
        return {
            "metric_family": "semantic_cosine",
            "provider": "cohere",
            "model": manifest["model"],
            "input_type": manifest["input_type"],
            "output_dimension": int(manifest["output_dimension"]),
            "min_sim": float(min_sim),
            "label_restricted": False,
        }
    raise ValueError(f"Unsupported pair-cache backend: {backend}")


def _explicit_spans(spans: Iterable[AnnotationSpan]) -> list[AnnotationSpan]:
    explicit = [span for span in spans if not span.is_implicit]
    return sorted(
        explicit,
        key=lambda span: (
            _offset_sort_value(span.start),
            _offset_sort_value(span.end),
            span.node_id,
        ),
    )


def _offset_sort_value(value: int | None) -> int:
    return 10**18 if value is None else value


def _explicit_span_counts(spans: Iterable[AnnotationSpan]) -> dict[str, int]:
    return dict(sorted(Counter(span.label for span in spans).items()))


def _match_record(
    span_a: AnnotationSpan,
    span_b: AnnotationSpan,
    similarity: float,
    *,
    source_file_a: str,
    source_file_b: str,
) -> dict:
    return {
        "label_a": span_a.label,
        "label_b": span_b.label,
        "labels_match": span_a.label == span_b.label,
        "node_id_a": span_a.node_id,
        "node_id_b": span_b.node_id,
        "text_a": span_a.text,
        "text_b": span_b.text,
        "similarity": similarity,
        "source_file_a": source_file_a,
        "source_file_b": source_file_b,
    }


def _unmatched_record(span: AnnotationSpan, *, source_file: str) -> dict:
    return {
        "label": span.label,
        "node_id": span.node_id,
        "text": span.text,
        "source_file": source_file,
    }


def _embedding_for_text(text: str, text_to_embedding: Mapping[str, np.ndarray]) -> np.ndarray:
    try:
        return np.asarray(text_to_embedding[text], dtype=float)
    except KeyError as exc:
        raise KeyError(f"Missing semantic embedding for text: {text!r}") from exc


def _cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    if vector_a.shape != vector_b.shape:
        raise ValueError(f"Semantic embedding shape mismatch: {vector_a.shape} vs {vector_b.shape}")
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        raise ValueError("Semantic embeddings must have non-zero norm for cosine similarity.")
    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def _print_confusion(confusion: Mapping[str, Mapping[str, int]]) -> None:
    labels = sorted(set(confusion) | {label for row in confusion.values() for label in row})
    if not labels:
        print("\t(no matches)")
        return
    header = "\tA\\B\t" + "\t".join(labels)
    print(header)
    for label_a in labels:
        row = confusion.get(label_a, {})
        counts = "\t".join(str(int(row.get(label_b, 0))) for label_b in labels)
        print(f"\t{label_a}\t{counts}")


def _format_float(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.4f}"


if __name__ == "__main__":  # pragma: no cover
    main()
