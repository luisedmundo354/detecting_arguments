"""Reusable span alignment for graph-based IAA metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from dataset_statistics.modules import AnnotationSpan

from span_iaa_ned import (
    MatchingResult,
    compute_optimal_matching,
    compute_optimal_matching_from_similarity_matrix,
)


@dataclass(frozen=True)
class SpanMatch:
    """Aligned explicit spans from two annotators."""

    label: str
    span_a: AnnotationSpan
    span_b: AnnotationSpan
    similarity: float


@dataclass(frozen=True)
class SpanAlignment:
    """Full 1:1 alignment metadata for two explicit-span sets."""

    matches: List[SpanMatch]
    unmatched_a: List[AnnotationSpan]
    unmatched_b: List[AnnotationSpan]


def align_explicit_spans(
    spans_a: Sequence[AnnotationSpan],
    spans_b: Sequence[AnnotationSpan],
    *,
    metric: str,
    min_sim: float,
) -> SpanAlignment:
    """Align explicit spans label-by-label using the span soft-F1 matcher."""

    explicit_a = [span for span in spans_a if not span.is_implicit]
    explicit_b = [span for span in spans_b if not span.is_implicit]
    return _align_grouped_explicit_spans(
        explicit_a,
        explicit_b,
        matcher=lambda label_spans_a, label_spans_b: compute_optimal_matching(
            [span.text for span in label_spans_a],
            [span.text for span in label_spans_b],
            metric=metric,
            min_similarity=min_sim,
        ),
    )


def align_explicit_spans_semantic(
    spans_a: Sequence[AnnotationSpan],
    spans_b: Sequence[AnnotationSpan],
    *,
    text_to_embedding: Mapping[str, np.ndarray],
    min_sim: float,
) -> SpanAlignment:
    """Align explicit spans label-by-label using cosine similarity on embeddings."""

    explicit_a = [span for span in spans_a if not span.is_implicit]
    explicit_b = [span for span in spans_b if not span.is_implicit]
    return _align_grouped_explicit_spans(
        explicit_a,
        explicit_b,
        matcher=lambda label_spans_a, label_spans_b: _semantic_matching_for_label(
            label_spans_a,
            label_spans_b,
            text_to_embedding=text_to_embedding,
            min_sim=min_sim,
        ),
    )


def _align_grouped_explicit_spans(
    explicit_a: Sequence[AnnotationSpan],
    explicit_b: Sequence[AnnotationSpan],
    *,
    matcher,
) -> SpanAlignment:
    spans_by_label_a = _group_by_label(explicit_a)
    spans_by_label_b = _group_by_label(explicit_b)
    all_labels = sorted(set(spans_by_label_a) | set(spans_by_label_b))

    matches: List[SpanMatch] = []
    unmatched_a: List[AnnotationSpan] = []
    unmatched_b: List[AnnotationSpan] = []

    for label in all_labels:
        label_spans_a = spans_by_label_a.get(label, [])
        label_spans_b = spans_by_label_b.get(label, [])
        matching = matcher(label_spans_a, label_spans_b)
        matches.extend(_materialize_matches(label, label_spans_a, label_spans_b, matching))
        unmatched_a.extend(label_spans_a[index] for index in matching.unmatched_indices_a)
        unmatched_b.extend(label_spans_b[index] for index in matching.unmatched_indices_b)

    return SpanAlignment(matches=matches, unmatched_a=unmatched_a, unmatched_b=unmatched_b)


def _group_by_label(spans: Iterable[AnnotationSpan]) -> Dict[str, List[AnnotationSpan]]:
    grouped: Dict[str, List[AnnotationSpan]] = {}
    for span in spans:
        grouped.setdefault(span.label, []).append(span)
    for label_spans in grouped.values():
        label_spans.sort(key=lambda span: (span.start or 10**18, span.end or 10**18, span.node_id))
    return grouped


def _materialize_matches(
    label: str,
    spans_a: Sequence[AnnotationSpan],
    spans_b: Sequence[AnnotationSpan],
    matching: MatchingResult,
) -> List[SpanMatch]:
    return [
        SpanMatch(
            label=label,
            span_a=spans_a[index_a],
            span_b=spans_b[index_b],
            similarity=similarity,
        )
        for index_a, index_b, similarity in matching.matches
    ]


def _semantic_matching_for_label(
    spans_a: Sequence[AnnotationSpan],
    spans_b: Sequence[AnnotationSpan],
    *,
    text_to_embedding: Mapping[str, np.ndarray],
    min_sim: float,
) -> MatchingResult:
    similarity_matrix = np.zeros((len(spans_a), len(spans_b)), dtype=float)
    for index_a, span_a in enumerate(spans_a):
        embedding_a = _embedding_for_text(span_a.text, text_to_embedding)
        for index_b, span_b in enumerate(spans_b):
            embedding_b = _embedding_for_text(span_b.text, text_to_embedding)
            similarity = _cosine_similarity(embedding_a, embedding_b)
            similarity_matrix[index_a, index_b] = similarity if similarity >= min_sim else 0.0
    return compute_optimal_matching_from_similarity_matrix(similarity_matrix)


def _embedding_for_text(text: str, text_to_embedding: Mapping[str, np.ndarray]) -> np.ndarray:
    try:
        return np.asarray(text_to_embedding[text], dtype=float)
    except KeyError as exc:
        raise KeyError(f"Missing semantic embedding for text: {text!r}") from exc


def _cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    if vector_a.shape != vector_b.shape:
        raise ValueError(
            f"Semantic embedding shape mismatch: {vector_a.shape} vs {vector_b.shape}"
        )
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        raise ValueError("Semantic embeddings must have non-zero norm for cosine similarity.")
    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
