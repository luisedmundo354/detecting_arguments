"""Span-level F1 helpers built on top of the existing span_iaa_ned module."""

from __future__ import annotations

from typing import Dict, Iterable

from span_iaa_ned import compute_iaa


def compute_f1_for_document(
    annotations: Dict[str, Dict[str, list]],
    categories: Iterable[str],
    *,
    metric: str = "yujianbo",
    min_sim: float = 0.1,
) -> Dict[str, float]:
    """Compute mean pairwise F1 for a single document."""

    return compute_iaa(
        annotations,
        list(categories),
        metric=metric,
        min_sim=min_sim,
    )


def micro_average_f1(
    per_doc_f1: Dict[object, Dict[str, float]],
    pair_counts: Dict[object, int],
    categories: Iterable[str],
) -> Dict[str, float]:
    """Micro-average F1 across documents weighted by annotator pairs."""

    results: Dict[str, float] = {}
    for cat in categories:
        numerator = 0.0
        denominator = 0
        for ref_id, scores in per_doc_f1.items():
            pairs = pair_counts.get(ref_id, 0)
            if pairs <= 0:
                continue
            score = scores.get(cat)
            if score is None:
                continue
            numerator += score * pairs
            denominator += pairs
        results[cat] = (numerator / denominator) if denominator else float("nan")
    return results
