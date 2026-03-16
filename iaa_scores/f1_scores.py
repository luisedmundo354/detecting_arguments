"""Span-level F1 helpers built on top of the existing span_iaa_ned module."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

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


def compute_f1_for_document_from_pairs(
    document_record: Mapping[str, object],
    categories: Iterable[str],
) -> Dict[str, float]:
    """Compute per-label soft F1 from a persisted pair cache document record."""

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

    soft_tp_by_label: Dict[str, float] = {}
    for match in document_record.get("matches", []):
        label = match.get("label")
        similarity = float(match.get("similarity", 0.0))
        if similarity < 0.0:
            raise ValueError("Pair cache match similarity cannot be negative.")
        soft_tp_by_label[label] = soft_tp_by_label.get(label, 0.0) + similarity

    scores: Dict[str, float] = {}
    for category in categories:
        count_a = int(label_counts_a.get(category, 0))
        count_b = int(label_counts_b.get(category, 0))
        if count_a == 0 and count_b == 0:
            scores[category] = 1.0
            continue
        if count_a == 0 or count_b == 0:
            scores[category] = 0.0
            continue

        soft_tp = float(soft_tp_by_label.get(category, 0.0))
        precision = soft_tp / count_a
        recall = soft_tp / count_b
        scores[category] = (
            (2.0 * precision * recall) / (precision + recall)
            if (precision + recall) > 0.0
            else 0.0
        )
    return scores


def compute_f1_from_pair_cache(
    pair_cache: Mapping[str, object],
    categories: Iterable[str],
) -> tuple[Dict[object, Dict[str, float]], Dict[object, int], Dict[str, float]]:
    """Compute per-document and overall F1 from a persisted pair cache."""

    per_doc_f1: Dict[object, Dict[str, float]] = {}
    pair_counts: Dict[object, int] = {}
    for document_record in pair_cache.get("documents", []):
        ref_id = document_record.get("ref_id")
        per_doc_f1[ref_id] = compute_f1_for_document_from_pairs(document_record, categories)
        pair_counts[ref_id] = 1
    overall_f1 = micro_average_f1(per_doc_f1, pair_counts, categories)
    return per_doc_f1, pair_counts, overall_f1
