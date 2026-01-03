"""Krippendorff alpha utilities for both nominal and unitizing modes."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import krippendorff

from .models import DocumentOffsets, DocumentSpans

AnnotationsByAnnotator = Dict[str, Dict[str, List[str]]]
OffsetsByAnnotator = Dict[str, Dict[str, List[Tuple[int, int]]]]


def compute_alpha_per_doc(
    all_annotations_for_doc: AnnotationsByAnnotator,
    categories: Iterable[str],
) -> Dict[str, float]:
    """Nominal Krippendorff alpha per category for a single document."""

    alpha_by_cat: Dict[str, float] = {}
    annotators = sorted(all_annotations_for_doc.keys())

    for cat in categories:
        units = sorted(
            {
                span
                for ann in all_annotations_for_doc.values()
                for span in ann.get(cat, [])
            }
        )
        if not units:
            alpha_by_cat[cat] = 1.0
            continue

        reliability = []
        for annotator in annotators:
            spans = set(all_annotations_for_doc.get(annotator, {}).get(cat, []))
            reliability.append([1.0 if unit in spans else 0.0 for unit in units])

        matrix = np.array(reliability, dtype=float)
        vals = np.unique(matrix[~np.isnan(matrix)])
        if vals.size <= 1 or len(annotators) < 2:
            alpha_by_cat[cat] = 1.0 if vals.size == 1 else float("nan")
            continue

        alpha = krippendorff.alpha(
            reliability_data=matrix,
            level_of_measurement="nominal",
        )
        alpha_by_cat[cat] = float(alpha)

    return alpha_by_cat


def compute_alpha_overall(
    grouped_by_doc: Dict[object, DocumentSpans],
    categories: Iterable[str],
) -> Dict[str, float]:
    """Nominal Krippendorff alpha across all documents."""

    alpha_by_cat: Dict[str, float] = {}
    annotators = sorted({a for doc in grouped_by_doc.values() for a in doc.keys()})

    for cat in categories:
        units = []
        seen = set()
        for ref_id, ann_by_annot in grouped_by_doc.items():
            for ann in ann_by_annot.values():
                for span in ann.get(cat, []):
                    key = (ref_id, span)
                    if key not in seen:
                        seen.add(key)
                        units.append(key)

        if not units:
            alpha_by_cat[cat] = float("nan")
            continue

        reliability = []
        for annotator in annotators:
            row = []
            for ref_id, span in units:
                if annotator not in grouped_by_doc.get(ref_id, {}):
                    row.append(np.nan)
                else:
                    spans = set(grouped_by_doc[ref_id][annotator].get(cat, []))
                    row.append(1.0 if span in spans else 0.0)
            reliability.append(row)

        alpha = krippendorff.alpha(
            reliability_data=np.array(reliability, dtype=float),
            level_of_measurement="nominal",
        )
        alpha_by_cat[cat] = float(alpha)

    return alpha_by_cat


def compute_alpha_u_per_doc(
    all_offsets_for_doc: OffsetsByAnnotator,
    categories: Iterable[str],
    continuum_len: int | None = None,
    include_background: bool = False,
) -> Dict[str, float]:
    """Krippendorff alpha for unitizing (alpha_u) per document."""

    if len(all_offsets_for_doc) < 2:
        return {cat: float("nan") for cat in categories}

    max_end = 0
    for per_ann in all_offsets_for_doc.values():
        for spans in per_ann.values():
            for start, end in spans:
                max_end = max(max_end, end)
    if continuum_len is None:
        continuum_len = max_end
    else:
        continuum_len = max(continuum_len, max_end)

    results = {}
    for cat in categories:
        iv_by_coder = {
            coder: all_offsets_for_doc.get(coder, {}).get(cat, [])
            for coder in all_offsets_for_doc
        }
        coincidence = _coincidence_lengths_binary(
            iv_by_coder,
            continuum_len,
            include_background=include_background,
        )
        results[cat] = _alpha_from_coincidence(coincidence)

    return results


def compute_alpha_u_overall(
    grouped_offsets_by_doc: Dict[object, DocumentOffsets],
    categories: Iterable[str],
    include_background: bool = False,
    doc_lengths: Dict[object, int] | None = None,
) -> Dict[str, float]:
    """Corpus-level alpha_u by summing coincidence matrices across docs."""

    overall = {}
    for cat in categories:
        coincidence_sum = np.zeros((2, 2), dtype=float)
        for ref_id, ann_by_annot in grouped_offsets_by_doc.items():
            continuum_len = 0
            for coder_offsets in ann_by_annot.values():
                for start, end in coder_offsets.get(cat, []):
                    continuum_len = max(continuum_len, end)
            if doc_lengths and ref_id in doc_lengths:
                continuum_len = max(continuum_len, doc_lengths[ref_id])
            iv_by_coder = {
                coder: ann_by_annot.get(coder, {}).get(cat, [])
                for coder in ann_by_annot
            }
            coincidence_sum += _coincidence_lengths_binary(
                iv_by_coder,
                continuum_len,
                include_background=include_background,
            )
        overall[cat] = _alpha_from_coincidence(coincidence_sum)

    return overall


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _boundaries(intervals_by_coder: Dict[str, List[Tuple[int, int]]], continuum_len: int) -> List[int]:
    boundaries = {0, continuum_len}
    for intervals in intervals_by_coder.values():
        for start, end in intervals:
            boundaries.add(start)
            boundaries.add(end)
    return sorted(boundaries)


def _covers(intervals: List[Tuple[int, int]], position: int) -> int:
    for start, end in intervals:
        if start <= position < end:
            return 1
        if position < start:
            return 0
    return 0


def _coincidence_lengths_binary(
    intervals_by_coder: OffsetsByAnnotator,
    continuum_len: int,
    include_background: bool = False,
) -> np.ndarray:
    merged = {coder: _merge_intervals(intervals) for coder, intervals in intervals_by_coder.items()}
    coders = sorted(merged)
    boundaries = _boundaries(merged, continuum_len)

    coincidence = np.zeros((2, 2), dtype=float)
    for i in range(len(coders)):
        for j in range(i + 1, len(coders)):
            coder_i = merged[coders[i]]
            coder_j = merged[coders[j]]
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                if start >= end:
                    continue
                cover_i = _covers(coder_i, start)
                cover_j = _covers(coder_j, start)
                if include_background or cover_i == 1 or cover_j == 1:
                    length = end - start
                    coincidence[cover_i, cover_j] += length
                    coincidence[cover_j, cover_i] += length
    return coincidence


def _alpha_from_coincidence(coincidence: np.ndarray) -> float:
    n = float(coincidence.sum())
    if n == 0:
        return 1.0
    marginals = coincidence.sum(axis=1)
    expected = np.empty_like(coincidence, dtype=float)
    denom = max(n - 1.0, 1.0)
    for v in range(coincidence.shape[0]):
        for w in range(coincidence.shape[1]):
            if v == w:
                expected[v, w] = (marginals[v] * max(marginals[v] - 1.0, 0.0)) / denom
            else:
                expected[v, w] = (marginals[v] * marginals[w]) / denom

    Do = float(coincidence[np.eye(coincidence.shape[0]) == 0].sum())
    De = float(expected[np.eye(expected.shape[0]) == 0].sum())
    if De == 0:
        return 1.0
    return 1.0 - (Do / De)
