"""Span-level IAA using pairwise soft-F1 with normalized edit distance."""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# Distance functions that return values in [0, 1].
from abydos.distance import YujianBo, HigueraMico


_YUJIANBO_DISTANCE = YujianBo()
_HIGUERAMICO_DISTANCE = HigueraMico()

def _yujianbo_distance(span_text_a: str, span_text_b: str) -> float:
    """Distance in [0, 1] between two span strings."""

    return _YUJIANBO_DISTANCE.dist(span_text_a, span_text_b)


def _higueramico_distance(span_text_a: str, span_text_b: str) -> float:
    """Distance in [0, 1] between two span strings."""

    return _HIGUERAMICO_DISTANCE.dist(span_text_a, span_text_b)

# Supported distance function lookup.
_DISTANCE_FN_BY_NAME: Dict[str, Callable[[str, str], float]] = {
    "yujianbo": _yujianbo_distance,
    "higueramico": _higueramico_distance,
}


def _similarity_matrix(
    span_texts_a: List[str],
    span_texts_b: List[str],
    distance_fn: Callable[[str, str], float],
    min_similarity: float = 0.0,
) -> np.ndarray:
    """Return similarity scores for every (span_a, span_b) pair."""

    span_text_count_a = len(span_texts_a)
    span_text_count_b = len(span_texts_b)
    if span_text_count_a == 0 or span_text_count_b == 0:
        return np.zeros((span_text_count_a, span_text_count_b), dtype=float)

    similarity_matrix = np.empty((span_text_count_a, span_text_count_b), dtype=float)
    for span_index_a, span_text_a in enumerate(span_texts_a):
        for span_index_b, span_text_b in enumerate(span_texts_b):
            distance = float(distance_fn(span_text_a, span_text_b))
            similarity = 1.0 - distance
            similarity_matrix[span_index_a, span_index_b] = (
                similarity if similarity >= min_similarity else 0.0
            )
    return similarity_matrix


def _soft_tp_from_matching(similarity_matrix: np.ndarray) -> float:
    """Return soft TP as the sum of similarities for the best 1-1 matches."""

    span_text_count_a, span_text_count_b = similarity_matrix.shape
    if span_text_count_a == 0 and span_text_count_b == 0:
        return 0.0

    padded_size = max(span_text_count_a, span_text_count_b)
    padded_similarity_matrix = np.zeros((padded_size, padded_size), dtype=float)
    padded_similarity_matrix[:span_text_count_a, :span_text_count_b] = similarity_matrix

    # Indices of best 1-1 matches (includes padded rows/cols -> unmatched spans).
    matched_row_indices, matched_col_indices = linear_sum_assignment(-padded_similarity_matrix)
    soft_tp_sum = float(padded_similarity_matrix[matched_row_indices, matched_col_indices].sum())
    return soft_tp_sum


def _pairwise_prf1(
    span_texts_a: List[str],
    span_texts_b: List[str],
    distance_fn: Callable[[str, str], float],
    min_similarity: float = 0.0,
) -> Tuple[float, float, float]:
    """Return precision, recall, F1 for one annotator pair and one category."""

    span_text_count_a = len(span_texts_a)
    span_text_count_b = len(span_texts_b)

    # Both annotators have no spans for this category.
    if span_text_count_a == 0 and span_text_count_b == 0:
        return 1.0, 1.0, 1.0
    # Only one annotator has spans for this category.
    if span_text_count_a == 0 or span_text_count_b == 0:
        return 0.0, 0.0, 0.0

    similarity_matrix = _similarity_matrix(
        span_texts_a,
        span_texts_b,
        distance_fn,
        min_similarity=min_similarity,
    )
    soft_tp_sum = _soft_tp_from_matching(similarity_matrix)

    precision = soft_tp_sum / span_text_count_a
    recall = soft_tp_sum / span_text_count_b
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2.0 * precision * recall) / (precision + recall)
    return precision, recall, f1


def compute_iaa(
    annotations: Dict[str, Dict[str, List[str]]],
    categories: List[str],
    *,
    metric: str = "yujianbo",
    min_sim: float = 0.0,
) -> Dict[str, float]:
    """Return mean pairwise F1 per category across annotator pairs."""

    distance_name = metric
    if distance_name not in _DISTANCE_FN_BY_NAME:
        raise ValueError(
            f"Unsupported metric '{distance_name}'. Choose from {sorted(_DISTANCE_FN_BY_NAME.keys())}"
        )
    distance_fn = _DISTANCE_FN_BY_NAME[distance_name]

    annotator_ids = list(annotations)
    annotator_id_pairs = list(itertools.combinations(annotator_ids, 2))

    pairwise_f1_values_by_category_name: Dict[str, List[float]] = {
        category_name: [] for category_name in categories
    }

    for annotator_id_a, annotator_id_b in tqdm(
        annotator_id_pairs,
        desc="Annotator pairs",
    ):
        spans_by_category_a = annotations[annotator_id_a]
        spans_by_category_b = annotations[annotator_id_b]
        for category_name in categories:
            span_texts_a = spans_by_category_a.get(category_name, []) or []
            span_texts_b = spans_by_category_b.get(category_name, []) or []
            _, _, f1 = _pairwise_prf1(
                span_texts_a,
                span_texts_b,
                distance_fn,
                min_similarity=min_sim,
            )
            pairwise_f1_values_by_category_name[category_name].append(f1)

    # Mean across annotator pairs (0.0 when there are no pairs).
    return {
        category_name: (float(np.mean(pairwise_f1_values)) if pairwise_f1_values else 0.0)
        for category_name, pairwise_f1_values in pairwise_f1_values_by_category_name.items()
    }

if __name__ == "__main__":
    import json
    import pathlib
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: python {pathlib.Path(__file__).name} annotations.json")
        sys.exit(1)
    annotations_by_annotator_id = json.loads(
        pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    )
    category_names = sorted(
        {category_name for ann in annotations_by_annotator_id.values() for category_name in ann}
    )
    print("Categories:", category_names)
    print(json.dumps(compute_iaa(annotations_by_annotator_id, category_names), indent=2))
