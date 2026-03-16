"""Span-level IAA using pairwise soft-F1 with normalized edit distance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import itertools
import numpy as np
from tqdm import tqdm

# Distance functions that return values in [0, 1].
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - dependency handled at call time
    linear_sum_assignment = None

try:
    from abydos.distance import YujianBo, HigueraMico
except ImportError:  # pragma: no cover - dependency handled at call time
    YujianBo = None
    HigueraMico = None


_YUJIANBO_DISTANCE = YujianBo() if YujianBo is not None else None
_HIGUERAMICO_DISTANCE = HigueraMico() if HigueraMico is not None else None

def _yujianbo_distance(span_text_a: str, span_text_b: str) -> float:
    """Distance in [0, 1] between two span strings."""

    if _YUJIANBO_DISTANCE is None:
        raise ImportError("abydos is required for the 'yujianbo' span distance metric.")
    return _YUJIANBO_DISTANCE.dist(span_text_a, span_text_b)


def _higueramico_distance(span_text_a: str, span_text_b: str) -> float:
    """Distance in [0, 1] between two span strings."""

    if _HIGUERAMICO_DISTANCE is None:
        raise ImportError("abydos is required for the 'higueramico' span distance metric.")
    return _HIGUERAMICO_DISTANCE.dist(span_text_a, span_text_b)

# Supported distance function lookup.
_DISTANCE_FN_BY_NAME: Dict[str, Callable[[str, str], float]] = {
    "yujianbo": _yujianbo_distance,
    "higueramico": _higueramico_distance,
}


@dataclass(frozen=True)
class MatchingResult:
    """Best 1:1 matching between two text sequences."""

    matches: List[Tuple[int, int, float]]
    unmatched_indices_a: List[int]
    unmatched_indices_b: List[int]


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

    matching = compute_optimal_matching_from_similarity_matrix(similarity_matrix)
    return float(sum(similarity for _, _, similarity in matching.matches))


def resolve_distance_function(metric: str) -> Callable[[str, str], float]:
    """Return the requested distance function or fail loudly."""

    if metric not in _DISTANCE_FN_BY_NAME:
        raise ValueError(
            f"Unsupported metric '{metric}'. Choose from {sorted(_DISTANCE_FN_BY_NAME.keys())}"
        )
    return _DISTANCE_FN_BY_NAME[metric]


def compute_optimal_matching(
    span_texts_a: List[str],
    span_texts_b: List[str],
    *,
    metric: str = "yujianbo",
    min_similarity: float = 0.0,
) -> MatchingResult:
    """Return the best 1:1 matching used by the soft-F1 scorer."""

    if linear_sum_assignment is None:
        raise ImportError("scipy is required for Hungarian matching in span IAA.")
    distance_fn = resolve_distance_function(metric)
    similarity_matrix = _similarity_matrix(
        span_texts_a,
        span_texts_b,
        distance_fn,
        min_similarity=min_similarity,
    )
    return compute_optimal_matching_from_similarity_matrix(similarity_matrix)


def compute_optimal_matching_from_similarity_matrix(
    similarity_matrix: np.ndarray,
) -> MatchingResult:
    """Return the best 1:1 matching for a precomputed similarity matrix."""

    if linear_sum_assignment is None:
        raise ImportError("scipy is required for Hungarian matching in span IAA.")
    span_text_count_a, span_text_count_b = similarity_matrix.shape
    if span_text_count_a == 0 and span_text_count_b == 0:
        return MatchingResult(matches=[], unmatched_indices_a=[], unmatched_indices_b=[])

    padded_size = max(span_text_count_a, span_text_count_b)
    padded_similarity_matrix = np.zeros((padded_size, padded_size), dtype=float)
    padded_similarity_matrix[:span_text_count_a, :span_text_count_b] = similarity_matrix

    matched_row_indices, matched_col_indices = linear_sum_assignment(-padded_similarity_matrix)

    matches: List[Tuple[int, int, float]] = []
    matched_a = set()
    matched_b = set()
    for row_index, col_index in zip(matched_row_indices.tolist(), matched_col_indices.tolist()):
        if row_index >= span_text_count_a or col_index >= span_text_count_b:
            continue
        similarity = float(similarity_matrix[row_index, col_index])
        if similarity <= 0.0:
            continue
        matches.append((row_index, col_index, similarity))
        matched_a.add(row_index)
        matched_b.add(col_index)

    unmatched_indices_a = [
        index for index in range(span_text_count_a) if index not in matched_a
    ]
    unmatched_indices_b = [
        index for index in range(span_text_count_b) if index not in matched_b
    ]
    return MatchingResult(
        matches=matches,
        unmatched_indices_a=unmatched_indices_a,
        unmatched_indices_b=unmatched_indices_b,
    )


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

    distance_fn = resolve_distance_function(metric)

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
