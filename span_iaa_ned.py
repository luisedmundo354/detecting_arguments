# span_iaa_ned.py
"""
Fast inter-annotator agreement (IAA) for span tasks using [0,1]-normalized
edit distances (NED). Designed to plug into the folder-based script:

    f1_by_cat = iaa_ned.compute_iaa(all_annotations, categories)

Inputs:
    annotations: {annotator_id: {category: [span_text, ...]}, ...}
    categories : list[str]

Returns:
    {category: F1}   # mean of pairwise F1s across annotator pairs

Only normalized-in-[0,1] distances are used, so precision/recall/F1 remain valid.

Requires:
    pip install numpy scipy abydos
"""
from __future__ import annotations

from typing import Dict, List, Callable, Tuple
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# Normalized edit distances from Abydos
from abydos.distance import YujianBo, HigueraMico



_YB = YujianBo()
_HM = HigueraMico()

def _yujianbo_nld(a: str, b: str) -> float:
    # Metric in [0,1] as per Yujian & Bo
    return _YB.dist(a, b)

def _higueramico_nld(a: str, b: str) -> float:
    # Contextual normalized edit distance, also normalized in [0,1] in Abydos
    return _HM.dist(a, b)

# Registry of supported metrics (all normalized)
_METRICS: Dict[str, Callable[[str, str], float]] = {
    "yujianbo": _yujianbo_nld,
    "higueramico": _higueramico_nld,
}



def _similarity_matrix(
        spans_a: List[str],
        spans_b: List[str],
        metric: Callable[[str, str], float],
        min_sim: float = 0.0,
) -> np.ndarray:
    """
    Build similarity matrix S where S[i, j] = 1 - NED(a_i, b_j).
    Entries below min_sim are clipped to 0 (acts as 'forbidden' when padded).
    """
    n, m = len(spans_a), len(spans_b)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=float)

    S = np.empty((n, m), dtype=float)
    for i, sa in enumerate(spans_a):
        for j, sb in enumerate(spans_b):
            s = 1.0 - float(metric(sa, sb))
            S[i, j] = s if s >= min_sim else 0.0
    return S


def _soft_tp_from_matching(S: np.ndarray) -> float:
    """
    Compute soft true positives as the sum of matched similarities.

    We pad to a square matrix with zeros so the Hungarian algorithm
    can 'leave spans unmatched' by pairing them with a zero column/row.
    """
    n, m = S.shape
    if n == 0 and m == 0:
        return 0.0
    N = max(n, m)
    Ssq = np.zeros((N, N), dtype=float)
    Ssq[:n, :m] = S
    # Hungarian expects a cost matrix; maximize S by minimizing -S
    row_idx, col_idx = linear_sum_assignment(-Ssq)
    tp_soft = float(Ssq[row_idx, col_idx].sum())
    return tp_soft


def _pairwise_prf1(
        spans_a: List[str],
        spans_b: List[str],
        metric: Callable[[str, str], float],
        min_sim: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Precision, recall, F1 for a single annotator pair within one category,
    using 'soft' true positives (sum of matched similarities).
    """
    n, m = len(spans_a), len(spans_b)

    # Trivial perfect agreement if both empty
    if n == 0 and m == 0:
        return 1.0, 1.0, 1.0
    # If exactly one is empty, there can be no matches
    if n == 0 or m == 0:
        return 0.0, 0.0, 0.0

    S = _similarity_matrix(spans_a, spans_b, metric, min_sim=min_sim)
    tp_soft = _soft_tp_from_matching(S)

    precision = tp_soft / n
    recall = tp_soft / m
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
    """
    Compute mean pairwise F1 for each category across all unordered
    annotator pairs. Returns {category: F1}.

    Args:
        annotations: {annotator: {category: [span_text, ...]}}
        categories : list of categories to score
        metric     : one of {"yujianbo", "higueramico"} (all in [0,1])
        min_sim    : optional similarity threshold in [0,1].
                     Pairs below this are treated like 'no match'.

    Notes:
        - This is a *mean over annotator pairs* (macro over pairs).
        - Soft TP = sum of matched similarities from Hungarian alignment.
    """
    if metric not in _METRICS:
        raise ValueError(f"Unsupported metric '{metric}'. "
                         f"Choose from {sorted(_METRICS.keys())}")
    dist = _METRICS[metric]

    ann_ids = list(annotations)
    pairs = list(itertools.combinations(ann_ids, 2))

    f1_by_cat: Dict[str, List[float]] = {c: [] for c in categories}

    for a1, a2 in tqdm(pairs, desc="Pairs processing progress"):
        ann1, ann2 = annotations[a1], annotations[a2]
        for cat in categories:
            spans_a = ann1.get(cat, []) or []
            spans_b = ann2.get(cat, []) or []
            _, _, f1 = _pairwise_prf1(spans_a, spans_b, dist, min_sim=min_sim)
            f1_by_cat[cat].append(f1)

    # Mean over annotator pairs; if no pairs, return 0.0
    return {
        cat: (float(np.mean(vals)) if vals else 0.0)
        for cat, vals in f1_by_cat.items()
    }

if __name__ == "__main__":
    import json, sys, pathlib
    if len(sys.argv) != 2:
        print(f"Usage: python {pathlib.Path(__file__).name} annotations.json")
        sys.exit(1)
    data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
    cats = sorted({c for ann in data.values() for c in ann})
    print("Categories:", cats)
    print(json.dumps(compute_iaa(data, cats), indent=2))
