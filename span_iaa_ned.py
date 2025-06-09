"""
span_iaa.py – Fast inter-annotator agreement for span tasks
==========================================================

Usage
-----
import span_iaa as iaa

f1_by_cat = iaa.compute_iaa(
    annotations = {
        "ann_1": {"weakness": [...], "strength": [...], ...},
        "ann_2": {...},
        ...
    },
    categories = ["weakness", "strength", "neutral"]
)

print(f1_by_cat)        # {'weakness': 0.52, 'strength': 0.74, 'neutral': 0.69}
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import itertools
import numpy as np
from jedi.inference.gradual.typing import Callable
from scipy.optimize import linear_sum_assignment
import Levenshtein
from abydos.distance import YujianBo, HigueraMico
from functools import lru_cache

# ---------------------------------------------------------------------------
# 1.  Three calculations: Levenshtein distance, Yujian-Bo and Higuera-Mico
# ---------------------------------------------------------------------------

def _marzal_ned(s: str, t: str) -> float:
    """
    Todo: This implementation is not ready. It doesn't follow min(cost(p))
    Parameters
    ----------
    s, t : str
        Two spans to compare.

    Returns
    -------
    float
        Normalised edit distance in [0, 2]  (0 = identical, 2 = worst-case)
    """
    if s == t:
        return 0.0

    ops = Levenshtein.editops(s, t)      # optimal edit script
    if not ops:
        return 0.0                       # equal strings caught above anyway

    ins  = sum(op[0] == 'insert'  for op in ops)
    dele = sum(op[0] == 'delete'  for op in ops)
    subs = sum(op[0] == 'replace' for op in ops)

    weight = ins + dele + subs
    length = len(ops)                  #  path length |p|

    return weight / length

_YB  = YujianBo()
_HM  = HigueraMico()

def _yujianbo_nld(s: str, t: str) -> float:
    return round(_YB.dist(s, t), 6)

def _higuera_mico_ned(s: str, t: str) -> float:
    return round(_HM.dist(s, t), 6)

@lru_cache(maxsize=100_000)
def _ned_cached(a: str, b: str, metric: Callable[[str, str], float]) -> float:
    return metric(a, b)

# ---------------------------------------------------------------------------
# 2.  Maximum-weight bipartite matching for one annotator pair, one category
# ---------------------------------------------------------------------------

def _similarity_matrix(spans_a: List[str], spans_b: List[str], metric: Callable[[str, str], float]) -> np.ndarray:
    """Return S[i,j] =1 - ned"""
    n, m = len(spans_a), len(spans_b)
    s_matrix = np.empty((n, m), dtype=float)
    for i, sa in enumerate(spans_a):
        for j, sb in enumerate(spans_b):
            s_matrix[i, j] = 1.0 - _ned_cached(sa, sb, metric)
    return s_matrix

def _match_and_f1(spans_a: List[str],
                  spans_b: List[str]) -> Dict[str, Tuple[float, float, float]]:
    """
    Align two span lists, then compute precision, recall, F1 (micro).

    Returns
    -------
    prec, rec, f1 : float
    """

    metrics: Dict[str, Callable[[List[str], List[str]], float]] = {
        "marzal": _marzal_ned,
        "yujianbo": _yujianbo_nld,
       # "higuera_mico": _higuera_mico_ned,
    }

    if not spans_a and not spans_b:
        return {name: (1.0, 1.0, 1.0) for name in metrics}           # trivially perfect agreement
    if not spans_a or not spans_b:
        return {name: (0.0, 0.0, 0.0) for name in metrics}

    n, m = len(spans_a), len(spans_b)
    scores: Dict[str, Tuple[float, float, float]] = {}

    for name, metric in metrics.items():
        s_matrix = _similarity_matrix(spans_a, spans_b, metric)

        # Maximum-weight matching  (Hungarian wants *cost*, so negate)
        row_idx, col_idx = linear_sum_assignment(-s_matrix)
        tp_soft = s_matrix[row_idx, col_idx].sum()     # soft TP

        precision = tp_soft / n
        rec  = tp_soft / m
        f1   = 0.0 if tp_soft == 0 else (2 * precision * rec) / (precision + rec)

        scores[name] = (precision, rec, f1)

    return scores

# ---------------------------------------------------------------------------
# 3.  Aggregate over annotators and categories
# ---------------------------------------------------------------------------

def compute_iaa(annotations: Dict[str, Dict[str, List[str]]],
                categories: List[str]
               ) -> dict[str, dict[str, float]]:
    """
    Compute micro-average pairwise F1 for each category
    annotations
        {annotator_id: {category: [span_str, ...], ...}, ...}
    categories: List[str]
    """
    # All unordered annotator pairs
    annotators = list(annotations)
    pairs = list(itertools.combinations(annotators, 2))

    metric_names = ('marzal', 'yujianbo', 'higuera_mico')
    f1_by_cat: Dict[str, Dict[str, List[float]]] = {c: {m: [] for m in metric_names} for c in categories}

    for a1, a2 in pairs:
        ann1, ann2 = annotations[a1], annotations[a2]
        for cat in categories:
            scores = _match_and_f1(ann1.get(cat, []), ann2.get(cat, []))
            for metric, (_p, _r, f1) in scores.items():
                f1_by_cat[cat][metric].append(f1)

     # Mean over annotator pairs
    mean_by_cat: Dict[str, Dict[str, float]] = {}
    for cat, metric_dict in f1_by_cat.items():
        mean_by_cat[cat] = {
            metric: (sum(vals) / len(vals) if vals else 0.0)
            for metric, vals in metric_dict.items()
        }

    return mean_by_cat


# ---------------------------------------------------------------------------
# 4.  Convenience CLI -- `python span_iaa.py example.json`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json, sys, pathlib, textwrap
    if len(sys.argv) != 2:
        print(textwrap.dedent(f"""
            Usage:
                python {pathlib.Path(__file__).name} annotations.json
            The JSON file must have the structure described in compute_iaa().
        """).strip())
        sys.exit(1)

    data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
    cats = sorted(next(iter(data.values())).keys())
    print("Categories:", cats)
    print(json.dumps(compute_iaa(data, cats), indent=2))