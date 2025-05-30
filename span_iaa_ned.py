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
from typing import Dict, List, Tuple
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
import Levenshtein

# ---------------------------------------------------------------------------
# 1.  Normalised Edit Distance (Marzal & Vidal)
# ---------------------------------------------------------------------------

def _ned(s: str, t: str) -> float:
    """
    Marzal & Vidal NED with costs 1/1/2 (ins/del/sub); O(|edit path|).

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

    weight = ins + dele + 2 * subs       # Marzal & Vidal weighted cost
    length = len(ops)                    # path length |p|

    return weight / length


# Cache the NED results because many spans repeat across pairwise comparisons
from functools import lru_cache
@lru_cache(maxsize=100_000)
def _ned_cached(a: str, b: str) -> float:
    return _ned(a, b)


# ---------------------------------------------------------------------------
# 2.  Maximum-weight bipartite matching for one annotator pair, one category
# ---------------------------------------------------------------------------

def _match_and_f1(spans_a: List[str],
                  spans_b: List[str]) -> Tuple[float, float, float]:
    """
    Align two span lists, then compute precision, recall, F1 (micro).

    Returns
    -------
    prec, rec, f1 : float
    """
    if not spans_a and not spans_b:
        return 1.0, 1.0, 1.0             # trivially perfect agreement
    if not spans_a or not spans_b:
        return 0.0, 0.0, 0.0

    # Build similarity matrix S[i,j] = 1 − NED
    n, m = len(spans_a), len(spans_b)
    S = np.empty((n, m), dtype=float)
    for i, sa in enumerate(spans_a):
        for j, sb in enumerate(spans_b):
            S[i, j] = 1.0 - _ned_cached(sa, sb)

    # Maximum-weight matching  (Hungarian wants *cost*, so negate)
    row_idx, col_idx = linear_sum_assignment(-S)        #  :contentReference[oaicite:5]{index=5}
    tp_soft = S[row_idx, col_idx].sum()                 # soft TP

    prec = tp_soft / n
    rec  = tp_soft / m
    f1   = 0.0 if tp_soft == 0 else (2 * prec * rec) / (prec + rec)

    return prec, rec, f1


# ---------------------------------------------------------------------------
# 3.  Aggregate over annotators and categories
# ---------------------------------------------------------------------------

def compute_iaa(annotations: Dict[str, Dict[str, List[str]]],
                categories: List[str]
               ) -> Dict[str, float]:
    """
    Compute micro-average pairwise F1 for each category.

    Parameters
    ----------
    annotations
        {annotator_id: {category: [span_str, ...], ...}, ...}
    categories
        List of categories to evaluate (must be keys in each annotator dict).

    Returns
    -------
    {category: mean_pairwise_F1}
    """
    # All unordered annotator pairs
    annotators = list(annotations)
    pairs = list(itertools.combinations(annotators, 2))

    f1_by_cat: Dict[str, List[float]] = {c: [] for c in categories}

    for a1, a2 in pairs:
        ann1, ann2 = annotations[a1], annotations[a2]
        for cat in categories:
            _, _, f1 = _match_and_f1(ann1.get(cat, []),
                                     ann2.get(cat, []))
            f1_by_cat[cat].append(f1)

    # Mean over pairs  (micro-average across annotator pairs)
    return {cat: (sum(vals) / len(vals) if vals else 0.0)
            for cat, vals in f1_by_cat.items()}


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