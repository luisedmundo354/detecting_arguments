import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

import span_iaa_ned as iaa_ned
import krippendorff


# ---------------------------------------------------------------------
# Existing loader (kept as-is) -> span TEXTS for NED/F1
# ---------------------------------------------------------------------
def load_annotations_from_file(path):
    """Load a single json (text spans)"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(path)

    annotator = data.get("completed_by", {}).get("email", '')
    ref_id = data.get("task", {}).get("data", {}).get("ref_id", {})

    try:
        ref_id = int(ref_id)
    except (TypeError, ValueError):
        pass

    ann = {}
    labels_set = set()
    for item in data.get("result", []):
        if item.get('type') != 'labels':
            continue
        cats = item.get('value', {}).get('labels', []) or []
        text = (item.get('value', {}).get('text', '')).strip()
        if not text:
            continue
        for cat in cats:
            labels_set.add(cat)
            ann.setdefault(cat, []).append(text)

    return annotator, ref_id, ann, labels_set


# ---------------------------------------------------------------------
# NEW: offset loader (minimal change) -> (start, end) for αᵤ
# Uses Label Studio's character offsets in result.value.start / result.value.end
# ---------------------------------------------------------------------
def load_offsets_from_file(path):
    """Load a single json (character offsets) for alpha_u"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotator = data.get("completed_by", {}).get("email", '')
    ref_id = data.get("task", {}).get("data", {}).get("ref_id", {})

    try:
        ref_id = int(ref_id)
    except (TypeError, ValueError):
        pass

    ann_offsets = {}
    labels_set = set()
    for item in data.get("result", []):
        if item.get('type') != 'labels':
            continue
        val = item.get('value', {}) or {}
        cats = val.get('labels', []) or []
        start = val.get('start') or val.get('startOffset')
        end   = val.get('end')   or val.get('endOffset')
        if start is None or end is None:
            continue
        s, e = int(start), int(end)
        if s >= e:
            continue
        for cat in cats:
            labels_set.add(cat)
            ann_offsets.setdefault(cat, []).append((s, e))

    return annotator, ref_id, ann_offsets, labels_set


# ---------------------------------------------------------------------
# Your existing nominal-alpha per doc (kept)
# ---------------------------------------------------------------------
def compute_alpha_per_doc(all_annotations_for_doc, categories):
    alpha_by_cat = {}
    annotators = sorted(all_annotations_for_doc.keys())

    for cat in categories:
        # units = all unique spans in this category
        units = sorted({u for ann in all_annotations_for_doc.values() for u in ann.get(cat, [])})
        if not units:
            alpha_by_cat[cat] = 1.0
            continue

        print("annotators:", annotators)
        print("categories:", categories)

        # Reliability matrix
        reliability = []
        for a in annotators:
            spans = set(all_annotations_for_doc.get(a, {}).get(cat, []))
            reliability.append([1.0 if u in spans else 0.0 for u in units])

        print("reliability", np.array(reliability, dtype=float))

        # Guard: domain-of-one value -> treat as perfect agreement
        M = np.array(reliability, dtype=float)
        vals = np.unique(M[~np.isnan(M)])
        if vals.size <= 1 or len(annotators) < 2:
            alpha_by_cat[cat] = 1.0 if vals.size == 1 else float('nan')
            continue

        alpha = krippendorff.alpha(
            reliability_data=M,
            level_of_measurement='nominal'
        )
        alpha_by_cat[cat] = float(alpha)

    return alpha_by_cat


# ---------------------------------------------------------------------
# Your existing nominal-alpha overall (kept)
# ---------------------------------------------------------------------
def compute_alpha_overall(grouped_by_doc, categories):
    alpha_by_cat = {}
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
            alpha_by_cat[cat] = float('nan')
            continue

        reliability = []
        for a in annotators:
            row = []
            for ref_id, span in units:
                # Missing if annotator did not label
                if a not in grouped_by_doc.get(ref_id, {}):
                    row.append(np.nan)
                else:
                    spans = set(grouped_by_doc[ref_id][a].get(cat, []))
                    row.append(1.0 if span in spans else 0.0)
            reliability.append(row)

        alpha = krippendorff.alpha(
            reliability_data=np.array(reliability, dtype=float),
            level_of_measurement='nominal'
        )
        alpha_by_cat[cat] = float(alpha)

    return alpha_by_cat


# =========================
# NEW: αᵤ (unitizing) code
# =========================

def _merge_intervals(iv):
    """Merge overlapping/adjacent intervals; αᵤ assumes no overlaps per coder."""
    if not iv:
        return []
    iv = sorted(iv)
    out = [iv[0]]
    for s, e in iv[1:]:
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out

def _boundaries(intervals_by_coder, L):
    """All change-points on the continuum (0, L, and all starts/ends)."""
    b = {0, L}
    for iv in intervals_by_coder.values():
        for s, e in iv:
            b.add(s); b.add(e)
    return sorted(b)

def _covers(iv, x):
    """Is position x inside any interval? 1/0."""
    # Linear scan (fine for modest sizes). Could be binary search with pointers if needed.
    for s, e in iv:
        if s <= x < e:
            return 1
        if x < s:
            return 0
    return 0

def _coincidence_lengths_binary(intervals_by_coder, L, include_background=False):
    """
    Observed coincidence matrix (2x2) from overlap *lengths* for a binary variable:
    value 1 = inside category; value 0 = outside.
    """
    merged = {c: _merge_intervals(iv) for c, iv in intervals_by_coder.items()}
    coders = sorted(merged)
    B = _boundaries(merged, L)

    o = np.zeros((2, 2), dtype=float)
    # sum over coder pairs and atomic segments
    for i in range(len(coders)):
        for j in range(i + 1, len(coders)):
            iv_i, iv_j = merged[coders[i]], merged[coders[j]]
            for a, b in zip(B[:-1], B[1:]):
                if a >= b:
                    continue
                li = _covers(iv_i, a)
                lj = _covers(iv_j, a)
                if include_background or (li == 1 or lj == 1):
                    o[li, lj] += (b - a)
    return o

def _alpha_from_o_nominal(o):
    """
    Compute α from a coincidence matrix o (length-based).
    """
    n = float(o.sum())
    if n == 0:
        return 1.0
    n_v = o.sum(axis=1)
    # expected coincidences (length-based analogue of standard α expectation)
    e = np.empty_like(o, dtype=float)
    denom = max(n - 1.0, 1.0)
    for v in range(o.shape[0]):
        for vp in range(o.shape[1]):
            if v == vp:
                e[v, vp] = (n_v[v] * max(n_v[v] - 1.0, 0.0)) / denom
            else:
                e[v, vp] = (n_v[v] * n_v[vp]) / denom

    Do = float(o[np.eye(o.shape[0]) == 0].sum())
    De = float(e[np.eye(e.shape[0]) == 0].sum())
    if De == 0:
        return 1.0  # unanimous domain
    return 1.0 - (Do / De)

def compute_alpha_u_per_doc(all_offsets_for_doc, categories, continuum_len=None, include_background=False):
    """
    αᵤ per category for ONE document.
    all_offsets_for_doc: {annotator: {category: [(start,end), ...]}}
    """
    if len(all_offsets_for_doc) < 2:
        return {cat: float('nan') for cat in categories}

    # pick continuum length: max end if not given
    if continuum_len is None:
        L = 0
        for per_ann in all_offsets_for_doc.values():
            for spans in per_ann.values():
                for s, e in spans:
                    L = max(L, e)
    else:
        L = int(continuum_len)

    out = {}
    for cat in categories:
        iv_by_coder = {coder: all_offsets_for_doc.get(coder, {}).get(cat, [])
                       for coder in all_offsets_for_doc}
        o = _coincidence_lengths_binary(iv_by_coder, L, include_background=include_background)
        out[cat] = _alpha_from_o_nominal(o)
    return out

def compute_alpha_u_overall(grouped_offsets_by_doc, categories, include_background=False):
    """
    αᵤ overall across documents: sum observed coincidence matrices then compute α.
    """
    overall = {}
    for cat in categories:
        O_sum = np.zeros((2, 2), dtype=float)
        for ref_id, ann_by_annot in grouped_offsets_by_doc.items():
            # continuum length per doc: max end
            L = 0
            for per_ann in ann_by_annot.values():
                for s, e in per_ann.get(cat, []):
                    L = max(L, e)
            iv_by_coder = {coder: ann_by_annot.get(coder, {}).get(cat, [])
                           for coder in ann_by_annot}
            O_sum += _coincidence_lengths_binary(iv_by_coder, L, include_background=include_background)
        overall[cat] = _alpha_from_o_nominal(O_sum)
    return overall


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Compute IAA F1 and Krippendorff alpha'
    )
    parser.add_argument(
        '--input-dir', required=True,
        help='Directory with annotations'
    )
    parser.add_argument(
        '--min-annotators', type=int, default=2,
        help='Minimum number of annotators'
    )
    # NEW: toggle αᵤ (unitizing) and whether to include background (0/0) regions
    parser.add_argument(
        '--alpha-u', action='store_true',
        help="Compute Krippendorff's alpha for unitizing (over character offsets)"
    )
    parser.add_argument(
        '--include-background', action='store_true',
        help="For alpha-u: include 0/0 background regions (defaults to False)"
    )

    args = parser.parse_args()
    root = Path(args.input_dir)
    if not root.exists() or not root.is_dir():
        parser.error(f'--input-dir {root} is not a directory')

    files = sorted(root.glob('*.json'))
    if not files:
        parser.error(f'--input-dir {root} is empty')

    grouped = {}
    grouped_offsets = {}   # NEW
    categories_set = set()

    for p in files:
        # load text spans (existing)
        annotator, ref_id, ann, labels = load_annotations_from_file(p)
        if ref_id is None:
            continue
        grouped.setdefault(ref_id, {})[annotator] = ann
        categories_set.update(labels)

        # load offsets (NEW, minimal change)
        a2, ref2, ann_offs, labels2 = load_offsets_from_file(p)
        if ref2 != ref_id or a2 != annotator:
            # not fatal; keep consistent by keying on ref_id/annotator we already used
            pass
        grouped_offsets.setdefault(ref_id, {})[annotator] = ann_offs
        categories_set.update(labels2)

    if not grouped:
        parser.error(f'No annotations found in {root}')

    # filter documents by annotator count
    grouped = {rid: ann_by_annot for rid, ann_by_annot in grouped.items()
               if len(ann_by_annot) >= args.min_annotators}
    grouped_offsets = {rid: grouped_offsets.get(rid, {})
                       for rid in grouped.keys()}

    if not grouped:
        parser.error('After filtering, no documents have enough annotators to compute IAA.')

    categories = sorted(categories_set)

    per_doc_f1 = {}
    per_doc_alpha = {}
    pair_counts = {}

    for ref_id, ann_by_annotator in tqdm(sorted(grouped.items())):
        # F1 (unchanged)
        f1_by_cat = iaa_ned.compute_iaa(ann_by_annotator, categories, metric="yujianbo", min_sim=0.1)
        per_doc_f1[ref_id] = f1_by_cat

        # Alpha: nominal (existing) OR αᵤ (NEW)
        if args.alpha_u:
            ann_by_annotator_offsets = grouped_offsets.get(ref_id, {})
            alpha_by_cat = compute_alpha_u_per_doc(
                ann_by_annotator_offsets, categories,
                continuum_len=None,
                include_background=args.include_background
            )
        else:
            alpha_by_cat = compute_alpha_per_doc(ann_by_annotator, categories)

        per_doc_alpha[ref_id] = alpha_by_cat

        n_annotators = len(ann_by_annotator)
        pair_counts[ref_id] = (n_annotators * (n_annotators - 1)) // 2

    # overall F1: micro-average over annotator pairs (unchanged)
    overall_f1 = {}
    for cat in categories:
        num = 0.0
        den = 0
        for ref_id, scores in per_doc_f1.items():
            pairs = pair_counts[ref_id]
            if pairs <= 0:
                continue
            s = scores.get(cat)
            if s is None:
                continue
            num += s * pairs
            den += pairs
        overall_f1[cat] = (num / den) if den else float('nan')

    # overall alpha: nominal (existing) OR αᵤ (NEW)
    if args.alpha_u:
        overall_alpha = compute_alpha_u_overall(grouped_offsets, categories,
                                                include_background=args.include_background)
    else:
        overall_alpha = compute_alpha_overall(grouped, categories)

    print(f'Found {len(files)} files; included {len(grouped)} documents')
    print('Categories:', categories)

    print('\nPer-document results:')
    for ref_id in sorted(grouped.keys()):
        print(f'-- ref_id: {ref_id} (annotators: {len(grouped[ref_id])}) --')
        print('F1 by category:')
        for cat in categories:
            print(f'\t{cat}: {per_doc_f1[ref_id][cat]}')
        print("Krippendorff alpha{} by category:".format(" (unitizing)" if args.alpha_u else ""))
        for cat in categories:
            val = per_doc_alpha[ref_id][cat]
            print(f'\t{cat}: {val}')

    print('\nOverall across documents:')
    print('F1 by category (micro average over annotator pairs):')
    for cat in categories:
        val = overall_f1.get(cat, float('nan'))
        print(f'\t{cat}: {val}')
    print("Krippendorff alpha{} by category:".format(" (unitizing)" if args.alpha_u else ""))
    for cat in categories:
        val = overall_alpha.get(cat, float('nan'))
        print(f'\t{cat}: {val}')


if __name__ == '__main__':
    main()
