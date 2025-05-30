from __future__ import annotations

import argparse
import random                  # shuffle in-place ➜ O(n) time, O(1) space :contentReference[oaicite:2]{index=2}
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple  # Dict fixes the earlier NameError  :contentReference[oaicite:3]{index=3}

from label_studio_format import text_to_ls_format  # your helper

WORD_GAP = 2_000  # target max difference in total words per annotator

# ───────────────────────────────────────────────────────────────────────────
# I/O helpers
# ───────────────────────────────────────────────────────────────────────────
def _collect_txt_files(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        sys.exit(f"[error] folder not found: {folder}")
    yield from folder.glob("*.txt")               # non-recursive

def _word_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").split())  # simple tokeniser

# ───────────────────────────────────────────────────────────────────────────
# Core balancer
# ───────────────────────────────────────────────────────────────────────────
def balanced_assignment(
        files: List[Tuple[Path, int]],
        annotators: List[str],
        overlap_ratio: float,
) -> Tuple[Dict[str, List[Tuple[Path, int]]], List[Tuple[Path, int]]]:
    """
    Distribute *files* across *annotators* while duplicating an
    ``overlap_ratio`` share to everyone and keeping workloads balanced.
    """
    random.shuffle(files)                                         # :contentReference[oaicite:4]{index=4}

    n_total   = len(files)
    n_overlap = round(n_total * overlap_ratio)
    overlap_docs   = files[:n_overlap]
    remaining_docs = files[n_overlap:]

    buckets      = {a: overlap_docs.copy() for a in annotators}
    bucket_loads = {a: sum(w for _, w in overlap_docs) for a in annotators}

    # first-fit decreasing heuristic
    remaining_docs.sort(key=lambda t: t[1], reverse=True)
    for doc, words in remaining_docs:
        tgt = min(bucket_loads, key=bucket_loads.get)
        buckets[tgt].append((doc, words))
        bucket_loads[tgt] += words

    # hill-climb swaps until within WORD_GAP
    for _ in range(1_000):
        heavy = max(bucket_loads, key=bucket_loads.get)
        light = min(bucket_loads, key=bucket_loads.get)
        if bucket_loads[heavy] - bucket_loads[light] <= WORD_GAP:
            break
        h_opts = [d for d in buckets[heavy] if d not in overlap_docs]
        l_opts = [d for d in buckets[light] if d not in overlap_docs]
        if not h_opts or not l_opts:
            break
        dh, dl = random.choice(h_opts), random.choice(l_opts)
        buckets[heavy].remove(dh); buckets[heavy].append(dl)
        buckets[light].remove(dl);  buckets[light].append(dh)
        bucket_loads[heavy] += dl[1] - dh[1]
        bucket_loads[light] += dh[1] - dl[1]

    return buckets, overlap_docs

# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:                                # :contentReference[oaicite:5]{index=5}
    p = argparse.ArgumentParser(
        description="Assign .txt cases to annotators (with overlap) and export LS-JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("cases_dir",  type=Path, help="Folder containing .txt files")
    p.add_argument("output_dir", type=Path, help="Destination root for JSONs")
    p.add_argument("--annotators", nargs="+", required=True, metavar="EMAIL",
                   help="Annotator e-mail addresses")                 # :contentReference[oaicite:6]{index=6}
    p.add_argument("--overlap",  type=float, default=0.10,
                   help="Fraction of docs everyone annotates")
    p.add_argument("--start-ref", type=int, default=1,
                   help="Initial reference ID")
    return p.parse_args()

# ───────────────────────────────────────────────────────────────────────────
# Entrypoint
# ───────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # 0️⃣  discover files
    txt_files = list(_collect_txt_files(args.cases_dir))
    if not txt_files:
        sys.exit("[error] No .txt files found")
    random.shuffle(txt_files)

    # 1️⃣  split overlap / remaining (mutually exclusive)
    n_overlap = round(len(txt_files) * args.overlap)
    overlap_docs   = [(p, _word_count(p)) for p in txt_files[:n_overlap]]
    remaining_docs = [(p, _word_count(p)) for p in txt_files[n_overlap:]]

    # 2️⃣  write overlap docs once
    ref = args.start_ref
    overlap_out = args.output_dir / "overlapping"
    for path, _ in overlap_docs:
        text_to_ls_format(path, overlap_out, ref_id=ref,
                          assigned_to=args.annotators)          # list → JSON array
        ref += 1

    # 3️⃣  balance the rest and write per-annotator
    assignment, _ = balanced_assignment(
        remaining_docs,
        annotators=args.annotators,
        overlap_ratio=0.0,     # no extra overlap inside
    )

    for annotator, docs in assignment.items():
        out_dir = args.output_dir / annotator
        for path, _ in docs:
            text_to_ls_format(path, out_dir, ref_id=ref, assigned_to=annotator)
            ref += 1

if __name__ == "__main__":
    main()


