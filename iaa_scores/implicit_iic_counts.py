"""Count implicit intermediate conclusions per document and annotator."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

from . import DEFAULT_ANNOTATION_DIR
from .annotation_graphs import load_annotation_graphs


def collect_implicit_iic_counts(root: Path) -> tuple[list[dict[str, object]], list[str]]:
    """Return one count row per ref_id and annotator."""

    graphs_by_doc = load_annotation_graphs(
        root,
        min_annotators=1,
        require_repaired_implicit_offsets=True,
    )

    label_set = set()
    rows: list[dict[str, object]] = []
    for ref_id, ann_by_annotator in sorted(graphs_by_doc.items()):
        for annotator, record in sorted(ann_by_annotator.items()):
            counts = Counter()
            for span in record.annotation.spans:
                if not span.is_normalized_implicit_intermediate:
                    continue
                counts[span.label] += 1
                counts["TOTAL"] += 1
                label_set.add(span.label)

            row = {
                "ref_id": ref_id,
                "source_file": record.path.name,
                "annotator": annotator,
            }
            for label in sorted(label_set):
                row[label] = int(counts.get(label, 0))
            row["TOTAL"] = int(counts.get("TOTAL", 0))
            rows.append(row)

    labels = sorted(label_set)
    for row in rows:
        for label in labels:
            row.setdefault(label, 0)
    return rows, labels


def summarize_implicit_iic_counts(
    rows: Sequence[dict[str, object]],
    labels: Sequence[str],
) -> list[dict[str, object]]:
    """Return one summary row per annotator with totals and per-document averages."""

    totals_by_annotator: dict[str, Counter[str]] = {}
    doc_counts_by_annotator: Counter[str] = Counter()

    for row in rows:
        annotator = str(row["annotator"])
        totals = totals_by_annotator.setdefault(annotator, Counter())
        for label in labels:
            totals[label] += int(row.get(label, 0))
        totals["TOTAL"] += int(row.get("TOTAL", 0))
        doc_counts_by_annotator[annotator] += 1

    summary_rows: list[dict[str, object]] = []
    for annotator in sorted(totals_by_annotator):
        doc_count = int(doc_counts_by_annotator[annotator])
        if doc_count <= 0:
            raise ValueError(f"Annotator {annotator!r} has no document rows to summarize.")
        totals = totals_by_annotator[annotator]
        row: dict[str, object] = {
            "ref_id": "SUMMARY",
            "source_file": f"documents={doc_count}",
            "annotator": annotator,
            "TOTAL": int(totals["TOTAL"]),
        }
        for label in labels:
            row[label] = int(totals[label])
            row[f"AVG_{label}"] = totals[label] / doc_count
        row["AVG_TOTAL"] = totals["TOTAL"] / doc_count
        summary_rows.append(row)
    return summary_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count implicit intermediate conclusions per document and annotator.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR,
        help="Directory with repaired IAA Label Studio exports.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.input_dir.expanduser()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"--input-dir {root} is not a directory")

    rows, labels = collect_implicit_iic_counts(root)
    summary_rows = summarize_implicit_iic_counts(rows, labels)
    avg_columns = [f"AVG_{label}" for label in labels] + ["AVG_TOTAL"]
    writer = csv.writer(sys.stdout, delimiter="\t")
    writer.writerow(["ref_id", "source_file", "annotator", *labels, "TOTAL", *avg_columns])
    for row in rows:
        writer.writerow(
            [
                row["ref_id"],
                row["source_file"],
                row["annotator"],
                *[row[label] for label in labels],
                row["TOTAL"],
                *["" for _ in avg_columns],
            ]
        )
    for row in summary_rows:
        writer.writerow(
            [
                row["ref_id"],
                row["source_file"],
                row["annotator"],
                *[row[label] for label in labels],
                row["TOTAL"],
                *[row[column] for column in avg_columns],
            ]
        )


if __name__ == "__main__":  # pragma: no cover
    main()
