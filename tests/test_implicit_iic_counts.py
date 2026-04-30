from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from iaa_scores.implicit_iic_counts import (
    collect_implicit_iic_counts,
    summarize_implicit_iic_counts,
)


def _write_export(
    root: Path,
    filename: str,
    *,
    export_id: int,
    ref_id: int,
    annotator: str,
    case_text: str,
    results: list[dict],
) -> None:
    payload = {
        "id": export_id,
        "completed_by": {"email": annotator},
        "task": {
            "id": export_id,
            "data": {
                "assigned_to": annotator,
                "case_content": case_text,
                "ref_id": ref_id,
            },
        },
        "result": results,
    }
    (root / filename).write_text(json.dumps(payload), encoding="utf-8")


def _span(node_id: str, label: str, text: str, *, start: int | None, end: int | None) -> dict:
    return {
        "id": node_id,
        "type": "labels",
        "from_name": "label",
        "to_name": "text",
        "origin": "manual",
        "value": {
            "start": start,
            "end": end,
            "text": text,
            "labels": [label],
        },
    }


class ImplicitIICCountTests(unittest.TestCase):
    def test_collect_implicit_iic_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_text = "A. B."
            _write_export(
                root,
                "101_a.json",
                export_id=101,
                ref_id=1,
                annotator="a@example.com",
                case_text=case_text,
                results=[
                    _span("i1", "Analysis", "Implicit Intermediate Conclusion [x]", start=None, end=None),
                    _span("i2", "Rule", "Implicit Intermediate Conclusion [y]", start=None, end=None),
                ],
            )
            _write_export(
                root,
                "102_b.json",
                export_id=102,
                ref_id=1,
                annotator="b@example.com",
                case_text=case_text,
                results=[
                    _span("i3", "Analysis", "Implicit Intermediate Conclusion [z]", start=None, end=None),
                ],
            )

            rows, labels = collect_implicit_iic_counts(root)
            summary_rows = summarize_implicit_iic_counts(rows, labels)

            self.assertEqual(labels, ["Analysis", "Rule"])
            self.assertEqual(len(rows), 2)
            row_a = next(row for row in rows if row["annotator"] == "a@example.com")
            row_b = next(row for row in rows if row["annotator"] == "b@example.com")
            self.assertEqual(row_a["Analysis"], 1)
            self.assertEqual(row_a["Rule"], 1)
            self.assertEqual(row_a["TOTAL"], 2)
            self.assertEqual(row_b["Analysis"], 1)
            self.assertEqual(row_b["Rule"], 0)
            self.assertEqual(row_b["TOTAL"], 1)

            summary_a = next(row for row in summary_rows if row["annotator"] == "a@example.com")
            summary_b = next(row for row in summary_rows if row["annotator"] == "b@example.com")
            self.assertEqual(summary_a["ref_id"], "SUMMARY")
            self.assertEqual(summary_a["Analysis"], 1)
            self.assertEqual(summary_a["Rule"], 1)
            self.assertEqual(summary_a["TOTAL"], 2)
            self.assertEqual(summary_a["AVG_Analysis"], 1.0)
            self.assertEqual(summary_a["AVG_Rule"], 1.0)
            self.assertEqual(summary_a["AVG_TOTAL"], 2.0)
            self.assertEqual(summary_b["Analysis"], 1)
            self.assertEqual(summary_b["Rule"], 0)
            self.assertEqual(summary_b["TOTAL"], 1)
            self.assertEqual(summary_b["AVG_Analysis"], 1.0)
            self.assertEqual(summary_b["AVG_Rule"], 0.0)
            self.assertEqual(summary_b["AVG_TOTAL"], 1.0)


if __name__ == "__main__":
    unittest.main()
