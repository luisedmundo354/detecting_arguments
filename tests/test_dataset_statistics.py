from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset_statistics import (
    build_sentence_dataset,
    compute_dataset_statistics,
    describe_annotation_schema,
)


def _write_export(
    root: Path,
    filename: str,
    *,
    export_id: int,
    ref_id: int,
    annotator: str,
    case_text: str,
    spans: list[dict],
    relations: list[dict],
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
        "result": spans + relations,
    }
    (root / filename).write_text(json.dumps(payload), encoding="utf-8")


def _span(
    node_id: str,
    label: str,
    text: str,
    *,
    start: int | None,
    end: int | None,
    parent_id: str | None = None,
) -> dict:
    item = {
        "id": node_id,
        "type": "labels",
        "origin": "manual",
        "from_name": "label",
        "to_name": "text",
        "value": {
            "start": start,
            "end": end,
            "text": text,
            "labels": [label],
        },
    }
    if parent_id is not None:
        item["parentID"] = parent_id
    return item


def _relation(source_id: str, target_id: str, direction: str = "right") -> dict:
    return {
        "type": "relation",
        "from_id": source_id,
        "to_id": target_id,
        "direction": direction,
    }


def _span_for_substring(case_text: str, node_id: str, label: str, substring: str) -> dict:
    start = case_text.index(substring)
    end = start + len(substring)
    return _span(node_id, label, substring, start=start, end=end)


class DatasetStatisticsTests(unittest.TestCase):
    def test_compute_dataset_statistics_export_and_case_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_text = (
                "Alpha establishes a rule. "
                "Beta provides detailed analysis. "
                "Gamma ends the dispute."
            )

            _write_export(
                root,
                "101_annotator_a.json",
                export_id=101,
                ref_id=1,
                annotator="annotator_a@example.com",
                case_text=case_text,
                spans=[
                    _span_for_substring(
                        case_text, "r1", "Rule", "Alpha establishes a rule."
                    ),
                    _span_for_substring(
                        case_text,
                        "a1",
                        "Analysis",
                        "Beta provides detailed analysis.",
                    ),
                    _span(
                        "i1",
                        "Analysis",
                        "Implicit Intermediate Conclusion [abc]",
                        start=None,
                        end=None,
                    ),
                    _span_for_substring(
                        case_text, "c1", "Analysis", "Gamma ends the dispute."
                    ),
                ],
                relations=[_relation("r1", "c1"), _relation("a1", "c1")],
            )

            _write_export(
                root,
                "102_annotator_b.json",
                export_id=102,
                ref_id=1,
                annotator="annotator_b@example.com",
                case_text=case_text,
                spans=[
                    _span_for_substring(
                        case_text, "r2", "Rule", "Alpha establishes a rule."
                    ),
                    _span_for_substring(
                        case_text,
                        "a2",
                        "Analysis",
                        "Beta provides detailed analysis.",
                    ),
                    _span_for_substring(
                        case_text, "c2", "Conclusion", "Gamma ends the dispute."
                    ),
                ],
                relations=[_relation("r2", "c2"), _relation("a2", "c2")],
            )

            _write_export(
                root,
                "201_annotator_a.json",
                export_id=201,
                ref_id=2,
                annotator="annotator_a@example.com",
                case_text="Case Two 1985. Delta rule. Epsilon conclusion.",
                spans=[
                    _span("r3", "Rule", "Delta rule.", start=15, end=26),
                    _span("c3", "Conclusion", "Epsilon conclusion.", start=27, end=46),
                ],
                relations=[_relation("r3", "c3")],
            )

            export_stats = compute_dataset_statistics(root, dataset_name="synthetic", view="export")
            case_stats = compute_dataset_statistics(root, dataset_name="synthetic", view="case")

            self.assertEqual(export_stats.case_count, 3)
            self.assertEqual(export_stats.export_count, 3)
            self.assertEqual(export_stats.unique_ref_id_count, 2)
            self.assertEqual(export_stats.double_annotated_case_count, 1)
            self.assertEqual(export_stats.label_distribution["Rule"]["count"], 3)
            self.assertEqual(export_stats.label_distribution["Analysis"]["count"], 3)
            self.assertEqual(export_stats.all_node_label_distribution["Analysis"]["count"], 4)
            self.assertEqual(export_stats.summary_metrics["total_implicit_insertions"], 1)
            self.assertEqual(export_stats.summary_metrics["total_disconnected_spans"], 1)

            self.assertEqual(case_stats.case_count, 2)
            self.assertEqual(case_stats.unique_ref_id_count, 2)
            case_one = next(item for item in case_stats.case_statistics if item.ref_id == 1)
            self.assertEqual(case_one.annotation_count, 2)
            self.assertEqual(case_one.node_count, 3.5)
            self.assertEqual(case_one.implicit_insertion_count, 0.5)
            self.assertEqual(case_one.explicit_span_counts_by_label["Rule"], 1)
            self.assertEqual(case_one.explicit_span_counts_by_label["Analysis"], 1.5)
            self.assertEqual(case_one.explicit_span_counts_by_label["Conclusion"], 0.5)
            self.assertEqual(case_one.sentence_counts_by_label["Analysis"], 1.5)

    def test_build_sentence_dataset_case_view_preserves_votes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_text = (
                "Alpha establishes a rule. "
                "Beta provides detailed analysis. "
                "Gamma ends the dispute."
            )
            _write_export(
                root,
                "101_annotator_a.json",
                export_id=101,
                ref_id=1,
                annotator="annotator_a@example.com",
                case_text=case_text,
                spans=[
                    _span_for_substring(
                        case_text, "r1", "Rule", "Alpha establishes a rule."
                    ),
                    _span_for_substring(
                        case_text,
                        "a1",
                        "Analysis",
                        "Beta provides detailed analysis.",
                    ),
                    _span_for_substring(
                        case_text, "c1", "Analysis", "Gamma ends the dispute."
                    ),
                ],
                relations=[_relation("r1", "c1"), _relation("a1", "c1")],
            )
            _write_export(
                root,
                "102_annotator_b.json",
                export_id=102,
                ref_id=1,
                annotator="annotator_b@example.com",
                case_text=case_text,
                spans=[
                    _span_for_substring(
                        case_text, "r2", "Rule", "Alpha establishes a rule."
                    ),
                    _span_for_substring(
                        case_text,
                        "a2",
                        "Analysis",
                        "Beta provides detailed analysis.",
                    ),
                    _span_for_substring(
                        case_text, "c2", "Conclusion", "Gamma ends the dispute."
                    ),
                ],
                relations=[_relation("r2", "c2"), _relation("a2", "c2")],
            )

            records = build_sentence_dataset(root, dataset_name="synthetic", view="case")
            self.assertEqual(len(records), 3)
            self.assertEqual(records[0].passage_id, "1::SENT_00000")
            self.assertEqual(records[0].label, "Rule")
            self.assertEqual(records[2].label_votes, {"Analysis": 1, "Conclusion": 1})
            self.assertEqual(
                set(records[2].annotator_labels),
                {"101_annotator_a.json", "102_annotator_b.json"},
            )

    def test_describe_annotation_schema_reports_result_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_export(
                root,
                "101_annotator_a.json",
                export_id=101,
                ref_id=1,
                annotator="annotator_a@example.com",
                case_text="Alpha rule. Gamma end.",
                spans=[
                    _span("r1", "Rule", "Alpha rule.", start=0, end=11),
                    _span(
                        "i1",
                        "Analysis",
                        "Implicit Intermediate Conclusion [abc]",
                        start=None,
                        end=None,
                        parent_id="r1",
                    ),
                ],
                relations=[_relation("r1", "i1")],
            )

            schema = describe_annotation_schema([root])

            self.assertIn("task", schema)
            self.assertIn("data", schema["task"])
            self.assertIn("ref_id", schema["task"]["data"])
            result_by_type = {item["type"]: item for item in schema["result"]}
            self.assertIn("labels", result_by_type)
            self.assertIn("relation", result_by_type)
            self.assertIn("parentID", result_by_type["labels"])
            self.assertIn("value", result_by_type["labels"])


if __name__ == "__main__":
    unittest.main()
