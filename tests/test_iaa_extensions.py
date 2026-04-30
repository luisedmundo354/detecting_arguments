from __future__ import annotations

import json
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

from iaa_scores.sentence_corpus import build_sentence_corpus

EDGE_DEPS_AVAILABLE = (
    importlib.util.find_spec("scipy") is not None
    and importlib.util.find_spec("abydos") is not None
)
if EDGE_DEPS_AVAILABLE:
    from iaa_scores.edge_agreement import evaluate_edge_agreement


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


def _span_for_substring(case_text: str, node_id: str, label: str, substring: str) -> dict:
    start = case_text.index(substring)
    end = start + len(substring)
    return _span(node_id, label, substring, start=start, end=end)


def _relation(source_id: str, target_id: str) -> dict:
    return {
        "type": "relation",
        "from_id": source_id,
        "to_id": target_id,
        "direction": "right",
    }


class IAAExtensionTests(unittest.TestCase):
    def test_sentence_corpus_projects_to_sentence_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_text = (
                "Alpha establishes the governing rule. "
                "Beta applies that rule to the record."
            )
            results_a = [
                _span_for_substring(
                    case_text,
                    "r1",
                    "Rule",
                    "Alpha establishes the governing rule.",
                ),
                _span_for_substring(
                    case_text,
                    "a1",
                    "Analysis",
                    "Beta applies that rule to the record.",
                ),
                _relation("r1", "a1"),
            ]
            results_b = [
                _span_for_substring(
                    case_text,
                    "r2",
                    "Rule",
                    "Alpha establishes the governing rule.",
                ),
                _span_for_substring(
                    case_text,
                    "a2",
                    "Analysis",
                    "Beta applies that rule to the record.",
                ),
                _relation("r2", "a2"),
            ]
            _write_export(
                root,
                "101_a.json",
                export_id=101,
                ref_id=1,
                annotator="a@example.com",
                case_text=case_text,
                results=results_a,
            )
            _write_export(
                root,
                "102_b.json",
                export_id=102,
                ref_id=1,
                annotator="b@example.com",
                case_text=case_text,
                results=results_b,
            )

            corpus = build_sentence_corpus(root, min_annotators=2)

            self.assertEqual(corpus.categories, ["Analysis", "Rule"])
            self.assertEqual(
                corpus.doc_spans[1]["a@example.com"]["Rule"],
                ["Alpha establishes the governing rule."],
            )
            self.assertEqual(
                corpus.doc_spans[1]["a@example.com"]["Analysis"],
                ["Beta applies that rule to the record."],
            )
            self.assertEqual(
                corpus.doc_offsets[1]["b@example.com"]["Analysis"],
                [(37, 75)],
            )

    def test_sentence_corpus_fails_if_implicit_node_still_has_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_text = "Alpha establishes the governing rule."
            broken_results = [
                _span_for_substring(
                    case_text,
                    "r1",
                    "Rule",
                    "Alpha establishes the governing rule.",
                ),
                _span(
                    "i1",
                    "Analysis",
                    "Implicit Intermediate Conclusion [abc]",
                    start=0,
                    end=10,
                ),
            ]
            for export_id, annotator, filename in [
                (101, "a@example.com", "101_a.json"),
                (102, "b@example.com", "102_b.json"),
            ]:
                _write_export(
                    root,
                    filename,
                    export_id=export_id,
                    ref_id=1,
                    annotator=annotator,
                    case_text=case_text,
                    results=broken_results,
                )

            with self.assertRaisesRegex(ValueError, "Implicit intermediate conclusion still has offsets"):
                build_sentence_corpus(root, min_annotators=2)

    @unittest.skipUnless(EDGE_DEPS_AVAILABLE, "edge-agreement dependencies are unavailable")
    def test_edge_agreement_counts_missing_direct_edge_as_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_text = (
                "Alpha establishes the governing rule. "
                "Beta applies that rule to the record."
            )
            results_a = [
                _span_for_substring(
                    case_text,
                    "r1",
                    "Rule",
                    "Alpha establishes the governing rule.",
                ),
                _span_for_substring(
                    case_text,
                    "a1",
                    "Analysis",
                    "Beta applies that rule to the record.",
                ),
                _relation("r1", "a1"),
            ]
            results_b = [
                _span_for_substring(
                    case_text,
                    "r2",
                    "Rule",
                    "Alpha establishes the governing rule.",
                ),
                _span_for_substring(
                    case_text,
                    "a2",
                    "Analysis",
                    "Beta applies that rule to the record.",
                ),
            ]
            _write_export(
                root,
                "101_a.json",
                export_id=101,
                ref_id=1,
                annotator="a@example.com",
                case_text=case_text,
                results=results_a,
            )
            _write_export(
                root,
                "102_b.json",
                export_id=102,
                ref_id=1,
                annotator="b@example.com",
                case_text=case_text,
                results=results_b,
            )

            scores = evaluate_edge_agreement(root, metric="yujianbo", min_sim=0.1)
            doc_result = scores.per_doc[1]

            self.assertEqual(doc_result.context_count, 1)
            self.assertEqual(doc_result.agreement_count, 0)
            self.assertEqual(doc_result.positive_rate_by_annotator["a@example.com"], 1.0)
            self.assertEqual(doc_result.positive_rate_by_annotator["b@example.com"], 0.0)
            self.assertEqual(doc_result.observed_agreement, 0.0)
            self.assertEqual(doc_result.expected_agreement, 0.0)
            self.assertEqual(doc_result.kappa, 0.0)

    @unittest.skipUnless(EDGE_DEPS_AVAILABLE, "edge-agreement dependencies are unavailable")
    def test_edge_agreement_is_nan_without_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_text = "Alpha establishes the governing rule. Beta concludes the case."
            results_a = [
                _span_for_substring(case_text, "r1", "Rule", "Alpha establishes the governing rule."),
                _span_for_substring(case_text, "c1", "Conclusion", "Beta concludes the case."),
            ]
            results_b = [
                _span_for_substring(case_text, "r2", "Rule", "Alpha establishes the governing rule."),
                _span_for_substring(case_text, "c2", "Conclusion", "Beta concludes the case."),
            ]
            _write_export(root, "101_a.json", export_id=101, ref_id=1, annotator="a@example.com", case_text=case_text, results=results_a)
            _write_export(root, "102_b.json", export_id=102, ref_id=1, annotator="b@example.com", case_text=case_text, results=results_b)

            scores = evaluate_edge_agreement(root, metric="yujianbo", min_sim=0.1)
            doc_result = scores.per_doc[1]

            self.assertEqual(doc_result.context_count, 0)
            self.assertTrue(math.isnan(doc_result.observed_agreement))
            self.assertTrue(math.isnan(doc_result.kappa))


if __name__ == "__main__":
    unittest.main()
