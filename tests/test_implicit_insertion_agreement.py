from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from iaa_scores.implicit_insertion_agreement import (
    compute_implicit_insertion_agreement_for_document_from_pairs,
    evaluate_implicit_insertion_agreement_from_pair_cache,
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


def _span_for_substring(case_text: str, node_id: str, label: str, substring: str) -> dict:
    start = case_text.index(substring)
    end = start + len(substring)
    return _span(node_id, label, substring, start=start, end=end)


def _implicit(node_id: str, label: str = "Analysis", text: str = "Implicit Intermediate Conclusion [abc]") -> dict:
    return _span(node_id, label, text, start=None, end=None)


def _relation(source_id: str, target_id: str) -> dict:
    return {
        "type": "relation",
        "from_id": source_id,
        "to_id": target_id,
        "direction": "right",
    }


class ImplicitInsertionAgreementTests(unittest.TestCase):
    def test_perfect_parent_centered_insertion_agreement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_text = "Premise one. Final conclusion."
            # Raw direction from Label Studio: child -> parent. Normalized: parent -> child.
            results = [
                _span_for_substring(case_text, "u", "Rule", "Premise one."),
                _span_for_substring(case_text, "v", "Conclusion", "Final conclusion."),
                _implicit("i"),
                _relation("u", "i"),
                _relation("i", "v"),
            ]
            _write_export(root, "101_a.json", export_id=101, ref_id=1, annotator="a@example.com", case_text=case_text, results=results)
            _write_export(root, "102_b.json", export_id=102, ref_id=1, annotator="b@example.com", case_text=case_text, results=results)

            from dataset_statistics.parser import load_case_annotation

            annotation_a = load_case_annotation(root / "101_a.json")
            annotation_b = load_case_annotation(root / "102_b.json")
            pair_record = {
                "matches": [
                    {"node_id_a": "u", "node_id_b": "u"},
                    {"node_id_a": "v", "node_id_b": "v"},
                ]
            }

            result, decisions_a, decisions_b = compute_implicit_insertion_agreement_for_document_from_pairs(
                1,
                annotation_a,
                annotation_b,
                pair_record,
                files=(root / "101_a.json", root / "102_b.json"),
            )

            self.assertEqual(result.context_count, 1)
            self.assertEqual(decisions_a, [1])
            self.assertEqual(decisions_b, [1])
            self.assertEqual(result.agreement_count, 1)
            self.assertEqual(result.yes_yes_count, 1)
            self.assertEqual(result.yes_no_count, 0)
            self.assertEqual(result.no_yes_count, 0)
            self.assertEqual(result.no_no_count, 0)
            self.assertEqual(result.positive_agreement, 1.0)
            self.assertTrue(result.negative_agreement != result.negative_agreement)
            self.assertTrue(result.observed_agreement == 1.0)
            self.assertTrue(str(result.kappa) == "nan" or result.kappa != result.kappa)
            self.assertEqual(result.usable_implicit_nodes_by_annotator["a@example.com"], 1)

    def test_direct_vs_implicit_disagreement_on_same_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_text = "Premise one. Final conclusion."
            results_a = [
                _span_for_substring(case_text, "u_a", "Rule", "Premise one."),
                _span_for_substring(case_text, "v_a", "Conclusion", "Final conclusion."),
                _relation("u_a", "v_a"),
            ]
            results_b = [
                _span_for_substring(case_text, "u_b", "Rule", "Premise one."),
                _span_for_substring(case_text, "v_b", "Conclusion", "Final conclusion."),
                _implicit("i_b"),
                _relation("u_b", "i_b"),
                _relation("i_b", "v_b"),
            ]
            _write_export(root, "101_a.json", export_id=101, ref_id=1, annotator="a@example.com", case_text=case_text, results=results_a)
            _write_export(root, "102_b.json", export_id=102, ref_id=1, annotator="b@example.com", case_text=case_text, results=results_b)

            from dataset_statistics.parser import load_case_annotation

            annotation_a = load_case_annotation(root / "101_a.json")
            annotation_b = load_case_annotation(root / "102_b.json")
            pair_record = {
                "matches": [
                    {"node_id_a": "u_a", "node_id_b": "u_b"},
                    {"node_id_a": "v_a", "node_id_b": "v_b"},
                ]
            }

            result, decisions_a, decisions_b = compute_implicit_insertion_agreement_for_document_from_pairs(
                1,
                annotation_a,
                annotation_b,
                pair_record,
                files=(root / "101_a.json", root / "102_b.json"),
            )

            self.assertEqual(result.context_count, 1)
            self.assertEqual(decisions_a, [0])
            self.assertEqual(decisions_b, [1])
            self.assertEqual(result.yes_yes_count, 0)
            self.assertEqual(result.yes_no_count, 0)
            self.assertEqual(result.no_yes_count, 1)
            self.assertEqual(result.no_no_count, 0)
            self.assertEqual(result.positive_agreement, 0.0)
            self.assertEqual(result.negative_agreement, 0.0)
            self.assertEqual(result.observed_agreement, 0.0)
            self.assertEqual(result.expected_agreement, 0.0)
            self.assertEqual(result.kappa, 0.0)

    def test_multiple_implicit_children_of_same_parent_collapse_to_one_binary_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_text = "Premise one. Premise two. Final conclusion."
            results = [
                _span_for_substring(case_text, "u1", "Rule", "Premise one."),
                _span_for_substring(case_text, "u2", "Analysis", "Premise two."),
                _span_for_substring(case_text, "v", "Conclusion", "Final conclusion."),
                _implicit("i1"),
                _implicit("i2"),
                _relation("u1", "i1"),
                _relation("i1", "v"),
                _relation("u2", "i2"),
                _relation("i2", "v"),
            ]
            _write_export(root, "101_a.json", export_id=101, ref_id=1, annotator="a@example.com", case_text=case_text, results=results)
            _write_export(root, "102_b.json", export_id=102, ref_id=1, annotator="b@example.com", case_text=case_text, results=results)

            from dataset_statistics.parser import load_case_annotation

            annotation_a = load_case_annotation(root / "101_a.json")
            annotation_b = load_case_annotation(root / "102_b.json")
            pair_record = {
                "matches": [
                    {"node_id_a": "u1", "node_id_b": "u1"},
                    {"node_id_a": "u2", "node_id_b": "u2"},
                    {"node_id_a": "v", "node_id_b": "v"},
                ]
            }

            result, decisions_a, decisions_b = compute_implicit_insertion_agreement_for_document_from_pairs(
                1,
                annotation_a,
                annotation_b,
                pair_record,
                files=(root / "101_a.json", root / "102_b.json"),
            )

            self.assertEqual(result.context_count, 1)
            self.assertEqual(decisions_a, [1])
            self.assertEqual(result.context_source_counts["multi_implicit_child_parents_a"], 1)
            self.assertEqual(result.context_source_counts["multi_implicit_child_parents_b"], 1)

    def test_chain_counts_only_final_implicit_child_of_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_text = "Premise one. Final conclusion."
            # Raw chain: u -> i -> j -> v. Normalized: v -> j -> i -> u. Only v -> j counts.
            results = [
                _span_for_substring(case_text, "u", "Rule", "Premise one."),
                _span_for_substring(case_text, "v", "Conclusion", "Final conclusion."),
                _implicit("i"),
                _implicit("j"),
                _relation("u", "i"),
                _relation("i", "j"),
                _relation("j", "v"),
            ]
            _write_export(root, "101_a.json", export_id=101, ref_id=1, annotator="a@example.com", case_text=case_text, results=results)
            _write_export(root, "102_b.json", export_id=102, ref_id=1, annotator="b@example.com", case_text=case_text, results=results)

            from dataset_statistics.parser import load_case_annotation

            annotation_a = load_case_annotation(root / "101_a.json")
            annotation_b = load_case_annotation(root / "102_b.json")
            pair_record = {
                "matches": [
                    {"node_id_a": "u", "node_id_b": "u"},
                    {"node_id_a": "v", "node_id_b": "v"},
                ]
            }

            result, decisions_a, decisions_b = compute_implicit_insertion_agreement_for_document_from_pairs(
                1,
                annotation_a,
                annotation_b,
                pair_record,
                files=(root / "101_a.json", root / "102_b.json"),
            )

            self.assertEqual(result.context_count, 1)
            self.assertEqual(decisions_a, [1])
            self.assertEqual(result.context_source_counts["chain_child_parent_contexts_a"], 1)

    def test_pair_cache_fingerprint_mismatch_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            input_dir = repo_root / "annotations" / "final_annotations_iaa_set"
            input_dir.mkdir(parents=True, exist_ok=True)
            case_text = "Premise one. Final conclusion."
            results = [
                _span_for_substring(case_text, "u", "Rule", "Premise one."),
                _span_for_substring(case_text, "v", "Conclusion", "Final conclusion."),
                _relation("u", "v"),
            ]
            _write_export(input_dir, "101_a.json", export_id=101, ref_id=1, annotator="a@example.com", case_text=case_text, results=results)
            _write_export(input_dir, "102_b.json", export_id=102, ref_id=1, annotator="b@example.com", case_text=case_text, results=results)

            bad_pair_cache = {
                "dataset_fingerprint": "wrong",
                "documents": [
                    {
                        "ref_id": 1,
                        "matches": [
                            {"node_id_a": "u", "node_id_b": "u"},
                            {"node_id_a": "v", "node_id_b": "v"},
                        ],
                    }
                ],
            }

            with self.assertRaisesRegex(ValueError, "fingerprint does not match"):
                evaluate_implicit_insertion_agreement_from_pair_cache(input_dir, bad_pair_cache)


if __name__ == "__main__":
    unittest.main()
