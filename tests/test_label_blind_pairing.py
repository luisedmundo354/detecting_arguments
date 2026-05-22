from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

LABEL_BLIND_DEPS_AVAILABLE = (
    importlib.util.find_spec("scipy") is not None
)
if LABEL_BLIND_DEPS_AVAILABLE:
    from dataset_statistics.modules import AnnotationSpan

    from iaa_scores.label_blind_pairing_test import (
        _explicit_spans,
        build_or_load_label_blind_pair_cache,
        label_blind_pair_cache_path,
        load_label_blind_pair_cache,
        summarize_label_blind_pair_cache,
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


def _span(node_id: str, label: str, text: str, *, start: int, end: int) -> dict:
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


@unittest.skipUnless(LABEL_BLIND_DEPS_AVAILABLE, "label-blind pairing dependencies are unavailable")
class LabelBlindPairingTests(unittest.TestCase):
    def test_explicit_span_sorting_preserves_zero_offsets(self) -> None:
        spans = [
            AnnotationSpan("later", "Rule", "Later", start=10, end=15),
            AnnotationSpan("first", "Rule", "First", start=0, end=5),
        ]

        self.assertEqual([span.node_id for span in _explicit_spans(spans)], ["first", "later"])

    def test_label_blind_pairing_can_match_different_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            input_dir = repo_root / "annotations" / "final_annotations_iaa_set"
            input_dir.mkdir(parents=True, exist_ok=True)
            case_text = (
                "Alpha establishes the governing rule. "
                "Beta applies that rule to the record."
            )
            _write_export(
                input_dir,
                "101_a.json",
                export_id=101,
                ref_id=1,
                annotator="a@example.com",
                case_text=case_text,
                results=[
                    _span_for_substring(
                        case_text,
                        "a_rule",
                        "Rule",
                        "Alpha establishes the governing rule.",
                    ),
                    _span_for_substring(
                        case_text,
                        "a_analysis",
                        "Analysis",
                        "Beta applies that rule to the record.",
                    ),
                ],
            )
            _write_export(
                input_dir,
                "102_b.json",
                export_id=102,
                ref_id=1,
                annotator="b@example.com",
                case_text=case_text,
                results=[
                    _span_for_substring(
                        case_text,
                        "b_analysis_1",
                        "Analysis",
                        "Alpha establishes the governing rule.",
                    ),
                    _span_for_substring(
                        case_text,
                        "b_analysis_2",
                        "Analysis",
                        "Beta applies that rule to the record.",
                    ),
                ],
            )
            semantic_cache = {
                "manifest": {
                    "model": "embed-v4.0",
                    "input_type": "classification",
                    "output_dimension": 2,
                },
                "text_to_embedding": {
                    "Alpha establishes the governing rule.": np.array([1.0, 0.0]),
                    "Beta applies that rule to the record.": np.array([0.0, 1.0]),
                },
            }

            pair_cache = build_or_load_label_blind_pair_cache(
                repo_root=repo_root,
                input_dir=input_dir,
                backend="semantic",
                overwrite=True,
                metric=None,
                min_sim=0.1,
                semantic_cache=semantic_cache,
            )

            self.assertFalse(pair_cache["label_restricted"])
            cache_path = label_blind_pair_cache_path(
                repo_root=repo_root,
                input_dir=input_dir,
                backend="semantic",
                metric=None,
                min_sim=0.1,
                semantic_cache=semantic_cache,
            )
            self.assertTrue(cache_path.exists())
            self.assertTrue(cache_path.name.endswith("__label_blind.json"))
            self.assertNotIn("__label_restricted.json", cache_path.name)

            loaded_cache = load_label_blind_pair_cache(
                repo_root=repo_root,
                input_dir=input_dir,
                backend="semantic",
                metric=None,
                min_sim=0.1,
                semantic_cache=semantic_cache,
            )
            self.assertFalse(loaded_cache["label_restricted"])

            matches = pair_cache["documents"][0]["matches"]
            cross_label_matches = [
                match for match in matches if match["label_a"] != match["label_b"]
            ]
            self.assertEqual(len(cross_label_matches), 1)
            self.assertEqual(cross_label_matches[0]["label_a"], "Rule")
            self.assertEqual(cross_label_matches[0]["label_b"], "Analysis")
            self.assertFalse(cross_label_matches[0]["labels_match"])

            summary = summarize_label_blind_pair_cache(pair_cache)
            self.assertEqual(summary["cross_label"]["cross_label_count"], 1)
            self.assertEqual(summary["label_confusion"]["Rule"]["Analysis"], 1)
            self.assertEqual(summary["boundary"]["overall"]["matched_count"], 2)
            self.assertEqual(summary["boundary"]["overall"]["f1"], 1.0)

            corrupted_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            corrupted_payload["label_restricted"] = True
            cache_path.write_text(json.dumps(corrupted_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "label_restricted=False"):
                load_label_blind_pair_cache(
                    repo_root=repo_root,
                    input_dir=input_dir,
                    backend="semantic",
                    metric=None,
                    min_sim=0.1,
                    semantic_cache=semantic_cache,
                )


if __name__ == "__main__":
    unittest.main()
