from __future__ import annotations

import json
import math
import unittest

from dataset_statistics.modules import AnnotationSpan, CaseAnnotation, RelationEdge

from iaa_scores.diagnostics.segmentation_style_mismatch import (
    _strict_json_value,
    build_overlap_groups_for_document,
    compute_group_edge_agreement_for_document,
)


def _span(node_id: str, label: str, text: str, start: int, end: int) -> AnnotationSpan:
    return AnnotationSpan(
        node_id=node_id,
        label=label,
        text=text,
        start=start,
        end=end,
    )


def _relation(source_id: str, target_id: str) -> RelationEdge:
    return RelationEdge(source_id=source_id, target_id=target_id)


def _annotation(
    annotator: str,
    case_text: str,
    spans: list[AnnotationSpan],
    relations: list[RelationEdge] | None = None,
) -> CaseAnnotation:
    return CaseAnnotation(
        export_id=annotator,
        annotation_id=None,
        ref_id=1,
        source_file=f"{annotator}.json",
        annotator=annotator,
        assigned_to=(),
        case_text=case_text,
        spans=tuple(spans),
        relations=tuple(relations or []),
        header_text="",
        year=None,
        subtype_368=(),
    )


class SegmentationStyleMismatchTests(unittest.TestCase):
    def test_many_to_one_group_is_disjoint(self) -> None:
        case_text = "Alpha one. Alpha two."
        annotation_a = _annotation(
            "a@example.com",
            case_text,
            [
                _span("a1", "Rule", "Alpha one.", 0, 10),
                _span("a2", "Rule", "Alpha two.", 11, 21),
            ],
        )
        annotation_b = _annotation(
            "b@example.com",
            case_text,
            [_span("b1", "Rule", "Alpha one. Alpha two.", 0, 21)],
        )

        grouping = build_overlap_groups_for_document(
            1,
            annotation_a,
            annotation_b,
            label_mode="same-label",
            overlap_threshold=0.5,
        )

        self.assertEqual(len(grouping.groups), 1)
        self.assertEqual(grouping.groups[0].arity, "2:1")
        self.assertEqual(grouping.node_to_group_a["a1"], grouping.node_to_group_a["a2"])
        self.assertEqual(grouping.node_to_group_b["b1"], grouping.groups[0].group_id)
        self.assertEqual(len(set(grouping.node_to_group_b.values())), 1)

    def test_one_to_many_group_is_disjoint(self) -> None:
        case_text = "Alpha one. Alpha two."
        annotation_a = _annotation(
            "a@example.com",
            case_text,
            [_span("a1", "Rule", "Alpha one. Alpha two.", 0, 21)],
        )
        annotation_b = _annotation(
            "b@example.com",
            case_text,
            [
                _span("b1", "Rule", "Alpha one.", 0, 10),
                _span("b2", "Rule", "Alpha two.", 11, 21),
            ],
        )

        grouping = build_overlap_groups_for_document(
            1,
            annotation_a,
            annotation_b,
            label_mode="same-label",
            overlap_threshold=0.5,
        )

        self.assertEqual(len(grouping.groups), 1)
        self.assertEqual(grouping.groups[0].arity, "1:2")
        self.assertEqual(grouping.node_to_group_b["b1"], grouping.node_to_group_b["b2"])
        self.assertEqual(grouping.node_to_group_a["a1"], grouping.groups[0].group_id)

    def test_group_level_edge_agreement_recovers_split_merge_edge(self) -> None:
        case_text = "Alpha one. Alpha two. Target."
        annotation_a = _annotation(
            "a@example.com",
            case_text,
            [
                _span("a_s1", "Rule", "Alpha one.", 0, 10),
                _span("a_s2", "Rule", "Alpha two.", 11, 21),
                _span("a_t", "Analysis", "Target.", 22, 29),
            ],
            [_relation("a_t", "a_s1")],
        )
        annotation_b = _annotation(
            "b@example.com",
            case_text,
            [
                _span("b_s", "Rule", "Alpha one. Alpha two.", 0, 21),
                _span("b_t", "Analysis", "Target.", 22, 29),
            ],
            [_relation("b_t", "b_s")],
        )
        grouping = build_overlap_groups_for_document(
            1,
            annotation_a,
            annotation_b,
            label_mode="same-label",
            overlap_threshold=0.5,
        )

        summary = compute_group_edge_agreement_for_document(
            1,
            annotation_a,
            annotation_b,
            grouping,
            context_mode="edge-union",
        )

        self.assertEqual(summary["contexts"], 1)
        self.assertEqual(summary["yes_yes"], 1)
        self.assertEqual(summary["a_only"], 0)
        self.assertEqual(summary["b_only"], 0)
        self.assertEqual(summary["observed"], 1.0)
        self.assertTrue(math.isnan(summary["kappa"]))

        all_pairs = compute_group_edge_agreement_for_document(
            1,
            annotation_a,
            annotation_b,
            grouping,
            context_mode="all-pairs",
        )
        self.assertEqual(all_pairs["contexts"], 2)
        self.assertEqual(all_pairs["yes_yes"], 1)
        self.assertEqual(all_pairs["no_no"], 1)

    def test_internal_edges_are_excluded_and_counted(self) -> None:
        case_text = "Alpha one. Alpha two."
        annotation_a = _annotation(
            "a@example.com",
            case_text,
            [
                _span("a1", "Rule", "Alpha one.", 0, 10),
                _span("a2", "Rule", "Alpha two.", 11, 21),
            ],
            [_relation("a2", "a1")],
        )
        annotation_b = _annotation(
            "b@example.com",
            case_text,
            [_span("b1", "Rule", "Alpha one. Alpha two.", 0, 21)],
        )
        grouping = build_overlap_groups_for_document(
            1,
            annotation_a,
            annotation_b,
            label_mode="same-label",
            overlap_threshold=0.5,
        )

        summary = compute_group_edge_agreement_for_document(
            1,
            annotation_a,
            annotation_b,
            grouping,
            context_mode="edge-union",
        )

        self.assertEqual(summary["contexts"], 0)
        self.assertEqual(
            summary["edge_loss_by_annotator"]["a@example.com"]["internal_edges_within_group"],
            1,
        )

    def test_lost_edges_due_to_ungrouped_endpoint_are_counted(self) -> None:
        case_text = "Alpha one. Target."
        annotation_a = _annotation(
            "a@example.com",
            case_text,
            [
                _span("a1", "Rule", "Alpha one.", 0, 10),
                _span("a2", "Analysis", "Target.", 11, 18),
            ],
            [_relation("a2", "a1")],
        )
        annotation_b = _annotation(
            "b@example.com",
            case_text,
            [_span("b1", "Rule", "Alpha one.", 0, 10)],
        )
        grouping = build_overlap_groups_for_document(
            1,
            annotation_a,
            annotation_b,
            label_mode="same-label",
            overlap_threshold=0.5,
        )

        summary = compute_group_edge_agreement_for_document(
            1,
            annotation_a,
            annotation_b,
            grouping,
            context_mode="edge-union",
        )

        self.assertEqual(summary["contexts"], 0)
        self.assertEqual(
            summary["edge_loss_by_annotator"]["a@example.com"]["lost_edges_due_to_ungrouped_endpoint"],
            1,
        )

    def test_transitive_many_to_many_component_is_one_disjoint_group(self) -> None:
        case_text = "abcdefghijklmnopqr"
        annotation_a = _annotation(
            "a@example.com",
            case_text,
            [
                _span("a1", "Rule", "abcdefghij", 0, 10),
                _span("a2", "Rule", "ijklmnopqr", 8, 18),
            ],
        )
        annotation_b = _annotation(
            "b@example.com",
            case_text,
            [
                _span("b1", "Rule", "abcdefghijkl", 0, 12),
                _span("b2", "Rule", "klmnopqr", 10, 18),
            ],
        )

        grouping = build_overlap_groups_for_document(
            1,
            annotation_a,
            annotation_b,
            label_mode="same-label",
            overlap_threshold=0.4,
        )

        self.assertEqual(len(grouping.groups), 1)
        self.assertEqual(grouping.groups[0].arity, "2:2")
        self.assertEqual(set(grouping.node_to_group_a), {"a1", "a2"})
        self.assertEqual(set(grouping.node_to_group_b), {"b1", "b2"})

    def test_label_blind_mode_can_group_cross_label_spans(self) -> None:
        case_text = "Alpha one."
        annotation_a = _annotation(
            "a@example.com",
            case_text,
            [_span("a1", "Rule", "Alpha one.", 0, 10)],
        )
        annotation_b = _annotation(
            "b@example.com",
            case_text,
            [_span("b1", "Analysis", "Alpha one.", 0, 10)],
        )

        same_label = build_overlap_groups_for_document(
            1,
            annotation_a,
            annotation_b,
            label_mode="same-label",
            overlap_threshold=0.5,
        )
        label_blind = build_overlap_groups_for_document(
            1,
            annotation_a,
            annotation_b,
            label_mode="label-blind",
            overlap_threshold=0.5,
        )

        self.assertEqual(len(same_label.groups), 0)
        self.assertEqual(len(label_blind.groups), 1)
        self.assertEqual(label_blind.groups[0].label, "mixed:Analysis|Rule")

    def test_invalid_overlap_threshold_is_rejected_for_direct_grouping(self) -> None:
        case_text = "Alpha one."
        annotation_a = _annotation(
            "a@example.com",
            case_text,
            [_span("a1", "Rule", "Alpha one.", 0, 10)],
        )
        annotation_b = _annotation(
            "b@example.com",
            case_text,
            [_span("b1", "Rule", "Alpha one.", 0, 10)],
        )

        with self.assertRaisesRegex(ValueError, "overlap_threshold"):
            build_overlap_groups_for_document(
                1,
                annotation_a,
                annotation_b,
                overlap_threshold=1.1,
            )

    def test_strict_json_value_replaces_non_standard_nan_values(self) -> None:
        payload = {
            "observed": float("nan"),
            "nested": [1.0, float("inf"), -float("inf")],
        }

        strict_payload = _strict_json_value(payload)
        json.dumps(strict_payload, allow_nan=False)

        self.assertIsNone(strict_payload["observed"])
        self.assertEqual(strict_payload["nested"], [1.0, None, None])


if __name__ == "__main__":
    unittest.main()
