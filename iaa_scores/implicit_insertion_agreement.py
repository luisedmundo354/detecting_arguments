"""Implicit intermediate-conclusion insertion agreement with corrected edge direction."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

from dataset_statistics.modules import CaseAnnotation

from .annotation_graphs import load_annotation_graphs
from .cache_utils import compute_dataset_fingerprint
from .models import DocId, ImplicitInsertionDocResult, ImplicitInsertionScores
from .relation_utils import normalized_adjacency


def evaluate_implicit_insertion_agreement_from_pair_cache(
    root: Path,
    pair_cache: Mapping[str, object],
    *,
    min_annotators: int = 2,
    sample_context_limit: int = 5,
) -> ImplicitInsertionScores:
    """Compute implicit insertion agreement using a persisted pair cache."""

    dataset_fingerprint = compute_dataset_fingerprint(root)
    pair_cache_fingerprint = pair_cache.get("dataset_fingerprint")
    if pair_cache_fingerprint != dataset_fingerprint:
        raise ValueError(
            "Pair cache fingerprint does not match the repaired annotation graph dataset."
        )

    graphs_by_doc = load_annotation_graphs(
        root,
        min_annotators=min_annotators,
        require_repaired_implicit_offsets=True,
    )
    document_records = {
        document_record["ref_id"]: document_record
        for document_record in pair_cache.get("documents", [])
    }
    if set(document_records) != set(graphs_by_doc):
        raise ValueError(
            "Pair cache documents do not match the repaired annotation graph documents."
        )

    per_doc: Dict[DocId, ImplicitInsertionDocResult] = {}
    overall_decisions_a: List[int] = []
    overall_decisions_b: List[int] = []
    overall_usable_by_annotator = Counter()
    overall_excluded_by_annotator: Dict[str, Counter[str]] = {}
    canonical_annotators: Tuple[str, str] | None = None

    for ref_id, records_by_annotator in sorted(graphs_by_doc.items()):
        annotators = tuple(sorted(records_by_annotator))
        if len(annotators) != 2:
            raise ValueError(
                f"Implicit insertion agreement requires exactly two annotators per document; "
                f"ref_id {ref_id} has {len(annotators)} annotators."
            )
        if canonical_annotators is None:
            canonical_annotators = annotators
        elif annotators != canonical_annotators:
            raise ValueError(
                "Implicit insertion agreement requires a consistent annotator pair across documents; "
                f"expected {canonical_annotators}, found {annotators} for ref_id {ref_id}."
            )

        record_a = records_by_annotator[annotators[0]]
        record_b = records_by_annotator[annotators[1]]
        doc_result, decisions_a, decisions_b = compute_implicit_insertion_agreement_for_document_from_pairs(
            ref_id,
            record_a.annotation,
            record_b.annotation,
            document_records[ref_id],
            files=(record_a.path, record_b.path),
            sample_context_limit=sample_context_limit,
        )
        per_doc[ref_id] = doc_result
        overall_decisions_a.extend(decisions_a)
        overall_decisions_b.extend(decisions_b)
        for annotator, usable in doc_result.usable_implicit_nodes_by_annotator.items():
            overall_usable_by_annotator[annotator] += usable
        for annotator, reason_counts in doc_result.excluded_implicit_nodes_by_annotator.items():
            overall_excluded_by_annotator.setdefault(annotator, Counter()).update(reason_counts)

    overall_context_count = len(overall_decisions_a)
    overall_agreement_count = sum(
        1 for decision_a, decision_b in zip(overall_decisions_a, overall_decisions_b) if decision_a == decision_b
    )
    overall_yes_yes, overall_yes_no, overall_no_yes, overall_no_no = _binary_cell_counts(
        overall_decisions_a,
        overall_decisions_b,
    )
    overall_observed, overall_expected, overall_kappa = _compute_binary_agreement(
        overall_decisions_a,
        overall_decisions_b,
    )
    overall_positive_agreement, overall_negative_agreement = _compute_positive_negative_agreement(
        overall_yes_yes,
        overall_yes_no,
        overall_no_yes,
        overall_no_no,
    )
    overall_insertion_rate_by_annotator = {}
    if canonical_annotators is not None:
        overall_insertion_rate_by_annotator = {
            canonical_annotators[0]: _safe_mean_binary(overall_decisions_a),
            canonical_annotators[1]: _safe_mean_binary(overall_decisions_b),
        }

    return ImplicitInsertionScores(
        per_doc=per_doc,
        overall_context_count=overall_context_count,
        overall_agreement_count=overall_agreement_count,
        overall_yes_yes_count=overall_yes_yes,
        overall_yes_no_count=overall_yes_no,
        overall_no_yes_count=overall_no_yes,
        overall_no_no_count=overall_no_no,
        overall_observed_agreement=overall_observed,
        overall_expected_agreement=overall_expected,
        overall_kappa=overall_kappa,
        overall_positive_agreement=overall_positive_agreement,
        overall_negative_agreement=overall_negative_agreement,
        overall_insertion_rate_by_annotator=overall_insertion_rate_by_annotator,
        overall_usable_implicit_nodes_by_annotator=dict(sorted(overall_usable_by_annotator.items())),
        overall_excluded_implicit_nodes_by_annotator={
            annotator: dict(sorted(reason_counts.items()))
            for annotator, reason_counts in sorted(overall_excluded_by_annotator.items())
        },
    )


def compute_implicit_insertion_agreement_for_document_from_pairs(
    ref_id: DocId,
    annotation_a: CaseAnnotation,
    annotation_b: CaseAnnotation,
    document_record: Mapping[str, object],
    *,
    files: Sequence[Path],
    sample_context_limit: int = 5,
) -> Tuple[ImplicitInsertionDocResult, List[int], List[int]]:
    """Compute insertion agreement for one doubly annotated document."""

    if annotation_a.case_text != annotation_b.case_text:
        raise ValueError(
            f"Case text mismatch for ref_id {ref_id}; repaired IAA exports must share the same canonical text."
        )

    match_a_to_b, match_b_to_a = _build_match_maps_from_pair_record(ref_id, document_record)
    matched_parents_a = set(match_a_to_b)
    matched_parents_b = set(match_b_to_a)

    parent_state_a = _parent_centered_state(
        annotation_a,
        matched_parent_ids=matched_parents_a,
    )
    parent_state_b = _parent_centered_state(
        annotation_b,
        matched_parent_ids=matched_parents_b,
    )

    context_keys = []
    for parent_a in sorted(match_a_to_b):
        parent_b = match_a_to_b[parent_a]
        if parent_state_a["structured_parent_contexts"].get(parent_a, False) or parent_state_b["structured_parent_contexts"].get(parent_b, False):
            context_keys.append((parent_a, parent_b))

    decisions_a: List[int] = []
    decisions_b: List[int] = []
    sample_contexts: List[Dict[str, object]] = []
    for parent_a, parent_b in context_keys:
        decision_a = 1 if parent_state_a["insertion_parent_contexts"].get(parent_a, False) else 0
        decision_b = 1 if parent_state_b["insertion_parent_contexts"].get(parent_b, False) else 0
        decisions_a.append(decision_a)
        decisions_b.append(decision_b)
        if len(sample_contexts) < sample_context_limit:
            sample_contexts.append(
                {
                    "parent_node_a": parent_a,
                    "parent_node_b": parent_b,
                    "insertion_a": bool(decision_a),
                    "insertion_b": bool(decision_b),
                    "explicit_child_count_a": parent_state_a["explicit_child_counts"].get(parent_a, 0),
                    "explicit_child_count_b": parent_state_b["explicit_child_counts"].get(parent_b, 0),
                    "implicit_child_count_a": parent_state_a["implicit_child_counts"].get(parent_a, 0),
                    "implicit_child_count_b": parent_state_b["implicit_child_counts"].get(parent_b, 0),
                    "chain_child_present_a": parent_state_a["chain_child_parent_contexts"].get(parent_a, False),
                    "chain_child_present_b": parent_state_b["chain_child_parent_contexts"].get(parent_b, False),
                }
            )

    observed_agreement, expected_agreement, kappa = _compute_binary_agreement(decisions_a, decisions_b)
    agreement_count = sum(
        1 for decision_a, decision_b in zip(decisions_a, decisions_b) if decision_a == decision_b
    )
    yes_yes_count, yes_no_count, no_yes_count, no_no_count = _binary_cell_counts(
        decisions_a,
        decisions_b,
    )
    positive_agreement, negative_agreement = _compute_positive_negative_agreement(
        yes_yes_count,
        yes_no_count,
        no_yes_count,
        no_no_count,
    )
    context_source_counts = {
        "structured_parent_contexts_a": sum(parent_state_a["structured_parent_contexts"].values()),
        "structured_parent_contexts_b": sum(parent_state_b["structured_parent_contexts"].values()),
        "insertion_parent_contexts_a": sum(parent_state_a["insertion_parent_contexts"].values()),
        "insertion_parent_contexts_b": sum(parent_state_b["insertion_parent_contexts"].values()),
        "multi_implicit_child_parents_a": sum(parent_state_a["multi_implicit_child_parents"].values()),
        "multi_implicit_child_parents_b": sum(parent_state_b["multi_implicit_child_parents"].values()),
        "chain_child_parent_contexts_a": sum(parent_state_a["chain_child_parent_contexts"].values()),
        "chain_child_parent_contexts_b": sum(parent_state_b["chain_child_parent_contexts"].values()),
        "context_union": len(context_keys),
    }

    result = ImplicitInsertionDocResult(
        ref_id=ref_id,
        annotators=(annotation_a.annotator, annotation_b.annotator),
        files=tuple(files),
        context_count=len(context_keys),
        agreement_count=agreement_count,
        yes_yes_count=yes_yes_count,
        yes_no_count=yes_no_count,
        no_yes_count=no_yes_count,
        no_no_count=no_no_count,
        observed_agreement=observed_agreement,
        expected_agreement=expected_agreement,
        kappa=kappa,
        positive_agreement=positive_agreement,
        negative_agreement=negative_agreement,
        insertion_rate_by_annotator={
            annotation_a.annotator: _safe_mean_binary(decisions_a),
            annotation_b.annotator: _safe_mean_binary(decisions_b),
        },
        usable_implicit_nodes_by_annotator={
            annotation_a.annotator: parent_state_a["usable_implicit_node_count"],
            annotation_b.annotator: parent_state_b["usable_implicit_node_count"],
        },
        excluded_implicit_nodes_by_annotator={
            annotation_a.annotator: dict(sorted(parent_state_a["excluded_counts"].items())),
            annotation_b.annotator: dict(sorted(parent_state_b["excluded_counts"].items())),
        },
        context_source_counts=context_source_counts,
        sample_contexts=sample_contexts,
    )
    return result, decisions_a, decisions_b


def _parent_centered_state(
    annotation: CaseAnnotation,
    *,
    matched_parent_ids: Set[str],
) -> Dict[str, object]:
    spans_by_id = {span.node_id: span for span in annotation.spans}
    implicit_ids = {
        span.node_id for span in annotation.spans if span.is_normalized_implicit_intermediate
    }
    outgoing, incoming = normalized_adjacency(annotation)

    structured_parent_contexts: Dict[str, bool] = {}
    insertion_parent_contexts: Dict[str, bool] = {}
    chain_child_parent_contexts: Dict[str, bool] = {}
    multi_implicit_child_parents: Dict[str, bool] = {}
    explicit_child_counts: Dict[str, int] = {}
    implicit_child_counts: Dict[str, int] = {}
    excluded_counts: Counter[str] = Counter()
    usable_implicit_node_count = 0

    seen_usable_implicit_nodes: Set[str] = set()
    for implicit_id in sorted(implicit_ids):
        parent_ids = sorted(incoming.get(implicit_id, set()))
        if not parent_ids:
            excluded_counts["no_parent"] += 1
            continue

        explicit_parent_ids = [node_id for node_id in parent_ids if node_id not in implicit_ids]
        implicit_parent_ids = [node_id for node_id in parent_ids if node_id in implicit_ids]

        if implicit_parent_ids and not explicit_parent_ids:
            excluded_counts["implicit_parent_only"] += 1
            continue

        matched_explicit_parents = [
            node_id for node_id in explicit_parent_ids if node_id in matched_parent_ids
        ]
        if not matched_explicit_parents:
            excluded_counts["unmatched_explicit_parent"] += 1
            continue

        seen_usable_implicit_nodes.add(implicit_id)
        child_ids = sorted(outgoing.get(implicit_id, set()))
        has_chain_child = any(child_id in implicit_ids for child_id in child_ids)

        for parent_id in matched_explicit_parents:
            structured_parent_contexts[parent_id] = True
            insertion_parent_contexts[parent_id] = True
            chain_child_parent_contexts[parent_id] = chain_child_parent_contexts.get(parent_id, False) or has_chain_child

    usable_implicit_node_count = len(seen_usable_implicit_nodes)

    for parent_id in matched_parent_ids:
        child_ids = sorted(outgoing.get(parent_id, set()))
        if not child_ids:
            continue
        structured_parent_contexts[parent_id] = True
        explicit_children = [child_id for child_id in child_ids if child_id not in implicit_ids]
        implicit_children = [child_id for child_id in child_ids if child_id in implicit_ids]
        explicit_child_counts[parent_id] = len(explicit_children)
        implicit_child_counts[parent_id] = len(implicit_children)
        if implicit_children:
            insertion_parent_contexts[parent_id] = True
        if len(implicit_children) > 1:
            multi_implicit_child_parents[parent_id] = True
        if any(any(grandchild in implicit_ids for grandchild in outgoing.get(child_id, set())) for child_id in implicit_children):
            chain_child_parent_contexts[parent_id] = True

    return {
        "structured_parent_contexts": structured_parent_contexts,
        "insertion_parent_contexts": insertion_parent_contexts,
        "chain_child_parent_contexts": chain_child_parent_contexts,
        "multi_implicit_child_parents": multi_implicit_child_parents,
        "explicit_child_counts": explicit_child_counts,
        "implicit_child_counts": implicit_child_counts,
        "excluded_counts": excluded_counts,
        "usable_implicit_node_count": usable_implicit_node_count,
    }


def _build_match_maps_from_pair_record(
    ref_id: DocId,
    document_record: Mapping[str, object],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    match_a_to_b: Dict[str, str] = {}
    match_b_to_a: Dict[str, str] = {}
    for match in document_record.get("matches", []):
        node_id_a = match.get("node_id_a")
        node_id_b = match.get("node_id_b")
        if not isinstance(node_id_a, str) or not isinstance(node_id_b, str):
            raise ValueError(f"Malformed pair cache match ids for ref_id {ref_id}.")
        if node_id_a in match_a_to_b and match_a_to_b[node_id_a] != node_id_b:
            raise ValueError(f"Duplicate node_id_a mapping in pair cache for ref_id {ref_id}.")
        if node_id_b in match_b_to_a and match_b_to_a[node_id_b] != node_id_a:
            raise ValueError(f"Duplicate node_id_b mapping in pair cache for ref_id {ref_id}.")
        match_a_to_b[node_id_a] = node_id_b
        match_b_to_a[node_id_b] = node_id_a
    return match_a_to_b, match_b_to_a


def _compute_binary_agreement(
    decisions_a: Sequence[int],
    decisions_b: Sequence[int],
) -> Tuple[float, float, float]:
    if len(decisions_a) != len(decisions_b):
        raise ValueError("Binary agreement inputs must have identical lengths.")
    if not decisions_a:
        nan = float("nan")
        return nan, nan, nan

    total = len(decisions_a)
    agreement_count = sum(
        1 for decision_a, decision_b in zip(decisions_a, decisions_b) if decision_a == decision_b
    )
    observed = agreement_count / total
    p_a = sum(decisions_a) / total
    p_b = sum(decisions_b) / total
    expected = (p_a * p_b) + ((1.0 - p_a) * (1.0 - p_b))
    denominator = 1.0 - expected
    if math.isclose(denominator, 0.0):
        return observed, expected, float("nan")
    return observed, expected, (observed - expected) / denominator


def _binary_cell_counts(
    decisions_a: Sequence[int],
    decisions_b: Sequence[int],
) -> Tuple[int, int, int, int]:
    if len(decisions_a) != len(decisions_b):
        raise ValueError("Binary agreement inputs must have identical lengths.")
    yes_yes = yes_no = no_yes = no_no = 0
    for decision_a, decision_b in zip(decisions_a, decisions_b):
        if decision_a == 1 and decision_b == 1:
            yes_yes += 1
        elif decision_a == 1 and decision_b == 0:
            yes_no += 1
        elif decision_a == 0 and decision_b == 1:
            no_yes += 1
        elif decision_a == 0 and decision_b == 0:
            no_no += 1
        else:
            raise ValueError(
                f"Decisions must be binary 0/1 values, got {(decision_a, decision_b)}."
            )
    return yes_yes, yes_no, no_yes, no_no


def _compute_positive_negative_agreement(
    yes_yes_count: int,
    yes_no_count: int,
    no_yes_count: int,
    no_no_count: int,
) -> Tuple[float, float]:
    positive_denominator = (2 * yes_yes_count) + yes_no_count + no_yes_count
    negative_denominator = (2 * no_no_count) + yes_no_count + no_yes_count
    positive_agreement = (
        (2 * yes_yes_count) / positive_denominator
        if positive_denominator > 0
        else float("nan")
    )
    negative_agreement = (
        (2 * no_no_count) / negative_denominator
        if negative_denominator > 0
        else float("nan")
    )
    return positive_agreement, negative_agreement


def _safe_mean_binary(decisions: Sequence[int]) -> float:
    if not decisions:
        return float("nan")
    return sum(decisions) / len(decisions)
