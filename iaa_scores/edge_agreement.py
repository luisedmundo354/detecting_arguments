"""Direct-edge agreement over aligned explicit spans."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from dataset_statistics.modules import CaseAnnotation

from .alignment import align_explicit_spans
from .annotation_graphs import load_annotation_graphs
from .cache_utils import compute_dataset_fingerprint
from .models import DocId, EdgeAgreementDocResult, EdgeAgreementScores
from .relation_utils import normalized_edge_set


def evaluate_edge_agreement(
    root,
    *,
    min_annotators: int = 2,
    metric: str = "yujianbo",
    min_sim: float = 0.1,
) -> EdgeAgreementScores:
    """Compute direct-edge agreement for the repaired overlap set."""

    graphs_by_doc = load_annotation_graphs(
        root,
        min_annotators=min_annotators,
        require_repaired_implicit_offsets=True,
    )

    per_doc: Dict[DocId, EdgeAgreementDocResult] = {}
    overall_decisions_a: List[int] = []
    overall_decisions_b: List[int] = []
    canonical_annotators: Tuple[str, str] | None = None

    for ref_id, records_by_annotator in sorted(graphs_by_doc.items()):
        annotators = tuple(sorted(records_by_annotator))
        if len(annotators) != 2:
            raise ValueError(
                f"Direct-edge agreement requires exactly two annotators per document; "
                f"ref_id {ref_id} has {len(annotators)} annotators."
            )
        if canonical_annotators is None:
            canonical_annotators = annotators
        elif annotators != canonical_annotators:
            raise ValueError(
                "Direct-edge agreement requires a consistent annotator pair across documents; "
                f"expected {canonical_annotators}, found {annotators} for ref_id {ref_id}."
            )

        record_a = records_by_annotator[annotators[0]]
        record_b = records_by_annotator[annotators[1]]
        doc_result, decisions_a, decisions_b = compute_edge_agreement_for_document(
            ref_id,
            record_a.annotation,
            record_b.annotation,
            files=(record_a.path, record_b.path),
            metric=metric,
            min_sim=min_sim,
        )
        per_doc[ref_id] = doc_result
        overall_decisions_a.extend(decisions_a)
        overall_decisions_b.extend(decisions_b)

    overall_context_count = len(overall_decisions_a)
    overall_agreement_count = sum(
        1 for decision_a, decision_b in zip(overall_decisions_a, overall_decisions_b) if decision_a == decision_b
    )
    overall_observed, overall_expected, overall_kappa = _compute_binary_agreement(
        overall_decisions_a,
        overall_decisions_b,
    )

    positive_rate_by_annotator = {}
    if canonical_annotators is not None:
        positive_rate_by_annotator = {
            canonical_annotators[0]: _safe_mean_binary(overall_decisions_a),
            canonical_annotators[1]: _safe_mean_binary(overall_decisions_b),
        }

    return EdgeAgreementScores(
        per_doc=per_doc,
        overall_context_count=overall_context_count,
        overall_agreement_count=overall_agreement_count,
        overall_observed_agreement=overall_observed,
        overall_expected_agreement=overall_expected,
        overall_kappa=overall_kappa,
        overall_positive_rate_by_annotator=positive_rate_by_annotator,
    )


def evaluate_edge_agreement_from_pair_cache(
    root: Path,
    pair_cache: Mapping[str, object],
    *,
    min_annotators: int = 2,
) -> EdgeAgreementScores:
    """Compute direct-edge agreement from a persisted pair cache."""

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
    per_doc: Dict[DocId, EdgeAgreementDocResult] = {}
    overall_decisions_a: List[int] = []
    overall_decisions_b: List[int] = []
    canonical_annotators: Tuple[str, str] | None = None

    document_records = {
        document_record["ref_id"]: document_record
        for document_record in pair_cache.get("documents", [])
    }
    if set(document_records) != set(graphs_by_doc):
        raise ValueError(
            "Pair cache documents do not match the repaired annotation graph documents."
        )

    for ref_id, records_by_annotator in sorted(graphs_by_doc.items()):
        annotators = tuple(sorted(records_by_annotator))
        if len(annotators) != 2:
            raise ValueError(
                f"Direct-edge agreement requires exactly two annotators per document; "
                f"ref_id {ref_id} has {len(annotators)} annotators."
            )
        if canonical_annotators is None:
            canonical_annotators = annotators
        elif annotators != canonical_annotators:
            raise ValueError(
                "Direct-edge agreement requires a consistent annotator pair across documents; "
                f"expected {canonical_annotators}, found {annotators} for ref_id {ref_id}."
            )

        record_a = records_by_annotator[annotators[0]]
        record_b = records_by_annotator[annotators[1]]
        doc_result, decisions_a, decisions_b = compute_edge_agreement_for_document_from_pairs(
            ref_id,
            record_a.annotation,
            record_b.annotation,
            document_records[ref_id],
            files=(record_a.path, record_b.path),
        )
        per_doc[ref_id] = doc_result
        overall_decisions_a.extend(decisions_a)
        overall_decisions_b.extend(decisions_b)

    overall_context_count = len(overall_decisions_a)
    overall_agreement_count = sum(
        1 for decision_a, decision_b in zip(overall_decisions_a, overall_decisions_b) if decision_a == decision_b
    )
    overall_observed, overall_expected, overall_kappa = _compute_binary_agreement(
        overall_decisions_a,
        overall_decisions_b,
    )
    positive_rate_by_annotator = {}
    if canonical_annotators is not None:
        positive_rate_by_annotator = {
            canonical_annotators[0]: _safe_mean_binary(overall_decisions_a),
            canonical_annotators[1]: _safe_mean_binary(overall_decisions_b),
        }
    return EdgeAgreementScores(
        per_doc=per_doc,
        overall_context_count=overall_context_count,
        overall_agreement_count=overall_agreement_count,
        overall_observed_agreement=overall_observed,
        overall_expected_agreement=overall_expected,
        overall_kappa=overall_kappa,
        overall_positive_rate_by_annotator=positive_rate_by_annotator,
    )


def compute_edge_agreement_for_document(
    ref_id: DocId,
    annotation_a: CaseAnnotation,
    annotation_b: CaseAnnotation,
    *,
    files,
    metric: str,
    min_sim: float,
) -> Tuple[EdgeAgreementDocResult, List[int], List[int]]:
    """Compute direct-edge agreement for one doubly annotated document."""

    if annotation_a.case_text != annotation_b.case_text:
        raise ValueError(
            f"Case text mismatch for ref_id {ref_id}; repaired IAA exports must share the same canonical text."
        )

    alignment = align_explicit_spans(
        annotation_a.spans,
        annotation_b.spans,
        metric=metric,
        min_sim=min_sim,
    )
    match_a_to_b = {match.span_a.node_id: match.span_b.node_id for match in alignment.matches}
    match_b_to_a = {match.span_b.node_id: match.span_a.node_id for match in alignment.matches}

    explicit_node_ids_a = {match.span_a.node_id for match in alignment.matches}
    explicit_node_ids_b = {match.span_b.node_id for match in alignment.matches}
    direct_edges_a = _normalized_explicit_edges(annotation_a, allowed_node_ids=explicit_node_ids_a)
    direct_edges_b = _normalized_explicit_edges(annotation_b, allowed_node_ids=explicit_node_ids_b)

    contexts: Dict[Tuple[str, str, str, str], Tuple[str, str, str, str]] = {}
    for source_id, target_id in direct_edges_a:
        if source_id not in match_a_to_b or target_id not in match_a_to_b:
            continue
        contexts[(source_id, match_a_to_b[source_id], target_id, match_a_to_b[target_id])] = (
            source_id,
            target_id,
            match_a_to_b[source_id],
            match_a_to_b[target_id],
        )
    for source_id, target_id in direct_edges_b:
        if source_id not in match_b_to_a or target_id not in match_b_to_a:
            continue
        contexts[(match_b_to_a[source_id], source_id, match_b_to_a[target_id], target_id)] = (
            match_b_to_a[source_id],
            match_b_to_a[target_id],
            source_id,
            target_id,
        )

    decisions_a: List[int] = []
    decisions_b: List[int] = []
    for source_a, target_a, source_b, target_b in contexts.values():
        decisions_a.append(1 if (source_a, target_a) in direct_edges_a else 0)
        decisions_b.append(1 if (source_b, target_b) in direct_edges_b else 0)

    observed_agreement, expected_agreement, kappa = _compute_binary_agreement(decisions_a, decisions_b)
    agreement_count = sum(
        1 for decision_a, decision_b in zip(decisions_a, decisions_b) if decision_a == decision_b
    )
    positive_rate_by_annotator = {
        annotation_a.annotator: _safe_mean_binary(decisions_a),
        annotation_b.annotator: _safe_mean_binary(decisions_b),
    }

    result = EdgeAgreementDocResult(
        ref_id=ref_id,
        annotators=(annotation_a.annotator, annotation_b.annotator),
        files=tuple(files),
        context_count=len(decisions_a),
        agreement_count=agreement_count,
        observed_agreement=observed_agreement,
        expected_agreement=expected_agreement,
        kappa=kappa,
        positive_rate_by_annotator=positive_rate_by_annotator,
    )
    return result, decisions_a, decisions_b


def compute_edge_agreement_for_document_from_pairs(
    ref_id: DocId,
    annotation_a: CaseAnnotation,
    annotation_b: CaseAnnotation,
    document_record: Mapping[str, object],
    *,
    files,
) -> Tuple[EdgeAgreementDocResult, List[int], List[int]]:
    """Compute direct-edge agreement for one document using a cached pair list."""

    if annotation_a.case_text != annotation_b.case_text:
        raise ValueError(
            f"Case text mismatch for ref_id {ref_id}; repaired IAA exports must share the same canonical text."
        )

    match_a_to_b = {}
    match_b_to_a = {}
    for match in document_record.get("matches", []):
        node_id_a = match.get("node_id_a")
        node_id_b = match.get("node_id_b")
        if not isinstance(node_id_a, str) or not isinstance(node_id_b, str):
            raise ValueError(f"Malformed pair cache match ids for ref_id {ref_id}.")
        match_a_to_b[node_id_a] = node_id_b
        match_b_to_a[node_id_b] = node_id_a

    explicit_node_ids_a = set(match_a_to_b)
    explicit_node_ids_b = set(match_b_to_a)
    direct_edges_a = _normalized_explicit_edges(annotation_a, allowed_node_ids=explicit_node_ids_a)
    direct_edges_b = _normalized_explicit_edges(annotation_b, allowed_node_ids=explicit_node_ids_b)

    contexts: Dict[Tuple[str, str, str, str], Tuple[str, str, str, str]] = {}
    for source_id, target_id in direct_edges_a:
        if source_id not in match_a_to_b or target_id not in match_a_to_b:
            continue
        contexts[(source_id, match_a_to_b[source_id], target_id, match_a_to_b[target_id])] = (
            source_id,
            target_id,
            match_a_to_b[source_id],
            match_a_to_b[target_id],
        )
    for source_id, target_id in direct_edges_b:
        if source_id not in match_b_to_a or target_id not in match_b_to_a:
            continue
        contexts[(match_b_to_a[source_id], source_id, match_b_to_a[target_id], target_id)] = (
            match_b_to_a[source_id],
            match_b_to_a[target_id],
            source_id,
            target_id,
        )

    decisions_a: List[int] = []
    decisions_b: List[int] = []
    for source_a, target_a, source_b, target_b in contexts.values():
        decisions_a.append(1 if (source_a, target_a) in direct_edges_a else 0)
        decisions_b.append(1 if (source_b, target_b) in direct_edges_b else 0)

    observed_agreement, expected_agreement, kappa = _compute_binary_agreement(decisions_a, decisions_b)
    agreement_count = sum(
        1 for decision_a, decision_b in zip(decisions_a, decisions_b) if decision_a == decision_b
    )
    positive_rate_by_annotator = {
        annotation_a.annotator: _safe_mean_binary(decisions_a),
        annotation_b.annotator: _safe_mean_binary(decisions_b),
    }
    result = EdgeAgreementDocResult(
        ref_id=ref_id,
        annotators=(annotation_a.annotator, annotation_b.annotator),
        files=tuple(files),
        context_count=len(decisions_a),
        agreement_count=agreement_count,
        observed_agreement=observed_agreement,
        expected_agreement=expected_agreement,
        kappa=kappa,
        positive_rate_by_annotator=positive_rate_by_annotator,
    )
    return result, decisions_a, decisions_b


def _normalized_explicit_edges(
    annotation: CaseAnnotation,
    *,
    allowed_node_ids: Sequence[str] | set[str],
) -> set[Tuple[str, str]]:
    return normalized_edge_set(annotation, allowed_node_ids=set(allowed_node_ids))


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


def _safe_mean_binary(decisions: Sequence[int]) -> float:
    if not decisions:
        return float("nan")
    return sum(decisions) / len(decisions)
