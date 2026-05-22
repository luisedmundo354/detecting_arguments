"""Diagnostics for explicit-span segmentation mismatch and relaxed edge agreement."""

from __future__ import annotations

if __package__ in {None, ""}:  # pragma: no cover - enables direct script execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "iaa_scores.diagnostics"

import argparse
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from dataset_statistics.modules import AnnotationSpan, CaseAnnotation

from .. import DEFAULT_ANNOTATION_DIR
from ..annotation_graphs import load_annotation_graphs
from ..cache_utils import cache_root, compute_dataset_fingerprint, dump_json, load_json
from ..edge_agreement import _compute_binary_agreement, evaluate_edge_agreement_from_pair_cache
from ..models import DocId
from ..relation_utils import normalized_edge_set

LabelMode = str
ContextMode = str
Vertex = tuple[str, str]


@dataclass(frozen=True)
class OverlapLink:
    """A candidate overlap link between one explicit span from each annotator."""

    node_id_a: str
    node_id_b: str
    label_a: str
    label_b: str
    overlap_chars: int
    containment: float


@dataclass(frozen=True)
class SpanGroup:
    """A disjoint relaxed span group across annotators."""

    group_id: str
    label: str
    node_ids_a: tuple[str, ...]
    node_ids_b: tuple[str, ...]

    @property
    def arity(self) -> str:
        return f"{len(self.node_ids_a)}:{len(self.node_ids_b)}"


@dataclass(frozen=True)
class GroupingResult:
    """Relaxed groups plus node lookup maps."""

    groups: tuple[SpanGroup, ...]
    node_to_group_a: dict[str, str]
    node_to_group_b: dict[str, str]
    links: tuple[OverlapLink, ...]


def build_segmentation_diagnostic_report(
    input_dir: Path,
    *,
    repo_root: Path | None = None,
    label_mode: LabelMode = "same-label",
    overlap_threshold: float = 0.5,
    context_mode: ContextMode = "edge-union",
    include_all_pairs: bool = False,
    max_examples: int = 20,
) -> dict[str, object]:
    """Build a JSON-serializable segmentation and relaxed-edge diagnostic report."""

    _validate_label_mode(label_mode)
    _validate_context_mode(context_mode)
    _validate_overlap_threshold(overlap_threshold)

    repo_root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    input_dir = input_dir.expanduser().resolve()
    dataset_fingerprint = compute_dataset_fingerprint(input_dir)
    graphs_by_doc = load_annotation_graphs(
        input_dir,
        min_annotators=2,
        require_repaired_implicit_offsets=True,
    )

    report: dict[str, object] = {
        "diagnostic": "segmentation_style_mismatch",
        "dataset_path": _repo_relative(input_dir, repo_root),
        "dataset_fingerprint": dataset_fingerprint,
        "config": {
            "label_mode": label_mode,
            "overlap_threshold": float(overlap_threshold),
            "context_mode": context_mode,
            "include_all_pairs": bool(include_all_pairs),
            "max_examples": int(max_examples),
        },
        "baseline_pair_caches": _summarize_existing_pair_caches(
            repo_root=repo_root,
            input_dir=input_dir,
            include_all_pairs=include_all_pairs,
        ),
        "relaxed_overlap_groups": _summarize_relaxed_groups(
            graphs_by_doc,
            label_mode=label_mode,
            overlap_threshold=overlap_threshold,
            context_mode=context_mode,
            include_all_pairs=include_all_pairs,
            max_examples=max_examples,
        ),
    }
    return report


def build_overlap_groups_for_document(
    ref_id: DocId,
    annotation_a: CaseAnnotation,
    annotation_b: CaseAnnotation,
    *,
    label_mode: LabelMode = "same-label",
    overlap_threshold: float = 0.5,
) -> GroupingResult:
    """Build disjoint many-to-one/many-to-many groups for one document."""

    _validate_label_mode(label_mode)
    _validate_overlap_threshold(overlap_threshold)
    spans_a = _explicit_spans(annotation_a.spans)
    spans_b = _explicit_spans(annotation_b.spans)
    by_id_a = {span.node_id: span for span in spans_a}
    by_id_b = {span.node_id: span for span in spans_b}

    adjacency: dict[Vertex, set[Vertex]] = defaultdict(set)
    links: list[OverlapLink] = []
    for span_a in spans_a:
        for span_b in spans_b:
            if label_mode == "same-label" and span_a.label != span_b.label:
                continue
            overlap = _overlap_chars(span_a, span_b)
            if overlap <= 0:
                continue
            containment = overlap / min(span_a.char_length, span_b.char_length)
            if containment < overlap_threshold:
                continue
            vertex_a = ("a", span_a.node_id)
            vertex_b = ("b", span_b.node_id)
            adjacency[vertex_a].add(vertex_b)
            adjacency[vertex_b].add(vertex_a)
            links.append(
                OverlapLink(
                    node_id_a=span_a.node_id,
                    node_id_b=span_b.node_id,
                    label_a=span_a.label,
                    label_b=span_b.label,
                    overlap_chars=overlap,
                    containment=containment,
                )
            )

    groups: list[SpanGroup] = []
    visited: set[Vertex] = set()
    for start_vertex in sorted(adjacency, key=_vertex_sort_key):
        if start_vertex in visited:
            continue
        component = _connected_component(start_vertex, adjacency, visited)
        node_ids_a = tuple(
            sorted(
                node_id
                for side, node_id in component
                if side == "a"
            )
        )
        node_ids_b = tuple(
            sorted(
                node_id
                for side, node_id in component
                if side == "b"
            )
        )
        if not node_ids_a or not node_ids_b:
            continue
        label = _group_label(
            [by_id_a[node_id] for node_id in node_ids_a],
            [by_id_b[node_id] for node_id in node_ids_b],
        )
        groups.append(
            SpanGroup(
                group_id=f"{ref_id}:G{len(groups) + 1:04d}",
                label=label,
                node_ids_a=node_ids_a,
                node_ids_b=node_ids_b,
            )
        )

    node_to_group_a: dict[str, str] = {}
    node_to_group_b: dict[str, str] = {}
    for group in groups:
        for node_id in group.node_ids_a:
            if node_id in node_to_group_a:
                raise ValueError(
                    f"Node {node_id!r} from annotator A appears in multiple relaxed groups."
                )
            node_to_group_a[node_id] = group.group_id
        for node_id in group.node_ids_b:
            if node_id in node_to_group_b:
                raise ValueError(
                    f"Node {node_id!r} from annotator B appears in multiple relaxed groups."
                )
            node_to_group_b[node_id] = group.group_id

    return GroupingResult(
        groups=tuple(groups),
        node_to_group_a=node_to_group_a,
        node_to_group_b=node_to_group_b,
        links=tuple(links),
    )


def compute_group_edge_agreement_for_document(
    ref_id: DocId,
    annotation_a: CaseAnnotation,
    annotation_b: CaseAnnotation,
    grouping: GroupingResult,
    *,
    context_mode: ContextMode = "edge-union",
) -> dict[str, object]:
    """Compute direct-edge agreement after collapsing explicit spans to groups."""

    _validate_context_mode(context_mode)
    if annotation_a.case_text != annotation_b.case_text:
        raise ValueError(
            f"Case text mismatch for ref_id {ref_id}; repaired IAA exports must share text."
        )

    grouped_edges_a, edge_loss_a = _collapse_annotation_edges(
        annotation_a,
        grouping.node_to_group_a,
    )
    grouped_edges_b, edge_loss_b = _collapse_annotation_edges(
        annotation_b,
        grouping.node_to_group_b,
    )
    group_ids = sorted(group.group_id for group in grouping.groups)
    decisions_a, decisions_b = _group_edge_decisions(
        grouped_edges_a,
        grouped_edges_b,
        group_ids,
        context_mode=context_mode,
    )
    summary = _binary_summary(decisions_a, decisions_b)
    summary["edge_loss_by_annotator"] = {
        annotation_a.annotator: edge_loss_a,
        annotation_b.annotator: edge_loss_b,
    }
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run experimental diagnostics for span segmentation style mismatch and "
            "relaxed group-level direct-edge agreement."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR,
        help="Directory with repaired IAA Label Studio JSON exports.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("same-label", "label-blind"),
        default="same-label",
        help="Whether overlap groups require matching labels.",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.5,
        help="Minimum containment overlap against the shorter span for grouping.",
    )
    parser.add_argument(
        "--context-mode",
        choices=("edge-union", "all-pairs"),
        default="edge-union",
        help="Context set used for the primary relaxed edge-agreement metric.",
    )
    parser.add_argument(
        "--include-all-pairs",
        action="store_true",
        help="Also report all ordered group/span pair contexts as a sensitivity check.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Maximum split/merge examples to include in the JSON and printed report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the full diagnostic report.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"--input-dir {input_dir} is not a directory")

    report = build_segmentation_diagnostic_report(
        input_dir,
        repo_root=repo_root,
        label_mode=args.label_mode,
        overlap_threshold=args.overlap_threshold,
        context_mode=args.context_mode,
        include_all_pairs=args.include_all_pairs,
        max_examples=args.max_examples,
    )
    _print_report(report)

    if args.output is not None:
        output_path = args.output.expanduser()
        dump_json(_strict_json_value(report), output_path)
        print(f"\nWrote diagnostic JSON: {output_path}")


def _summarize_existing_pair_caches(
    *,
    repo_root: Path,
    input_dir: Path,
    include_all_pairs: bool,
) -> dict[str, object]:
    pair_dir = cache_root(repo_root) / "pairs"
    if not pair_dir.exists():
        return {"evaluated": [], "skipped": []}

    dataset_fingerprint = compute_dataset_fingerprint(input_dir)
    evaluated: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for path in sorted(pair_dir.glob("*.json")):
        try:
            pair_cache = load_json(path)
            if pair_cache.get("dataset_fingerprint") != dataset_fingerprint:
                skipped.append({"path": _repo_relative(path, repo_root), "reason": "fingerprint_mismatch"})
                continue
            scores = evaluate_edge_agreement_from_pair_cache(input_dir, pair_cache)
            edge_union = _pair_cache_edge_summary(
                input_dir,
                pair_cache,
                context_mode="edge-union",
            )
            record = {
                "path": _repo_relative(path, repo_root),
                "backend": pair_cache.get("backend"),
                "label_restricted": pair_cache.get("label_restricted"),
                "config": pair_cache.get("config"),
                "edge_union": edge_union,
                "existing_score_object": {
                    "contexts": scores.overall_context_count,
                    "observed": scores.overall_observed_agreement,
                    "expected": scores.overall_expected_agreement,
                    "kappa": scores.overall_kappa,
                    "positive_rate_by_annotator": scores.overall_positive_rate_by_annotator,
                },
            }
            if include_all_pairs:
                record["all_pairs"] = _pair_cache_edge_summary(
                    input_dir,
                    pair_cache,
                    context_mode="all-pairs",
                )
            evaluated.append(record)
        except Exception as exc:  # pragma: no cover - defensive reporting for ad hoc caches
            skipped.append({"path": _repo_relative(path, repo_root), "reason": str(exc)})

    return {"evaluated": evaluated, "skipped": skipped}


def _pair_cache_edge_summary(
    input_dir: Path,
    pair_cache: Mapping[str, object],
    *,
    context_mode: ContextMode,
) -> dict[str, object]:
    _validate_context_mode(context_mode)
    dataset_fingerprint = compute_dataset_fingerprint(input_dir)
    if pair_cache.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("Pair cache fingerprint does not match the current dataset.")

    graphs_by_doc = load_annotation_graphs(
        input_dir,
        min_annotators=2,
        require_repaired_implicit_offsets=True,
    )
    document_records = {
        document_record["ref_id"]: document_record
        for document_record in pair_cache.get("documents", [])
    }
    if set(document_records) != set(graphs_by_doc):
        raise ValueError("Pair cache documents do not match the annotation graph documents.")

    overall_a: list[int] = []
    overall_b: list[int] = []
    per_doc: dict[str, object] = {}
    annotator_pair: tuple[str, str] | None = None
    for ref_id, records_by_annotator in sorted(graphs_by_doc.items()):
        annotators = tuple(sorted(records_by_annotator))
        if len(annotators) != 2:
            raise ValueError(
                f"Edge diagnostics require exactly two annotators; ref_id {ref_id} has {len(annotators)}."
            )
        annotator_pair = _update_annotator_pair(
            annotator_pair,
            annotators,
            ref_id=ref_id,
            context="edge diagnostics",
        )
        record_a = records_by_annotator[annotators[0]]
        record_b = records_by_annotator[annotators[1]]
        match_a_to_b, match_b_to_a = _match_maps_from_pair_record(
            ref_id,
            document_records[ref_id],
        )
        decisions_a, decisions_b = _pair_cache_edge_decisions(
            record_a.annotation,
            record_b.annotation,
            match_a_to_b,
            match_b_to_a,
            context_mode=context_mode,
        )
        overall_a.extend(decisions_a)
        overall_b.extend(decisions_b)
        doc_summary = _binary_summary(decisions_a, decisions_b)
        doc_summary.pop("_decisions_a")
        doc_summary.pop("_decisions_b")
        per_doc[str(ref_id)] = doc_summary

    summary = _binary_summary(overall_a, overall_b)
    summary.pop("_decisions_a")
    summary.pop("_decisions_b")
    if annotator_pair is not None:
        summary["positive_rate_by_annotator"] = {
            annotator_pair[0]: summary["positive_rate_a"],
            annotator_pair[1]: summary["positive_rate_b"],
        }
    summary["per_doc"] = per_doc
    return summary


def _summarize_relaxed_groups(
    graphs_by_doc,
    *,
    label_mode: LabelMode,
    overlap_threshold: float,
    context_mode: ContextMode,
    include_all_pairs: bool,
    max_examples: int,
) -> dict[str, object]:
    overall_group_counts = Counter()
    overall_group_counts_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    overall_span_coverage: dict[str, Counter[str]] = defaultdict(Counter)
    overall_implicit_counts: dict[str, Counter[str]] = defaultdict(Counter)
    overall_edge_loss: dict[str, Counter[str]] = defaultdict(Counter)
    overall_decisions_a: list[int] = []
    overall_decisions_b: list[int] = []
    overall_all_pair_decisions_a: list[int] = []
    overall_all_pair_decisions_b: list[int] = []
    examples: list[dict[str, object]] = []
    per_doc: dict[str, object] = {}
    annotator_pair: tuple[str, str] | None = None

    for ref_id, records_by_annotator in sorted(graphs_by_doc.items()):
        annotators = tuple(sorted(records_by_annotator))
        if len(annotators) != 2:
            raise ValueError(
                f"Relaxed diagnostics require exactly two annotators; ref_id {ref_id} has {len(annotators)}."
            )
        annotator_pair = _update_annotator_pair(
            annotator_pair,
            annotators,
            ref_id=ref_id,
            context="relaxed diagnostics",
        )
        record_a = records_by_annotator[annotators[0]]
        record_b = records_by_annotator[annotators[1]]
        grouping = build_overlap_groups_for_document(
            ref_id,
            record_a.annotation,
            record_b.annotation,
            label_mode=label_mode,
            overlap_threshold=overlap_threshold,
        )
        group_summary = _grouping_summary(
            grouping,
            record_a.annotation,
            record_b.annotation,
        )
        edge_summary = compute_group_edge_agreement_for_document(
            ref_id,
            record_a.annotation,
            record_b.annotation,
            grouping,
            context_mode=context_mode,
        )
        overall_decisions_a.extend(edge_summary.pop("_decisions_a"))
        overall_decisions_b.extend(edge_summary.pop("_decisions_b"))

        all_pairs_summary = None
        if include_all_pairs:
            all_pairs_summary = compute_group_edge_agreement_for_document(
                ref_id,
                record_a.annotation,
                record_b.annotation,
                grouping,
                context_mode="all-pairs",
            )
            overall_all_pair_decisions_a.extend(all_pairs_summary.pop("_decisions_a"))
            overall_all_pair_decisions_b.extend(all_pairs_summary.pop("_decisions_b"))

        for arity, count in group_summary["arity_counts"].items():
            overall_group_counts[arity] += int(count)
        for label, arity_counts in group_summary["arity_counts_by_label"].items():
            overall_group_counts_by_label[label].update(
                {arity: int(count) for arity, count in arity_counts.items()}
            )
        for annotator, coverage_counts in group_summary["span_coverage_by_annotator"].items():
            overall_span_coverage[annotator].update(
                {key: int(value) for key, value in coverage_counts.items()}
            )
        for annotator, label_counts in group_summary["implicit_span_counts_by_annotator"].items():
            overall_implicit_counts[annotator].update(
                {key: int(value) for key, value in label_counts.items()}
            )
        for annotator, loss_counts in edge_summary["edge_loss_by_annotator"].items():
            overall_edge_loss[annotator].update(
                {key: int(value) for key, value in loss_counts.items()}
            )

        if len(examples) < max_examples:
            examples.extend(
                _group_examples(
                    ref_id,
                    grouping,
                    record_a.annotation,
                    record_b.annotation,
                    limit=max_examples - len(examples),
                )
            )

        doc_record: dict[str, object] = {
            "annotators": list(annotators),
            "files": [record_a.path.name, record_b.path.name],
            "groups": group_summary,
            "edge_agreement": edge_summary,
        }
        if all_pairs_summary is not None:
            doc_record["all_pairs_edge_agreement"] = all_pairs_summary
        per_doc[str(ref_id)] = doc_record

    overall_edge_summary = _binary_summary(overall_decisions_a, overall_decisions_b)
    overall_edge_summary.pop("_decisions_a")
    overall_edge_summary.pop("_decisions_b")
    if annotator_pair is not None:
        overall_edge_summary["positive_rate_by_annotator"] = {
            annotator_pair[0]: overall_edge_summary["positive_rate_a"],
            annotator_pair[1]: overall_edge_summary["positive_rate_b"],
        }
    overall_edge_summary["edge_loss_by_annotator"] = {
        annotator: dict(sorted(counts.items()))
        for annotator, counts in sorted(overall_edge_loss.items())
    }

    overall: dict[str, object] = {
        "arity_counts": dict(sorted(overall_group_counts.items())),
        "arity_counts_by_label": {
            label: dict(sorted(counts.items()))
            for label, counts in sorted(overall_group_counts_by_label.items())
        },
        "span_coverage_by_annotator": {
            annotator: dict(sorted(counts.items()))
            for annotator, counts in sorted(overall_span_coverage.items())
        },
        "implicit_span_counts_by_annotator": {
            annotator: dict(sorted(counts.items()))
            for annotator, counts in sorted(overall_implicit_counts.items())
        },
        "edge_agreement": overall_edge_summary,
    }
    if include_all_pairs:
        all_pairs_overall = _binary_summary(overall_all_pair_decisions_a, overall_all_pair_decisions_b)
        all_pairs_overall.pop("_decisions_a")
        all_pairs_overall.pop("_decisions_b")
        if annotator_pair is not None:
            all_pairs_overall["positive_rate_by_annotator"] = {
                annotator_pair[0]: all_pairs_overall["positive_rate_a"],
                annotator_pair[1]: all_pairs_overall["positive_rate_b"],
            }
        overall["all_pairs_edge_agreement"] = all_pairs_overall

    return {
        "overall": overall,
        "per_doc": per_doc,
        "examples": examples,
    }


def _grouping_summary(
    grouping: GroupingResult,
    annotation_a: CaseAnnotation,
    annotation_b: CaseAnnotation,
) -> dict[str, object]:
    arity_counts = Counter(group.arity for group in grouping.groups)
    arity_counts_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    for group in grouping.groups:
        arity_counts_by_label[group.label][group.arity] += 1

    explicit_a = _explicit_spans(annotation_a.spans)
    explicit_b = _explicit_spans(annotation_b.spans)
    grouped_a = set(grouping.node_to_group_a)
    grouped_b = set(grouping.node_to_group_b)

    return {
        "group_count": len(grouping.groups),
        "candidate_link_count": len(grouping.links),
        "arity_counts": dict(sorted(arity_counts.items())),
        "arity_counts_by_label": {
            label: dict(sorted(counts.items()))
            for label, counts in sorted(arity_counts_by_label.items())
        },
        "span_coverage_by_annotator": {
            annotation_a.annotator: _span_coverage_counts(explicit_a, grouped_a),
            annotation_b.annotator: _span_coverage_counts(explicit_b, grouped_b),
        },
        "span_counts_by_label_by_annotator": {
            annotation_a.annotator: _label_counts(explicit_a),
            annotation_b.annotator: _label_counts(explicit_b),
        },
        "grouped_span_counts_by_label_by_annotator": {
            annotation_a.annotator: _label_counts(
                span for span in explicit_a if span.node_id in grouped_a
            ),
            annotation_b.annotator: _label_counts(
                span for span in explicit_b if span.node_id in grouped_b
            ),
        },
        "ungrouped_span_counts_by_label_by_annotator": {
            annotation_a.annotator: _label_counts(
                span for span in explicit_a if span.node_id not in grouped_a
            ),
            annotation_b.annotator: _label_counts(
                span for span in explicit_b if span.node_id not in grouped_b
            ),
        },
        "implicit_span_counts_by_annotator": {
            annotation_a.annotator: _label_counts(span for span in annotation_a.spans if span.is_implicit),
            annotation_b.annotator: _label_counts(span for span in annotation_b.spans if span.is_implicit),
        },
    }


def _span_coverage_counts(
    explicit_spans: Sequence[AnnotationSpan],
    grouped_node_ids: set[str],
) -> dict[str, int]:
    return {
        "explicit_total": len(explicit_spans),
        "grouped": sum(1 for span in explicit_spans if span.node_id in grouped_node_ids),
        "ungrouped": sum(1 for span in explicit_spans if span.node_id not in grouped_node_ids),
    }


def _collapse_annotation_edges(
    annotation: CaseAnnotation,
    node_to_group: Mapping[str, str],
) -> tuple[set[tuple[str, str]], dict[str, int]]:
    explicit_ids = {span.node_id for span in annotation.spans if not span.is_implicit}
    grouped_ids = set(node_to_group)
    explicit_edges = normalized_edge_set(annotation, allowed_node_ids=explicit_ids)
    all_edges = normalized_edge_set(annotation)

    grouped_edges: set[tuple[str, str]] = set()
    internal_edge_count = 0
    lost_endpoint_count = 0
    for source_id, target_id in explicit_edges:
        source_group = node_to_group.get(source_id)
        target_group = node_to_group.get(target_id)
        if source_group is None or target_group is None:
            lost_endpoint_count += 1
            continue
        if source_group == target_group:
            internal_edge_count += 1
            continue
        grouped_edges.add((source_group, target_group))

    return grouped_edges, {
        "all_edges": len(all_edges),
        "explicit_explicit_edges": len(explicit_edges),
        "edges_touching_implicit": sum(
            1 for source_id, target_id in all_edges
            if source_id not in explicit_ids or target_id not in explicit_ids
        ),
        "grouped_explicit_edges": sum(
            1 for source_id, target_id in explicit_edges
            if source_id in grouped_ids and target_id in grouped_ids
        ),
        "collapsed_group_edges": len(grouped_edges),
        "internal_edges_within_group": internal_edge_count,
        "lost_edges_due_to_ungrouped_endpoint": lost_endpoint_count,
    }


def _group_edge_decisions(
    grouped_edges_a: set[tuple[str, str]],
    grouped_edges_b: set[tuple[str, str]],
    group_ids: Sequence[str],
    *,
    context_mode: ContextMode,
) -> tuple[list[int], list[int]]:
    if context_mode == "edge-union":
        contexts = sorted(grouped_edges_a | grouped_edges_b)
    elif context_mode == "all-pairs":
        contexts = [
            (source_group, target_group)
            for source_group in group_ids
            for target_group in group_ids
            if source_group != target_group
        ]
    else:  # pragma: no cover - guarded by argparse and validators
        raise ValueError(f"Unsupported context mode: {context_mode}")

    decisions_a = [1 if context in grouped_edges_a else 0 for context in contexts]
    decisions_b = [1 if context in grouped_edges_b else 0 for context in contexts]
    return decisions_a, decisions_b


def _pair_cache_edge_decisions(
    annotation_a: CaseAnnotation,
    annotation_b: CaseAnnotation,
    match_a_to_b: Mapping[str, str],
    match_b_to_a: Mapping[str, str],
    *,
    context_mode: ContextMode,
) -> tuple[list[int], list[int]]:
    direct_edges_a = normalized_edge_set(annotation_a, allowed_node_ids=set(match_a_to_b))
    direct_edges_b = normalized_edge_set(annotation_b, allowed_node_ids=set(match_b_to_a))

    if context_mode == "edge-union":
        contexts: dict[tuple[str, str, str, str], tuple[str, str, str, str]] = {}
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
        values = list(contexts.values())
    elif context_mode == "all-pairs":
        matched_pairs = sorted(match_a_to_b.items())
        values = [
            (source_a, target_a, source_b, target_b)
            for source_a, source_b in matched_pairs
            for target_a, target_b in matched_pairs
            if source_a != target_a and source_b != target_b
        ]
    else:  # pragma: no cover - guarded by argparse and validators
        raise ValueError(f"Unsupported context mode: {context_mode}")

    decisions_a: list[int] = []
    decisions_b: list[int] = []
    for source_a, target_a, source_b, target_b in values:
        decisions_a.append(1 if (source_a, target_a) in direct_edges_a else 0)
        decisions_b.append(1 if (source_b, target_b) in direct_edges_b else 0)
    return decisions_a, decisions_b


def _match_maps_from_pair_record(
    ref_id: DocId,
    document_record: Mapping[str, object],
) -> tuple[dict[str, str], dict[str, str]]:
    match_a_to_b: dict[str, str] = {}
    match_b_to_a: dict[str, str] = {}
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


def _binary_summary(decisions_a: Sequence[int], decisions_b: Sequence[int]) -> dict[str, object]:
    observed, expected, kappa = _compute_binary_agreement(decisions_a, decisions_b)
    yes_yes, yes_no, no_yes, no_no = _binary_cell_counts(decisions_a, decisions_b)
    total = len(decisions_a)
    agreement_count = yes_yes + no_no
    return {
        "contexts": total,
        "agreement_count": agreement_count,
        "yes_yes": yes_yes,
        "a_only": yes_no,
        "b_only": no_yes,
        "no_no": no_no,
        "observed": observed,
        "expected": expected,
        "kappa": kappa,
        "positive_rate_a": _safe_mean_binary(decisions_a),
        "positive_rate_b": _safe_mean_binary(decisions_b),
        "_decisions_a": list(decisions_a),
        "_decisions_b": list(decisions_b),
    }


def _binary_cell_counts(
    decisions_a: Sequence[int],
    decisions_b: Sequence[int],
) -> tuple[int, int, int, int]:
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
            raise ValueError(f"Decisions must be binary 0/1 values, got {(decision_a, decision_b)}.")
    return yes_yes, yes_no, no_yes, no_no


def _group_examples(
    ref_id: DocId,
    grouping: GroupingResult,
    annotation_a: CaseAnnotation,
    annotation_b: CaseAnnotation,
    *,
    limit: int,
) -> list[dict[str, object]]:
    if limit <= 0:
        return []
    spans_a = {span.node_id: span for span in annotation_a.spans}
    spans_b = {span.node_id: span for span in annotation_b.spans}
    examples: list[dict[str, object]] = []
    for group in grouping.groups:
        if group.arity == "1:1":
            continue
        examples.append(
            {
                "ref_id": ref_id,
                "group_id": group.group_id,
                "label": group.label,
                "arity": group.arity,
                "annotator_a": annotation_a.annotator,
                "annotator_b": annotation_b.annotator,
                "spans_a": [_span_example(spans_a[node_id]) for node_id in group.node_ids_a],
                "spans_b": [_span_example(spans_b[node_id]) for node_id in group.node_ids_b],
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _span_example(span: AnnotationSpan) -> dict[str, object]:
    return {
        "node_id": span.node_id,
        "label": span.label,
        "start": span.start,
        "end": span.end,
        "text": _truncate(span.text),
    }


def _print_report(report: Mapping[str, object]) -> None:
    config = report["config"]
    relaxed = report["relaxed_overlap_groups"]
    overall = relaxed["overall"]
    edge = overall["edge_agreement"]

    print("=== Segmentation Style Mismatch Diagnostic ===")
    print(f"Dataset: {report['dataset_path']}")
    print(f"Fingerprint: {report['dataset_fingerprint']}")
    print(
        "Config: "
        f"label_mode={config['label_mode']}, "
        f"overlap_threshold={config['overlap_threshold']}, "
        f"context_mode={config['context_mode']}"
    )
    if config["context_mode"] == "edge-union":
        print(
            "Note: edge-union contexts include group pairs where at least one "
            "annotator has a direct edge; use --include-all-pairs for no/no contexts."
        )

    print("\nRelaxed overlap groups:")
    print(f"  arity counts: {_compact_counts(overall['arity_counts'])}")
    print(f"  span coverage: {overall['span_coverage_by_annotator']}")

    print("\nRelaxed group-level edge agreement:")
    _print_edge_summary(edge)
    if "all_pairs_edge_agreement" in overall:
        print("\nAll ordered group-pairs sensitivity:")
        _print_edge_summary(overall["all_pairs_edge_agreement"])

    baseline = report["baseline_pair_caches"]
    evaluated = baseline["evaluated"]
    print("\nExisting pair-cache baselines:")
    if not evaluated:
        print("  no current-fingerprint pair caches found")
    for record in evaluated:
        print(f"  {record['path']}")
        _print_edge_summary(record["edge_union"], indent="    ")
        if "all_pairs" in record:
            print("    all-pairs:")
            _print_edge_summary(record["all_pairs"], indent="      ")

    examples = relaxed["examples"]
    if examples:
        print("\nSplit/merge examples:")
        for example in examples[: int(config["max_examples"])]:
            print(
                f"  ref_id={example['ref_id']} {example['label']} "
                f"{example['arity']} group={example['group_id']}"
            )


def _print_edge_summary(summary: Mapping[str, object], *, indent: str = "  ") -> None:
    print(
        indent
        + "contexts={contexts}, observed={observed}, expected={expected}, "
        "kappa={kappa}, cells yy/a_only/b_only/nn={yes_yes}/{a_only}/{b_only}/{no_no}".format(
            contexts=summary["contexts"],
            observed=_format_float(summary["observed"]),
            expected=_format_float(summary["expected"]),
            kappa=_format_float(summary["kappa"]),
            yes_yes=summary["yes_yes"],
            a_only=summary["a_only"],
            b_only=summary["b_only"],
            no_no=summary["no_no"],
        )
    )


def _compact_counts(counts: Mapping[str, object]) -> str:
    if not counts:
        return "{}"
    return "{" + ", ".join(f"{key}: {value}" for key, value in counts.items()) + "}"


def _format_float(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    return f"{number:.3f}"


def _connected_component(
    start_vertex: Vertex,
    adjacency: Mapping[Vertex, set[Vertex]],
    visited: set[Vertex],
) -> set[Vertex]:
    component: set[Vertex] = set()
    queue: deque[Vertex] = deque([start_vertex])
    visited.add(start_vertex)
    while queue:
        vertex = queue.popleft()
        component.add(vertex)
        for neighbor in sorted(adjacency.get(vertex, set()), key=_vertex_sort_key):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return component


def _group_label(spans_a: Sequence[AnnotationSpan], spans_b: Sequence[AnnotationSpan]) -> str:
    labels = sorted({span.label for span in spans_a} | {span.label for span in spans_b})
    if len(labels) == 1:
        return labels[0]
    return "mixed:" + "|".join(labels)


def _overlap_chars(span_a: AnnotationSpan, span_b: AnnotationSpan) -> int:
    if span_a.start is None or span_a.end is None or span_b.start is None or span_b.end is None:
        return 0
    return max(0, min(span_a.end, span_b.end) - max(span_a.start, span_b.start))


def _explicit_spans(spans: Iterable[AnnotationSpan]) -> list[AnnotationSpan]:
    return sorted(
        [span for span in spans if not span.is_implicit],
        key=lambda span: (
            _offset_sort_value(span.start),
            _offset_sort_value(span.end),
            span.node_id,
        ),
    )


def _label_counts(spans: Iterable[AnnotationSpan]) -> dict[str, int]:
    return dict(sorted(Counter(span.label for span in spans).items()))


def _safe_mean_binary(decisions: Sequence[int]) -> float:
    if not decisions:
        return float("nan")
    return sum(decisions) / len(decisions)


def _strict_json_value(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    return value


def _update_annotator_pair(
    current_pair: tuple[str, str] | None,
    annotators: tuple[str, str],
    *,
    ref_id: DocId,
    context: str,
) -> tuple[str, str]:
    if current_pair is None:
        return annotators
    if annotators != current_pair:
        raise ValueError(
            f"{context} require a consistent annotator pair across documents; "
            f"expected {current_pair}, found {annotators} for ref_id {ref_id}."
        )
    return current_pair


def _offset_sort_value(value: int | None) -> int:
    return 10**18 if value is None else value


def _vertex_sort_key(vertex: Vertex) -> tuple[str, str]:
    return vertex


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _truncate(text: str, *, max_chars: int = 240) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


def _validate_label_mode(label_mode: str) -> None:
    if label_mode not in {"same-label", "label-blind"}:
        raise ValueError("label_mode must be either 'same-label' or 'label-blind'.")


def _validate_overlap_threshold(overlap_threshold: float) -> None:
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be between 0 and 1.")


def _validate_context_mode(context_mode: str) -> None:
    if context_mode not in {"edge-union", "all-pairs"}:
        raise ValueError("context_mode must be either 'edge-union' or 'all-pairs'.")


if __name__ == "__main__":  # pragma: no cover
    main()
