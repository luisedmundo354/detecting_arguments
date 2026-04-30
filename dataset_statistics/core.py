"""Public statistics and sentence-dataset builders."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .graph_metrics import (
    summarize_annotation_graph,
    summarize_lengths,
    summarize_lengths_by_label,
)
from .modules import CaseAnnotation, CaseStatistics, DatasetCase, DatasetStatistics, SentenceRecord
from .parser import build_dataset_cases, collect_dataset_metadata, load_case_annotations
from .sentence_utils import count_words, overlap_length, split_sentences_with_offsets

LABEL_UNLABELED = "Unlabeled"
DEFAULT_LABEL_ORDER = (
    "Background Facts",
    "Procedural History",
    "Rule",
    "Analysis",
    "Conclusion",
    LABEL_UNLABELED,
)


def compute_dataset_statistics(
    annotation_dir: Path | str,
    *,
    dataset_name: Optional[str] = None,
    view: str = "export",
    save_txt: bool = False,
    output_path: Optional[Path | str] = None,
) -> DatasetStatistics:
    """Compute descriptive statistics for an annotation directory."""

    annotations = load_case_annotations(annotation_dir)
    dataset_cases = build_dataset_cases(annotation_dir, view=view)
    source_dir = str(Path(annotation_dir))
    name = dataset_name or Path(annotation_dir).name

    case_statistics = tuple(_compute_case_statistics(case) for case in dataset_cases)
    metadata = collect_dataset_metadata(annotations)
    annotator_counts = Counter(annotation.annotator for annotation in annotations)

    explicit_label_counts = _sum_case_dicts(
        [case_stats.explicit_span_counts_by_label for case_stats in case_statistics]
    )
    all_node_label_counts = _sum_case_dicts(
        [case_stats.all_node_counts_by_label for case_stats in case_statistics]
    )
    sentence_label_counts = _sum_case_dicts(
        [case_stats.sentence_counts_by_label for case_stats in case_statistics]
    )

    year_distribution = Counter(
        str(case_stats.year)
        for case_stats in case_statistics
        if case_stats.year is not None
    )
    subtype_case_counts = Counter()
    for case_stats in case_statistics:
        for subtype in case_stats.subtype_368:
            subtype_case_counts[subtype] += 1

    summary_metrics = _build_dataset_summary(
        case_statistics,
        explicit_label_counts=explicit_label_counts,
        subtype_case_counts=subtype_case_counts,
    )

    notes: List[str] = [
        "Case years and Section 368 subtype counts are heuristic and based on explicit text matches.",
        "Disconnected spans are label nodes with no incident relation edge.",
        "Implicit insertions are null-offset nodes normalized from 'Implicit Intermediate Conclusion [...]' plus the single stray 'Intermediate' placeholder.",
    ]
    if view == "case":
        notes.append(
            "Case-collapsed statistics average annotation-derived metrics across exports that share the same ref_id."
        )

    stats = DatasetStatistics(
        dataset_name=name,
        source_dir=source_dir,
        view_name=view,
        case_count=len(dataset_cases),
        export_count=metadata["export_count"],
        unique_ref_id_count=metadata["unique_ref_id_count"],
        double_annotated_case_count=metadata["double_annotated_case_count"],
        annotator_counts=dict(sorted(annotator_counts.items())),
        label_distribution=_format_distribution(explicit_label_counts),
        all_node_label_distribution=_format_distribution(all_node_label_counts),
        sentence_label_distribution=_format_distribution(sentence_label_counts),
        year_distribution=dict(sorted(year_distribution.items())),
        subtype_368_distribution=_format_distribution(
            subtype_case_counts, denominator=len(case_statistics)
        ),
        summary_metrics=summary_metrics,
        case_statistics=case_statistics,
        notes=tuple(notes),
    )

    if save_txt:
        from .reporting import save_text_report

        save_text_report(stats, output_path=output_path)

    return stats


def build_sentence_dataset(
    annotation_dir: Path | str,
    *,
    dataset_name: Optional[str] = None,
    view: str = "export",
    save_jsonl: bool = False,
    output_path: Optional[Path | str] = None,
) -> List[SentenceRecord]:
    """Build a sentence-level dataset using the corpus sentence assignment rules."""

    dataset_cases = build_dataset_cases(annotation_dir, view=view)
    records: List[SentenceRecord] = []
    for case in dataset_cases:
        if view == "export":
            records.extend(_build_export_sentence_records(case))
        elif view == "case":
            records.extend(_build_case_sentence_records(case))
        else:
            raise ValueError("view must be either 'export' or 'case'")

    if save_jsonl:
        name = dataset_name or Path(annotation_dir).name
        path = Path(output_path) if output_path else _default_sentence_output_path(name, view)
        _write_jsonl([record.to_dict() for record in records], path)

    return records


def _compute_case_statistics(dataset_case: DatasetCase) -> CaseStatistics:
    annotations = list(dataset_case.annotations)
    base_annotation = annotations[0]
    base_sentence_projection = _project_annotation_to_sentences(base_annotation)

    annotation_stats = [
        _compute_single_annotation_statistics(annotation)
        for annotation in annotations
    ]

    if len(annotation_stats) == 1:
        stats = annotation_stats[0]
        return CaseStatistics(
            case_key=dataset_case.case_key,
            ref_id=dataset_case.ref_id,
            source_files=dataset_case.source_files,
            annotators=dataset_case.annotators,
            annotation_count=1,
            year=base_annotation.year,
            subtype_368=base_annotation.subtype_368,
            case_word_count=count_words(base_annotation.case_text),
            case_sentence_count=len(base_sentence_projection["sentence_spans"]),
            explicit_span_count=float(stats["explicit_span_count"]),
            implicit_insertion_count=float(stats["implicit_count"]),
            node_count=float(stats["node_count"]),
            edge_count=float(stats["edge_count"]),
            argument_tree_count=float(stats["tree_count"]),
            average_depth=stats["average_depth"],
            max_depth=float(stats["max_depth"]),
            branching_factor=float(stats["branching_factor"]),
            disconnected_span_count=float(stats["disconnected_count"]),
            disconnected_span_percentage=_safe_percentage(
                stats["disconnected_count"], stats["node_count"]
            ),
            implicit_insertion_percentage=_safe_percentage(
                stats["implicit_count"], stats["node_count"]
            ),
            explicit_span_counts_by_label=_normalize_mapping(
                stats["explicit_counts_by_label"]
            ),
            all_node_counts_by_label=_normalize_mapping(
                stats["all_node_counts_by_label"]
            ),
            sentence_counts_by_label=_normalize_mapping(
                stats["sentence_counts_by_label"]
            ),
            label_word_counts=_normalize_mapping(stats["label_word_counts"]),
            span_length_summary=stats["span_length_summary"],
            span_length_by_label=stats["span_length_by_label"],
            imbalance=stats["imbalance"],
        )

    averaged = _average_annotation_statistics(annotation_stats)
    subtype_union = sorted(
        {subtype for annotation in annotations for subtype in annotation.subtype_368}
    )
    years = [annotation.year for annotation in annotations if annotation.year is not None]

    return CaseStatistics(
        case_key=dataset_case.case_key,
        ref_id=dataset_case.ref_id,
        source_files=dataset_case.source_files,
        annotators=dataset_case.annotators,
        annotation_count=len(annotations),
        year=years[0] if years else None,
        subtype_368=tuple(subtype_union),
        case_word_count=count_words(base_annotation.case_text),
        case_sentence_count=len(base_sentence_projection["sentence_spans"]),
        explicit_span_count=averaged["explicit_span_count"],
        implicit_insertion_count=averaged["implicit_count"],
        node_count=averaged["node_count"],
        edge_count=averaged["edge_count"],
        argument_tree_count=averaged["tree_count"],
        average_depth=averaged["average_depth"],
        max_depth=averaged["max_depth"],
        branching_factor=averaged["branching_factor"],
        disconnected_span_count=averaged["disconnected_count"],
        disconnected_span_percentage=_safe_percentage(
            averaged["disconnected_count"], averaged["node_count"]
        ),
        implicit_insertion_percentage=_safe_percentage(
            averaged["implicit_count"], averaged["node_count"]
        ),
        explicit_span_counts_by_label=_normalize_mapping(
            averaged["explicit_counts_by_label"]
        ),
        all_node_counts_by_label=_normalize_mapping(
            averaged["all_node_counts_by_label"]
        ),
        sentence_counts_by_label=_normalize_mapping(
            averaged["sentence_counts_by_label"]
        ),
        label_word_counts=_normalize_mapping(averaged["label_word_counts"]),
        span_length_summary=averaged["span_length_summary"],
        span_length_by_label=averaged["span_length_by_label"],
        imbalance=averaged["imbalance"],
    )


def _compute_single_annotation_statistics(annotation: CaseAnnotation) -> Dict[str, Any]:
    graph_summary = summarize_annotation_graph(annotation)
    sentence_projection = _project_annotation_to_sentences(annotation)
    span_length_summary = _build_span_length_summary(graph_summary)
    span_length_by_label = _build_span_length_by_label(graph_summary)
    imbalance = _compute_imbalance(graph_summary["explicit_counts_by_label"])

    return {
        "explicit_span_count": float(graph_summary["explicit_span_count"]),
        "implicit_count": float(graph_summary["implicit_count"]),
        "node_count": float(graph_summary["node_count"]),
        "edge_count": float(graph_summary["edge_count"]),
        "tree_count": float(graph_summary["tree_count"]),
        "average_depth": graph_summary["average_depth"],
        "max_depth": float(graph_summary["max_depth"]),
        "branching_factor": float(graph_summary["branching_factor"]),
        "disconnected_count": float(graph_summary["disconnected_count"]),
        "explicit_counts_by_label": graph_summary["explicit_counts_by_label"],
        "all_node_counts_by_label": graph_summary["all_node_counts_by_label"],
        "sentence_counts_by_label": sentence_projection["sentence_counts_by_label"],
        "label_word_counts": graph_summary["word_counts_by_label"],
        "span_length_summary": span_length_summary,
        "span_length_by_label": span_length_by_label,
        "imbalance": imbalance,
    }


def _project_annotation_to_sentences(annotation: CaseAnnotation) -> Dict[str, Any]:
    sentence_spans = split_sentences_with_offsets(annotation.case_text)
    explicit_spans = [
        span
        for span in annotation.spans
        if not span.is_implicit and span.start is not None and span.end is not None
    ]
    explicit_spans.sort(
        key=lambda span: (int(span.start or 10**18), int(span.end or 10**18), span.node_id)
    )

    sentence_records: List[Dict[str, Any]] = []
    touched_sentence_ids_by_label: Dict[str, set[int]] = defaultdict(set)

    for index, sentence_span in enumerate(sentence_spans):
        best_key: Optional[Tuple[int, int, int, str]] = None
        best_span = None
        for span in explicit_spans:
            overlap = overlap_length(
                sentence_span.start, sentence_span.end, int(span.start), int(span.end)
            )
            if overlap <= 0:
                continue
            touched_sentence_ids_by_label[span.label].add(index)
            candidate_key = (-overlap, int(span.start), int(span.end), span.node_id)
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_span = span

        label = best_span.label if best_span is not None else LABEL_UNLABELED
        source_node_id = best_span.node_id if best_span is not None else None
        sentence_records.append(
            {
                "sentence_index": index,
                "label": label,
                "source_node_id": source_node_id,
                "text": sentence_span.text,
                "start": sentence_span.start,
                "end": sentence_span.end,
            }
        )

    sentence_counts_by_label = {
        label: len(indices)
        for label, indices in touched_sentence_ids_by_label.items()
    }
    return {
        "sentence_spans": sentence_spans,
        "sentence_records": sentence_records,
        "sentence_counts_by_label": sentence_counts_by_label,
    }


def _build_export_sentence_records(dataset_case: DatasetCase) -> List[SentenceRecord]:
    annotation = dataset_case.annotations[0]
    projection = _project_annotation_to_sentences(annotation)
    records: List[SentenceRecord] = []
    doc_id = annotation.export_id
    for projected in projection["sentence_records"]:
        passage_id = f"{doc_id}::SENT_{projected['sentence_index']:05d}"
        records.append(
            SentenceRecord(
                passage_id=passage_id,
                doc_id=doc_id,
                ref_id=annotation.ref_id,
                view_name="export",
                source_files=(annotation.source_file,),
                label=projected["label"],
                text=projected["text"],
                start=int(projected["start"]),
                end=int(projected["end"]),
                source_node_id=projected["source_node_id"],
                is_implicit=False,
                order=int(projected["sentence_index"]),
                annotator_labels={annotation.source_file: projected["label"]},
                label_votes={projected["label"]: 1},
            )
        )
    return records


def _build_case_sentence_records(dataset_case: DatasetCase) -> List[SentenceRecord]:
    projections = [
        _project_annotation_to_sentences(annotation)
        for annotation in dataset_case.annotations
    ]
    base_projection = projections[0]
    source_files = dataset_case.source_files
    doc_id = dataset_case.case_key

    records: List[SentenceRecord] = []
    for index, sentence_span in enumerate(base_projection["sentence_spans"]):
        label_votes = Counter()
        annotator_labels: Dict[str, str] = {}
        chosen_node_id: Optional[str] = None
        projection_by_label: Dict[str, Optional[str]] = {}

        for annotation, projection in zip(dataset_case.annotations, projections):
            sentence_info = projection["sentence_records"][index]
            label = str(sentence_info["label"])
            label_votes[label] += 1
            annotator_labels[annotation.source_file] = label
            projection_by_label.setdefault(label, sentence_info["source_node_id"])

        chosen_label = _choose_majority_label(label_votes)
        chosen_node_id = projection_by_label.get(chosen_label)
        passage_id = f"{doc_id}::SENT_{index:05d}"
        records.append(
            SentenceRecord(
                passage_id=passage_id,
                doc_id=doc_id,
                ref_id=dataset_case.ref_id,
                view_name="case",
                source_files=source_files,
                label=chosen_label,
                text=sentence_span.text,
                start=int(sentence_span.start),
                end=int(sentence_span.end),
                source_node_id=chosen_node_id,
                is_implicit=False,
                order=int(index),
                annotator_labels=annotator_labels,
                label_votes=dict(sorted(label_votes.items())),
            )
        )
    return records


def _average_annotation_statistics(stats_by_annotation: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "explicit_span_count": _mean([item["explicit_span_count"] for item in stats_by_annotation]),
        "implicit_count": _mean([item["implicit_count"] for item in stats_by_annotation]),
        "node_count": _mean([item["node_count"] for item in stats_by_annotation]),
        "edge_count": _mean([item["edge_count"] for item in stats_by_annotation]),
        "tree_count": _mean([item["tree_count"] for item in stats_by_annotation]),
        "average_depth": _mean_optional([item["average_depth"] for item in stats_by_annotation]),
        "max_depth": _mean([item["max_depth"] for item in stats_by_annotation]),
        "branching_factor": _mean([item["branching_factor"] for item in stats_by_annotation]),
        "disconnected_count": _mean([item["disconnected_count"] for item in stats_by_annotation]),
        "explicit_counts_by_label": _mean_mapping(
            [item["explicit_counts_by_label"] for item in stats_by_annotation]
        ),
        "all_node_counts_by_label": _mean_mapping(
            [item["all_node_counts_by_label"] for item in stats_by_annotation]
        ),
        "sentence_counts_by_label": _mean_mapping(
            [item["sentence_counts_by_label"] for item in stats_by_annotation]
        ),
        "label_word_counts": _mean_mapping(
            [item["label_word_counts"] for item in stats_by_annotation]
        ),
        "span_length_summary": _mean_nested_mapping(
            [item["span_length_summary"] for item in stats_by_annotation]
        ),
        "span_length_by_label": _mean_label_nested_mapping(
            [item["span_length_by_label"] for item in stats_by_annotation]
        ),
        "imbalance": _mean_nested_mapping(
            [item["imbalance"] for item in stats_by_annotation]
        ),
    }


def _build_dataset_summary(
    case_statistics: Sequence[CaseStatistics],
    *,
    explicit_label_counts: Mapping[str, float],
    subtype_case_counts: Mapping[str, int],
) -> Dict[str, Any]:
    case_word_counts = [case.case_word_count for case in case_statistics]
    case_sentence_counts = [case.case_sentence_count for case in case_statistics]
    span_counts = [case.explicit_span_count for case in case_statistics]
    node_counts = [case.node_count for case in case_statistics]
    edge_counts = [case.edge_count for case in case_statistics]
    tree_counts = [case.argument_tree_count for case in case_statistics]
    implicit_counts = [case.implicit_insertion_count for case in case_statistics]
    disconnected_counts = [case.disconnected_span_count for case in case_statistics]
    average_depths = [case.average_depth for case in case_statistics if case.average_depth is not None]
    max_depths = [case.max_depth for case in case_statistics]
    branching_factors = [case.branching_factor for case in case_statistics]

    label_imbalance = _compute_imbalance(explicit_label_counts)
    cases_with_368 = sum(1 for case in case_statistics if case.subtype_368)

    return {
        "total_case_words": _normalize_number(sum(case_word_counts)),
        "average_case_words": _normalize_number(_mean(case_word_counts)),
        "median_case_words": _normalize_number(statistics.median(case_word_counts)),
        "total_case_sentences": _normalize_number(sum(case_sentence_counts)),
        "average_case_sentences": _normalize_number(_mean(case_sentence_counts)),
        "median_case_sentences": _normalize_number(statistics.median(case_sentence_counts)),
        "total_explicit_spans": _normalize_number(sum(span_counts)),
        "average_explicit_spans_per_case": _normalize_number(_mean(span_counts)),
        "median_explicit_spans_per_case": _normalize_number(statistics.median(span_counts)),
        "total_nodes": _normalize_number(sum(node_counts)),
        "total_edges": _normalize_number(sum(edge_counts)),
        "total_argument_trees": _normalize_number(sum(tree_counts)),
        "total_implicit_insertions": _normalize_number(sum(implicit_counts)),
        "total_disconnected_spans": _normalize_number(sum(disconnected_counts)),
        "average_depth": _normalize_number(_mean_optional(average_depths)),
        "max_depth_observed": _normalize_number(max(max_depths) if max_depths else 0),
        "average_branching_factor": _normalize_number(_mean(branching_factors)),
        "cases_with_explicit_368_mentions": cases_with_368,
        "cases_with_explicit_368_percentage": _normalize_number(
            _safe_percentage(cases_with_368, len(case_statistics))
        ),
        "distinct_368_subtypes": sorted(subtype_case_counts),
        "label_imbalance": label_imbalance,
    }


def _build_span_length_summary(graph_summary: Dict[str, Any]) -> Dict[str, Optional[float]]:
    char_summary = summarize_lengths(graph_summary["span_char_lengths"])
    word_summary = summarize_lengths(graph_summary["span_word_lengths"])
    return {
        "char_average": char_summary["average"],
        "char_median": char_summary["median"],
        "word_average": word_summary["average"],
        "word_median": word_summary["median"],
    }


def _build_span_length_by_label(graph_summary: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    char_by_label = summarize_lengths_by_label(graph_summary["span_char_lengths_by_label"])
    word_by_label = summarize_lengths_by_label(graph_summary["span_word_lengths_by_label"])
    labels = _sort_labels(set(char_by_label) | set(word_by_label))
    summary: Dict[str, Dict[str, Optional[float]]] = {}
    for label in labels:
        char_summary = char_by_label.get(label, {"average": None, "median": None})
        word_summary = word_by_label.get(label, {"average": None, "median": None})
        summary[label] = {
            "char_average": char_summary["average"],
            "char_median": char_summary["median"],
            "word_average": word_summary["average"],
            "word_median": word_summary["median"],
        }
    return summary


def _compute_imbalance(counts: Mapping[str, float]) -> Dict[str, Optional[float]]:
    positive = [float(value) for value in counts.values() if float(value) > 0.0]
    total = sum(positive)
    if total <= 0:
        return {
            "majority_minority_ratio": None,
            "normalized_entropy": None,
        }

    majority_minority_ratio = None
    if len(positive) > 1:
        majority_minority_ratio = max(positive) / min(positive)

    if len(positive) <= 1:
        normalized_entropy = 0.0
    else:
        entropy = -sum((value / total) * math.log(value / total) for value in positive)
        normalized_entropy = entropy / math.log(len(positive))

    return {
        "majority_minority_ratio": majority_minority_ratio,
        "normalized_entropy": normalized_entropy,
    }


def _format_distribution(
    counts: Mapping[str, float],
    *,
    denominator: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    total = float(denominator) if denominator is not None else float(sum(counts.values()))
    labels = _sort_labels(counts)
    distribution: Dict[str, Dict[str, float]] = {}
    for label in labels:
        value = float(counts[label])
        distribution[label] = {
            "count": _normalize_number(value),
            "percentage": _normalize_number(_safe_percentage(value, total)),
        }
    return distribution


def _sum_case_dicts(dicts: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    for mapping in dicts:
        for key, value in mapping.items():
            totals[key] += float(value)
    return dict(totals)


def _mean_mapping(dicts: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    totals = _sum_case_dicts(dicts)
    divisor = len(dicts)
    return {key: value / divisor for key, value in totals.items()}


def _mean_nested_mapping(dicts: Sequence[Mapping[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    keys = set().union(*(mapping.keys() for mapping in dicts))
    return {
        key: _mean_optional([mapping.get(key) for mapping in dicts])
        for key in sorted(keys)
    }


def _mean_label_nested_mapping(
    dicts: Sequence[Mapping[str, Mapping[str, Optional[float]]]]
) -> Dict[str, Dict[str, Optional[float]]]:
    labels = set().union(*(mapping.keys() for mapping in dicts))
    output: Dict[str, Dict[str, Optional[float]]] = {}
    for label in _sort_labels(labels):
        inner_mappings = [mapping.get(label, {}) for mapping in dicts]
        output[label] = _mean_nested_mapping(inner_mappings)
    return output


def _mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    return float(statistics.mean(filtered)) if filtered else None


def _safe_percentage(value: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return (float(value) / float(total)) * 100.0


def _normalize_number(value: Optional[float]) -> Optional[float | int]:
    if value is None:
        return None
    rounded = round(float(value), 6)
    if abs(rounded - round(rounded)) < 1e-9:
        return int(round(rounded))
    return rounded


def _normalize_mapping(mapping: Mapping[str, float]) -> Dict[str, float]:
    return {label: _normalize_number(value) for label, value in sorted(mapping.items())}


def _sort_labels(labels: Iterable[str]) -> List[str]:
    labels = list(labels)
    order = {label: index for index, label in enumerate(DEFAULT_LABEL_ORDER)}
    return sorted(labels, key=lambda label: (order.get(label, len(order)), label))


def _choose_majority_label(label_votes: Counter[str]) -> str:
    ranked = sorted(
        label_votes.items(),
        key=lambda item: (
            -item[1],
            item[0] == LABEL_UNLABELED,
            DEFAULT_LABEL_ORDER.index(item[0]) if item[0] in DEFAULT_LABEL_ORDER else len(DEFAULT_LABEL_ORDER),
            item[0],
        ),
    )
    return ranked[0][0] if ranked else LABEL_UNLABELED


def _default_sentence_output_path(dataset_name: str, view: str) -> Path:
    safe_name = dataset_name.lower().replace(" ", "_")
    return Path(__file__).resolve().parent / f"{safe_name}_{view}_sentences.jsonl"


def _write_jsonl(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
