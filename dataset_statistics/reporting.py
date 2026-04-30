"""Human-readable reporting for dataset statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .modules import CaseStatistics, DatasetStatistics


def format_statistics_report(stats: DatasetStatistics) -> str:
    lines = [
        f"Dataset: {stats.dataset_name}",
        f"View: {stats.view_name}",
        f"Source directory: {stats.source_dir}",
        f"Cases in view: {stats.case_count}",
        f"Exports in source directory: {stats.export_count}",
        f"Unique ref_id cases: {stats.unique_ref_id_count}",
        f"Double annotated cases: {stats.double_annotated_case_count}",
        f"Annotators: {_format_simple_mapping(stats.annotator_counts)}",
        "",
        "Summary metrics:",
    ]
    for key, value in stats.summary_metrics.items():
        lines.append(f"  {key}: {_format_scalar(value)}")

    lines.extend(
        [
            "",
            "Explicit span label distribution:",
        ]
    )
    lines.extend(_format_distribution_block(stats.label_distribution))

    lines.extend(
        [
            "",
            "All node label distribution:",
        ]
    )
    lines.extend(_format_distribution_block(stats.all_node_label_distribution))

    lines.extend(
        [
            "",
            "Sentence label distribution:",
        ]
    )
    lines.extend(_format_distribution_block(stats.sentence_label_distribution))

    lines.extend(
        [
            "",
            "Year distribution:",
        ]
    )
    if stats.year_distribution:
        for year, count in stats.year_distribution.items():
            lines.append(f"  {year}: {count}")
    else:
        lines.append("  none")

    lines.extend(
        [
            "",
            "Section 368 subtype distribution:",
        ]
    )
    if stats.subtype_368_distribution:
        lines.extend(_format_distribution_block(stats.subtype_368_distribution))
    else:
        lines.append("  none")

    lines.extend(
        [
            "",
            "Per-case statistics:",
        ]
    )
    for case_stats in stats.case_statistics:
        lines.extend(_format_case_block(case_stats))

    if stats.notes:
        lines.extend(
            [
                "",
                "Notes:",
            ]
        )
        for note in stats.notes:
            lines.append(f"  - {note}")

    return "\n".join(lines)


def save_text_report(
    stats: DatasetStatistics,
    *,
    output_path: Path | str | None = None,
) -> Path:
    path = Path(output_path) if output_path else _default_report_path(stats)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_statistics_report(stats), encoding="utf-8")
    return path


def _default_report_path(stats: DatasetStatistics) -> Path:
    safe_name = stats.dataset_name.lower().replace(" ", "_")
    return (
        Path(__file__).resolve().parent
        / f"{safe_name}_{stats.view_name}_statistics_report.txt"
    )


def _format_distribution_block(
    distribution: Mapping[str, Mapping[str, object]]
) -> Sequence[str]:
    if not distribution:
        return ["  none"]
    return [
        f"  {label}: count={_format_scalar(values.get('count'))}, percentage={_format_scalar(values.get('percentage'))}"
        for label, values in distribution.items()
    ]


def _format_case_block(case_stats: CaseStatistics) -> Sequence[str]:
    lines = [
        f"  Case {case_stats.case_key}: ref_id={case_stats.ref_id}, annotations={case_stats.annotation_count}, files={', '.join(case_stats.source_files)}",
        f"    annotators={', '.join(case_stats.annotators) or 'unknown'}",
        f"    year={_format_scalar(case_stats.year)}, section_368={', '.join(case_stats.subtype_368) or 'none'}",
        f"    case_words={_format_scalar(case_stats.case_word_count)}, case_sentences={_format_scalar(case_stats.case_sentence_count)}",
        f"    explicit_spans={_format_scalar(case_stats.explicit_span_count)}, implicit_insertions={_format_scalar(case_stats.implicit_insertion_count)}, nodes={_format_scalar(case_stats.node_count)}, edges={_format_scalar(case_stats.edge_count)}, trees={_format_scalar(case_stats.argument_tree_count)}",
        f"    average_depth={_format_scalar(case_stats.average_depth)}, max_depth={_format_scalar(case_stats.max_depth)}, branching_factor={_format_scalar(case_stats.branching_factor)}",
        f"    disconnected_spans={_format_scalar(case_stats.disconnected_span_count)} ({_format_scalar(case_stats.disconnected_span_percentage)}%), implicit_percentage={_format_scalar(case_stats.implicit_insertion_percentage)}%",
        f"    explicit_span_counts_by_label={_format_simple_mapping(case_stats.explicit_span_counts_by_label)}",
        f"    all_node_counts_by_label={_format_simple_mapping(case_stats.all_node_counts_by_label)}",
        f"    sentence_counts_by_label={_format_simple_mapping(case_stats.sentence_counts_by_label)}",
        f"    label_word_counts={_format_simple_mapping(case_stats.label_word_counts)}",
        f"    span_length_summary={_format_simple_mapping(case_stats.span_length_summary)}",
        f"    span_length_by_label={_format_nested_mapping(case_stats.span_length_by_label)}",
        f"    imbalance={_format_simple_mapping(case_stats.imbalance)}",
    ]
    return lines


def _format_nested_mapping(mapping: Mapping[str, Mapping[str, object]]) -> str:
    parts = []
    for key, value in mapping.items():
        parts.append(f"{key}: {{{_format_simple_mapping(value)}}}")
    return "; ".join(parts) if parts else "none"


def _format_simple_mapping(mapping: Mapping[str, object]) -> str:
    if not mapping:
        return "none"
    return ", ".join(
        f"{key}={_format_scalar(value)}" for key, value in mapping.items()
    )


def _format_scalar(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
