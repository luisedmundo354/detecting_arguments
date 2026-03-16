"""Strict loaders for repaired IAA annotation graphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from dataset_statistics.modules import AnnotationSpan, CaseAnnotation
from dataset_statistics.parser import load_case_annotation

from .corpus import collect_annotation_files
from .models import DocId
from .relation_utils import raw_parent_id_matches_stored_relation


@dataclass(frozen=True)
class AnnotationGraphRecord:
    """Annotation plus its source path."""

    path: Path
    annotation: CaseAnnotation


DocumentGraphs = Dict[DocId, Dict[str, AnnotationGraphRecord]]


def load_annotation_graphs(
    root: Path,
    *,
    min_annotators: int = 2,
    require_repaired_implicit_offsets: bool = True,
) -> DocumentGraphs:
    """Load and validate Label Studio exports keyed by ref_id and annotator."""

    files = collect_annotation_files(root)
    if not files:
        raise ValueError(f"No JSON files found in {root}")

    grouped: DocumentGraphs = {}
    canonical_text_by_ref: Dict[DocId, str] = {}

    for path in files:
        annotation = load_case_annotation(path)
        if annotation.ref_id is None:
            raise ValueError(f"Missing task.data.ref_id in {path}")
        if not annotation.annotator:
            raise ValueError(f"Missing completed_by.email in {path}")

        _validate_annotation_graph(
            annotation,
            path,
            require_repaired_implicit_offsets=require_repaired_implicit_offsets,
        )

        by_annotator = grouped.setdefault(annotation.ref_id, {})
        if annotation.annotator in by_annotator:
            raise ValueError(
                f"Duplicate annotator '{annotation.annotator}' for ref_id {annotation.ref_id}: "
                f"{by_annotator[annotation.annotator].path} and {path}"
            )

        canonical_text = canonical_text_by_ref.get(annotation.ref_id)
        if canonical_text is None:
            canonical_text_by_ref[annotation.ref_id] = annotation.case_text
        elif canonical_text != annotation.case_text:
            raise ValueError(
                f"Case text mismatch for ref_id {annotation.ref_id}: repaired IAA exports must share identical canonical text."
            )

        by_annotator[annotation.annotator] = AnnotationGraphRecord(path=path, annotation=annotation)

    filtered = {
        ref_id: by_annotator
        for ref_id, by_annotator in grouped.items()
        if len(by_annotator) >= min_annotators
    }
    if not filtered:
        raise ValueError("After filtering, no documents have the requested number of annotators.")
    return filtered


def validate_repaired_iaa_exports(root: Path) -> None:
    """Fail loudly if the repaired IAA export invariants are violated."""

    load_annotation_graphs(root, min_annotators=1, require_repaired_implicit_offsets=True)


def _validate_annotation_graph(
    annotation: CaseAnnotation,
    path: Path,
    *,
    require_repaired_implicit_offsets: bool,
) -> None:
    node_ids = {span.node_id for span in annotation.spans}
    if len(node_ids) != len(annotation.spans):
        raise ValueError(f"Duplicate span ids found in {path}")

    for span in annotation.spans:
        _validate_span(
            span,
            annotation.case_text,
            path,
            require_repaired_implicit_offsets=require_repaired_implicit_offsets,
        )
        if span.parent_id is not None and span.is_normalized_implicit_intermediate:
            if not raw_parent_id_matches_stored_relation(annotation, span.node_id, span.parent_id):
                raise ValueError(
                    f"Implicit node {span.node_id} in {path} has parentID {span.parent_id} "
                    "that does not match any stored incoming relation."
                )

    for relation in annotation.relations:
        if relation.source_id not in node_ids:
            raise ValueError(f"Relation source '{relation.source_id}' not found in {path}")
        if relation.target_id not in node_ids:
            raise ValueError(f"Relation target '{relation.target_id}' not found in {path}")


def _validate_span(
    span: AnnotationSpan,
    case_text: str,
    path: Path,
    *,
    require_repaired_implicit_offsets: bool,
) -> None:
    text_length = len(case_text)
    looks_implicit = span.is_normalized_implicit_intermediate

    if looks_implicit:
        if require_repaired_implicit_offsets and (span.start is not None or span.end is not None):
            raise ValueError(
                f"Implicit intermediate conclusion still has offsets in {path} for node {span.node_id}. "
                f"Use the repaired final_annotations_iaa_set exports."
            )
        return

    if span.start is None or span.end is None:
        raise ValueError(
            f"Non-implicit span {span.node_id} in {path} has null offsets."
        )
    if span.start < 0 or span.end < 0 or span.start >= span.end:
        raise ValueError(f"Invalid offsets [{span.start}, {span.end}) for node {span.node_id} in {path}")
    if span.end > text_length:
        raise ValueError(
            f"Span {span.node_id} in {path} ends past document length {text_length}: end={span.end}"
        )

    observed_text = case_text[span.start : span.end]
    if observed_text != span.text:
        raise ValueError(
            f"Span text mismatch for node {span.node_id} in {path}: expected exact substring match after repair."
        )
