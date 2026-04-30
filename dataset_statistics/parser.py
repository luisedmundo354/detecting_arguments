"""Parsing helpers for Label Studio annotation exports."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .modules import AnnotationSpan, CaseAnnotation, DatasetCase, RelationEdge

YEAR_RE = re.compile(r"\b(18|19|20)\d{2}\b")
SECTION_368_RE = re.compile(
    r"\b(?:section|sec\.?|I\.R\.C\.?\s*§?)\s*368\s*\(([a-z])\)",
    flags=re.IGNORECASE,
)


def iter_annotation_files(annotation_dir: Path | str) -> List[Path]:
    root = Path(annotation_dir)
    if not root.exists():
        raise FileNotFoundError(f"Annotation directory not found: {root}")
    files = sorted(path for path in root.glob("*.json") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No JSON annotation files found in {root}")
    return files


def load_case_annotations(annotation_dir: Path | str) -> List[CaseAnnotation]:
    annotations: List[CaseAnnotation] = []
    for path in iter_annotation_files(annotation_dir):
        annotations.append(load_case_annotation(path))
    return annotations


def load_case_annotation(path: Path | str) -> CaseAnnotation:
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    task = payload.get("task", {}) or {}
    task_data = task.get("data", {}) or {}
    case_text = str(task_data.get("case_content") or "")
    header_text = extract_header(case_text)
    export_id = json_path.stem

    annotation_id = _coerce_optional_int(payload.get("id"))
    ref_id = _coerce_optional_int(task_data.get("ref_id"))
    annotator = _extract_annotator(payload, task_data)
    assigned_to = _extract_assigned_to(task_data)

    spans: List[AnnotationSpan] = []
    relations: List[RelationEdge] = []
    for item in payload.get("result", []) or []:
        item_type = item.get("type")
        if item_type == "labels":
            span = _parse_span(item)
            if span is not None:
                spans.append(span)
        elif item_type == "relation":
            relation = _parse_relation(item)
            if relation is not None:
                relations.append(relation)

    return CaseAnnotation(
        export_id=export_id,
        annotation_id=annotation_id,
        ref_id=ref_id,
        source_file=json_path.name,
        annotator=annotator,
        assigned_to=assigned_to,
        case_text=case_text,
        spans=tuple(spans),
        relations=tuple(relations),
        header_text=header_text,
        year=extract_case_year(header_text),
        subtype_368=extract_368_subtypes(case_text),
    )


def build_dataset_cases(
    annotation_dir: Path | str,
    *,
    view: str = "export",
) -> List[DatasetCase]:
    annotations = load_case_annotations(annotation_dir)
    if view == "export":
        return [
            DatasetCase(case_key=annotation.export_id, ref_id=annotation.ref_id, annotations=(annotation,))
            for annotation in annotations
        ]
    if view != "case":
        raise ValueError("view must be either 'export' or 'case'")

    grouped: Dict[Tuple[str, Optional[int]], List[CaseAnnotation]] = defaultdict(list)
    for annotation in annotations:
        key = str(annotation.ref_id) if annotation.ref_id is not None else annotation.export_id
        grouped[(key, annotation.ref_id)].append(annotation)

    dataset_cases: List[DatasetCase] = []
    for (case_key, ref_id), case_annotations in sorted(
        grouped.items(), key=lambda item: (_ref_sort_key(item[0][1]), item[0][0])
    ):
        ordered = sorted(case_annotations, key=lambda ann: ann.source_file)
        dataset_cases.append(
            DatasetCase(case_key=case_key, ref_id=ref_id, annotations=tuple(ordered))
        )
    return dataset_cases


def collect_dataset_metadata(annotations: Sequence[CaseAnnotation]) -> Dict[str, int]:
    ref_counts: Dict[int, int] = defaultdict(int)
    annotator_counts: Dict[str, int] = defaultdict(int)
    for annotation in annotations:
        if annotation.ref_id is not None:
            ref_counts[annotation.ref_id] += 1
        annotator_counts[annotation.annotator] += 1
    return {
        "export_count": len(annotations),
        "unique_ref_id_count": len(ref_counts),
        "double_annotated_case_count": sum(1 for count in ref_counts.values() if count > 1),
        "annotator_count": len(annotator_counts),
    }


def extract_header(case_text: str, *, max_nonempty_lines: int = 8) -> str:
    lines = [line.strip() for line in str(case_text).splitlines() if line.strip()]
    return "\n".join(lines[:max_nonempty_lines])


def extract_case_year(header_text: str) -> Optional[int]:
    match = YEAR_RE.search(header_text)
    if match is None:
        return None
    return int(match.group(0))


def extract_368_subtypes(case_text: str) -> Tuple[str, ...]:
    matches = {match.group(1).lower() for match in SECTION_368_RE.finditer(str(case_text))}
    return tuple(sorted(matches))


def _extract_annotator(payload: Dict[str, object], task_data: Dict[str, object]) -> str:
    completed_by = payload.get("completed_by", {}) or {}
    if isinstance(completed_by, dict):
        email = str(completed_by.get("email") or "").strip()
        if email:
            return email
    assigned_to = _extract_assigned_to(task_data)
    if assigned_to:
        return assigned_to[0]
    created_username = str(payload.get("created_username") or "").strip()
    if created_username:
        return created_username
    return "unknown"


def _extract_assigned_to(task_data: Dict[str, object]) -> Tuple[str, ...]:
    value = task_data.get("assigned_to")
    if isinstance(value, str):
        value = value.strip()
        return (value,) if value else ()
    if isinstance(value, list):
        assigned = [str(item).strip() for item in value if str(item).strip()]
        return tuple(assigned)
    return ()


def _parse_span(item: Dict[str, object]) -> Optional[AnnotationSpan]:
    value = item.get("value", {}) or {}
    if not isinstance(value, dict):
        return None
    labels = value.get("labels") or []
    label = str(labels[0]).strip() if labels else ""
    if not label:
        return None
    text = str(value.get("text") or "").strip()
    return AnnotationSpan(
        node_id=str(item.get("id") or ""),
        label=label,
        text=text,
        start=_coerce_optional_int(value.get("start")),
        end=_coerce_optional_int(value.get("end")),
        parent_id=_coerce_optional_str(item.get("parentID")),
        block_id=_extract_block_id(item.get("meta")),
    )


def _parse_relation(item: Dict[str, object]) -> Optional[RelationEdge]:
    source_id = _coerce_optional_str(item.get("from_id") or item.get("from"))
    target_id = _coerce_optional_str(item.get("to_id") or item.get("to"))
    if not source_id or not target_id:
        return None
    return RelationEdge(
        source_id=source_id,
        target_id=target_id,
        direction=str(item.get("direction") or "right"),
    )


def _extract_block_id(value: object) -> Optional[str]:
    if not isinstance(value, dict):
        return None
    return _coerce_optional_str(value.get("blockId"))


def _coerce_optional_int(value: object) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ref_sort_key(value: Optional[int]) -> Tuple[int, int]:
    if value is None:
        return (1, 10**18)
    return (0, int(value))
