"""File-system helpers for turning Label Studio exports into Python objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .models import AnnotationRecord, OffsetCollection, SpanCollection


def load_annotation_payload(path: Path) -> AnnotationRecord:
    """Read a single JSON annotation file and extract spans plus offsets."""

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    annotator = (data.get("completed_by", {}) or {}).get("email", "")
    ref_id = _coerce_ref_id((data.get("task", {}) or {}).get("data", {}).get("ref_id"))
    document_length = _extract_document_length(data)

    spans, span_labels = _extract_text_spans(data)
    offsets, offset_labels = _extract_offsets(data)
    labels = span_labels | offset_labels

    return AnnotationRecord(
        ref_id=ref_id,
        annotator=annotator,
        spans=spans,
        offsets=offsets,
        categories=labels,
        document_length=document_length,
    )


def _coerce_ref_id(ref_id):
    if ref_id is None:
        return None
    try:
        return int(ref_id)
    except (TypeError, ValueError):
        return ref_id


def _extract_document_length(data: Dict) -> int | None:
    task_data = (data.get("task", {}) or {}).get("data", {}) or {}
    case_content = task_data.get("case_content")
    if case_content is None:
        return None
    if isinstance(case_content, str):
        return len(case_content)
    return None


def _extract_text_spans(data: Dict) -> Tuple[SpanCollection, Set[str]]:
    spans: SpanCollection = {}
    labels: Set[str] = set()

    for item in data.get("result", []) or []:
        if item.get("type") != "labels":
            continue
        value = item.get("value", {}) or {}
        cats = value.get("labels", []) or []
        start = value.get("start")
        if start is None:
            start = value.get("startOffset")
        end = value.get("end")
        if end is None:
            end = value.get("endOffset")
        if start is None or end is None:
            continue
        try:
            span = (int(start), int(end))
        except (TypeError, ValueError):
            continue
        if span[0] >= span[1]:
            continue
        text = (value.get("text") or "").strip()
        if not text:
            continue
        for cat in cats:
            labels.add(cat)
            spans.setdefault(cat, []).append(text)

    return spans, labels


def _extract_offsets(data: Dict) -> Tuple[OffsetCollection, Set[str]]:
    offsets: OffsetCollection = {}
    labels: Set[str] = set()

    for item in data.get("result", []) or []:
        if item.get("type") != "labels":
            continue
        value = item.get("value", {}) or {}
        cats = value.get("labels", []) or []
        start = value.get("start")
        if start is None:
            start = value.get("startOffset")
        end = value.get("end")
        if end is None:
            end = value.get("endOffset")
        if start is None or end is None:
            continue
        try:
            span = (int(start), int(end))
        except (TypeError, ValueError):
            continue
        if span[0] >= span[1]:
            continue
        for cat in cats:
            labels.add(cat)
            offsets.setdefault(cat, []).append(span)

    return offsets, labels
