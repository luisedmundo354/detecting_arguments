"""Utilities for building a corpus of annotations from disk."""

from __future__ import annotations

from pathlib import Path
from typing import List

from .loaders import load_annotation_payload
from .models import CorpusData


def collect_annotation_files(root: Path) -> List[Path]:
    """Return all JSON files in the target directory, sorted for stability."""

    return sorted(p for p in root.glob("*.json") if p.is_file())


def build_corpus(root: Path, min_annotators: int) -> CorpusData:
    """Load every annotation file and keep only documents with enough coders."""

    files = collect_annotation_files(root)
    if not files:
        raise ValueError(f"No JSON files found in {root}")

    doc_spans = {}
    doc_offsets = {}
    doc_files = {}
    doc_lengths = {}
    categories = set()

    for path in files:
        record = load_annotation_payload(path)
        if record.ref_id is None:
            continue
        doc_spans.setdefault(record.ref_id, {})[record.annotator] = record.spans
        doc_offsets.setdefault(record.ref_id, {})[record.annotator] = record.offsets
        doc_files.setdefault(record.ref_id, []).append(path)
        if record.document_length is not None:
            doc_lengths[record.ref_id] = max(doc_lengths.get(record.ref_id, 0), record.document_length)
        categories.update(record.categories)

    if not doc_spans:
        raise ValueError(f"No annotations with valid ref_id values found in {root}")

    filtered_spans = {
        ref_id: ann_by_annotator
        for ref_id, ann_by_annotator in doc_spans.items()
        if len(ann_by_annotator) >= min_annotators
    }

    if not filtered_spans:
        raise ValueError(
            "After filtering, no documents have the requested number of annotators."
        )

    filtered_offsets = {
        ref_id: doc_offsets.get(ref_id, {})
        for ref_id in filtered_spans.keys()
    }
    filtered_doc_files = {
        ref_id: doc_files.get(ref_id, [])
        for ref_id in filtered_spans.keys()
    }
    filtered_doc_lengths = {
        ref_id: doc_lengths[ref_id]
        for ref_id in filtered_spans.keys()
        if ref_id in doc_lengths
    }

    return CorpusData(
        doc_spans=filtered_spans,
        doc_offsets=filtered_offsets,
        doc_files=filtered_doc_files,
        doc_lengths=filtered_doc_lengths,
        categories=sorted(categories),
        files=files,
    )
