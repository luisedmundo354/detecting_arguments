"""Sentence-level corpus projection for IAA."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from dataset_statistics.modules import AnnotationSpan
from dataset_statistics.sentence_utils import overlap_length, split_sentences_with_offsets

from .annotation_graphs import load_annotation_graphs
from .models import CorpusData, OffsetCollection, SpanCollection

UNLABELED = "Unlabeled"


def build_sentence_corpus(
    root: Path,
    min_annotators: int,
    *,
    include_categories: Iterable[str] | None = None,
) -> CorpusData:
    """Project repaired span annotations onto sentences for sentence-level IAA."""

    graphs_by_doc = load_annotation_graphs(
        root,
        min_annotators=min_annotators,
        require_repaired_implicit_offsets=True,
    )

    doc_spans = {}
    doc_offsets = {}
    doc_files = {}
    doc_lengths = {}
    categories: Set[str] = set()
    files = sorted(path for ann_by_annotator in graphs_by_doc.values() for path in [record.path for record in ann_by_annotator.values()])

    requested_categories = set(include_categories) if include_categories is not None else None

    for ref_id, annotations_by_annotator in sorted(graphs_by_doc.items()):
        sample_annotation = next(iter(annotations_by_annotator.values())).annotation
        sentence_spans = split_sentences_with_offsets(sample_annotation.case_text)
        if not sentence_spans:
            raise ValueError(f"Sentence projection produced no sentences for ref_id {ref_id}")

        doc_spans[ref_id] = {}
        doc_offsets[ref_id] = {}
        doc_files[ref_id] = []
        doc_lengths[ref_id] = len(sample_annotation.case_text)

        for annotator, record in sorted(annotations_by_annotator.items()):
            annotation = record.annotation
            spans, offsets, labels = _project_annotation(annotation, sentence_spans)
            if requested_categories is not None:
                spans = {
                    label: sentence_texts
                    for label, sentence_texts in spans.items()
                    if label in requested_categories
                }
                offsets = {
                    label: sentence_offsets
                    for label, sentence_offsets in offsets.items()
                    if label in requested_categories
                }
                labels = labels & requested_categories
            doc_spans[ref_id][annotator] = spans
            doc_offsets[ref_id][annotator] = offsets
            doc_files[ref_id].append(record.path)
            categories.update(labels)

    if not categories:
        raise ValueError("Sentence projection produced no labeled sentence categories.")

    return CorpusData(
        doc_spans=doc_spans,
        doc_offsets=doc_offsets,
        doc_files=doc_files,
        doc_lengths=doc_lengths,
        categories=sorted(categories),
        files=files,
    )


def _project_annotation(
    annotation,
    sentence_spans,
) -> Tuple[SpanCollection, OffsetCollection, Set[str]]:
    explicit_spans = [
        span
        for span in annotation.spans
        if not span.is_implicit and span.start is not None and span.end is not None
    ]
    explicit_spans.sort(key=lambda span: (int(span.start), int(span.end), span.node_id))

    spans: SpanCollection = defaultdict(list)
    offsets: OffsetCollection = defaultdict(list)
    labels: Set[str] = set()

    for sentence_span in sentence_spans:
        best_key = None
        best_span: AnnotationSpan | None = None
        for span in explicit_spans:
            overlap = overlap_length(
                sentence_span.start,
                sentence_span.end,
                int(span.start),
                int(span.end),
            )
            if overlap <= 0:
                continue
            candidate_key = (-overlap, int(span.start), int(span.end), span.node_id)
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_span = span

        if best_span is None:
            continue
        if best_span.label == UNLABELED:
            continue

        spans[best_span.label].append(sentence_span.text)
        offsets[best_span.label].append((int(sentence_span.start), int(sentence_span.end)))
        labels.add(best_span.label)

    return dict(spans), dict(offsets), labels
