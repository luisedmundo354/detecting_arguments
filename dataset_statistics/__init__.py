"""Dataset statistics package for corporate reorganization annotations."""

from .core import build_sentence_dataset, compute_dataset_statistics
from .modules import (
    AnnotationSpan,
    CaseAnnotation,
    CaseStatistics,
    DatasetCase,
    DatasetStatistics,
    RelationEdge,
    SentenceRecord,
    SentenceSpan,
)
from .reporting import format_statistics_report, save_text_report
from .schema import describe_annotation_schema

__all__ = [
    "AnnotationSpan",
    "CaseAnnotation",
    "CaseStatistics",
    "DatasetCase",
    "DatasetStatistics",
    "RelationEdge",
    "SentenceRecord",
    "SentenceSpan",
    "build_sentence_dataset",
    "compute_dataset_statistics",
    "describe_annotation_schema",
    "format_statistics_report",
    "save_text_report",
]
