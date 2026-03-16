"""Shared data structures and enums for the IAA pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Set, Tuple

DocId = Hashable
SpanCollection = Dict[str, List[str]]
OffsetCollection = Dict[str, List[Tuple[int, int]]]
DocumentSpans = Dict[str, SpanCollection]
DocumentOffsets = Dict[str, OffsetCollection]


class AlphaMode(str, Enum):
    """Supported Krippendorff alpha variants."""

    NOMINAL = "nominal"
    UNITIZING = "unitizing"

    def label(self) -> str:
        if self is AlphaMode.UNITIZING:
            return "Krippendorff alpha (unitizing)"
        return "Krippendorff alpha"


@dataclass(frozen=True)
class AnnotationRecord:
    """Container for a single Label Studio JSON export."""

    ref_id: Optional[DocId]
    annotator: str
    spans: SpanCollection
    offsets: OffsetCollection
    categories: Set[str]
    document_length: Optional[int] = None


@dataclass
class CorpusData:
    """Aggregated annotations keyed by document and annotator."""

    doc_spans: Dict[DocId, DocumentSpans]
    doc_offsets: Dict[DocId, DocumentOffsets]
    doc_files: Dict[DocId, List[Path]]
    doc_lengths: Dict[DocId, int]
    categories: List[str]
    files: List[Path]


@dataclass
class IAAScores:
    """Final per-document and corpus-level scores."""

    per_doc_f1: Dict[DocId, Dict[str, float]]
    per_doc_alpha: Dict[DocId, Dict[str, float]]
    overall_f1: Dict[str, float]
    overall_alpha: Dict[str, float]
    doc_pair_counts: Dict[DocId, int]


@dataclass(frozen=True)
class EdgeAgreementDocResult:
    """Per-document direct-edge agreement details."""

    ref_id: DocId
    annotators: Tuple[str, str]
    files: Tuple[Path, ...]
    context_count: int
    agreement_count: int
    observed_agreement: float
    expected_agreement: float
    kappa: float
    positive_rate_by_annotator: Dict[str, float]


@dataclass(frozen=True)
class EdgeAgreementScores:
    """Corpus-level direct-edge agreement details."""

    per_doc: Dict[DocId, EdgeAgreementDocResult]
    overall_context_count: int
    overall_agreement_count: int
    overall_observed_agreement: float
    overall_expected_agreement: float
    overall_kappa: float
    overall_positive_rate_by_annotator: Dict[str, float]


@dataclass(frozen=True)
class ImplicitInsertionDocResult:
    """Per-document implicit insertion agreement details."""

    ref_id: DocId
    annotators: Tuple[str, str]
    files: Tuple[Path, ...]
    context_count: int
    agreement_count: int
    yes_yes_count: int
    yes_no_count: int
    no_yes_count: int
    no_no_count: int
    observed_agreement: float
    expected_agreement: float
    kappa: float
    positive_agreement: float
    negative_agreement: float
    insertion_rate_by_annotator: Dict[str, float]
    usable_implicit_nodes_by_annotator: Dict[str, int]
    excluded_implicit_nodes_by_annotator: Dict[str, Dict[str, int]]
    context_source_counts: Dict[str, int]
    sample_contexts: List[Dict[str, object]]


@dataclass(frozen=True)
class ImplicitInsertionScores:
    """Corpus-level implicit insertion agreement details."""

    per_doc: Dict[DocId, ImplicitInsertionDocResult]
    overall_context_count: int
    overall_agreement_count: int
    overall_yes_yes_count: int
    overall_yes_no_count: int
    overall_no_yes_count: int
    overall_no_no_count: int
    overall_observed_agreement: float
    overall_expected_agreement: float
    overall_kappa: float
    overall_positive_agreement: float
    overall_negative_agreement: float
    overall_insertion_rate_by_annotator: Dict[str, float]
    overall_usable_implicit_nodes_by_annotator: Dict[str, int]
    overall_excluded_implicit_nodes_by_annotator: Dict[str, Dict[str, int]]
