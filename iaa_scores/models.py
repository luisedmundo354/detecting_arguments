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
