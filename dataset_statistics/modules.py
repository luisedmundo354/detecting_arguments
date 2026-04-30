"""Core data structures for dataset statistics over Label Studio exports."""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize(val) for key, val in value.__dict__.items()}
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(val) for key, val in value.items()}
    return value


class SerializableDataclass:
    """Mixin that exposes dataclasses as JSON-friendly dictionaries."""

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class AnnotationSpan(SerializableDataclass):
    node_id: str
    label: str
    text: str
    start: Optional[int]
    end: Optional[int]
    parent_id: Optional[str] = None
    block_id: Optional[str] = None

    @property
    def is_implicit(self) -> bool:
        return self.start is None or self.end is None

    @property
    def is_normalized_implicit_intermediate(self) -> bool:
        text_norm = self.text.strip().lower()
        return text_norm == "intermediate" or text_norm.startswith(
            "implicit intermediate conclusion"
        )

    @property
    def char_length(self) -> int:
        if self.is_implicit or self.start is None or self.end is None:
            return 0
        return max(0, int(self.end) - int(self.start))

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class RelationEdge(SerializableDataclass):
    source_id: str
    target_id: str
    direction: str = "right"


@dataclass(frozen=True)
class SentenceSpan(SerializableDataclass):
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class CaseAnnotation(SerializableDataclass):
    export_id: str
    annotation_id: Optional[int]
    ref_id: Optional[int]
    source_file: str
    annotator: str
    assigned_to: Tuple[str, ...]
    case_text: str
    spans: Tuple[AnnotationSpan, ...]
    relations: Tuple[RelationEdge, ...]
    header_text: str
    year: Optional[int]
    subtype_368: Tuple[str, ...]


@dataclass(frozen=True)
class DatasetCase(SerializableDataclass):
    case_key: str
    ref_id: Optional[int]
    annotations: Tuple[CaseAnnotation, ...]

    @property
    def source_files(self) -> Tuple[str, ...]:
        return tuple(annotation.source_file for annotation in self.annotations)

    @property
    def annotators(self) -> Tuple[str, ...]:
        ordered = []
        for annotation in self.annotations:
            if annotation.annotator not in ordered:
                ordered.append(annotation.annotator)
        return tuple(ordered)


@dataclass(frozen=True)
class SentenceRecord(SerializableDataclass):
    passage_id: str
    doc_id: str
    ref_id: Optional[int]
    view_name: str
    source_files: Tuple[str, ...]
    label: str
    text: str
    start: int
    end: int
    source_node_id: Optional[str]
    is_implicit: bool
    order: int
    annotator_labels: Dict[str, str] = field(default_factory=dict)
    label_votes: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseStatistics(SerializableDataclass):
    case_key: str
    ref_id: Optional[int]
    source_files: Tuple[str, ...]
    annotators: Tuple[str, ...]
    annotation_count: int
    year: Optional[int]
    subtype_368: Tuple[str, ...]
    case_word_count: int
    case_sentence_count: int
    explicit_span_count: float
    implicit_insertion_count: float
    node_count: float
    edge_count: float
    argument_tree_count: float
    average_depth: Optional[float]
    max_depth: float
    branching_factor: float
    disconnected_span_count: float
    disconnected_span_percentage: float
    implicit_insertion_percentage: float
    explicit_span_counts_by_label: Dict[str, float]
    all_node_counts_by_label: Dict[str, float]
    sentence_counts_by_label: Dict[str, float]
    label_word_counts: Dict[str, float]
    span_length_summary: Dict[str, Optional[float]]
    span_length_by_label: Dict[str, Dict[str, Optional[float]]]
    imbalance: Dict[str, Optional[float]]


@dataclass(frozen=True)
class DatasetStatistics(SerializableDataclass):
    dataset_name: str
    source_dir: str
    view_name: str
    case_count: int
    export_count: int
    unique_ref_id_count: int
    double_annotated_case_count: int
    annotator_counts: Dict[str, int]
    label_distribution: Dict[str, Dict[str, float]]
    all_node_label_distribution: Dict[str, Dict[str, float]]
    sentence_label_distribution: Dict[str, Dict[str, float]]
    year_distribution: Dict[str, int]
    subtype_368_distribution: Dict[str, Dict[str, float]]
    summary_metrics: Dict[str, Any]
    case_statistics: Tuple[CaseStatistics, ...]
    notes: Tuple[str, ...] = ()


SchemaValue = Mapping[str, Any] | Sequence[Any] | str | int | float | None
