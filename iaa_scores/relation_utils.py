"""Helpers for interpreting stored Label Studio relations."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Mapping, Sequence, Set, Tuple

from dataset_statistics.modules import CaseAnnotation


def normalized_edge_set(
    annotation: CaseAnnotation,
    *,
    allowed_node_ids: Set[str] | None = None,
) -> Set[Tuple[str, str]]:
    """Return corrected parent->child edges by inverting stored relation direction."""

    edges: Set[Tuple[str, str]] = set()
    for relation in annotation.relations:
        parent_id = relation.target_id
        child_id = relation.source_id
        if allowed_node_ids is not None and (
            parent_id not in allowed_node_ids or child_id not in allowed_node_ids
        ):
            continue
        edges.add((parent_id, child_id))
    return edges


def normalized_adjacency(
    annotation: CaseAnnotation,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Return outgoing and incoming adjacency maps under corrected direction."""

    node_ids = {span.node_id for span in annotation.spans}
    outgoing: Dict[str, Set[str]] = defaultdict(set)
    incoming: Dict[str, Set[str]] = defaultdict(set)
    for parent_id, child_id in normalized_edge_set(annotation):
        if parent_id not in node_ids or child_id not in node_ids:
            continue
        outgoing[parent_id].add(child_id)
        incoming[child_id].add(parent_id)
    return outgoing, incoming


def raw_parent_id_matches_stored_relation(annotation: CaseAnnotation, implicit_node_id: str, parent_id: str) -> bool:
    """Return whether parentID agrees with at least one stored raw incoming relation."""

    for relation in annotation.relations:
        if relation.target_id == implicit_node_id and relation.source_id == parent_id:
            return True
    return False
