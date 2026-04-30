"""Graph and span-level descriptive statistics for annotation exports."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .modules import AnnotationSpan, CaseAnnotation, RelationEdge
from .sentence_utils import count_words


def summarize_annotation_graph(annotation: CaseAnnotation) -> Dict[str, object]:
    spans_by_id: Dict[str, AnnotationSpan] = {span.node_id: span for span in annotation.spans}
    edges = [
        edge
        for edge in annotation.relations
        if edge.source_id in spans_by_id and edge.target_id in spans_by_id
    ]

    explicit_spans = [span for span in annotation.spans if not span.is_implicit]
    implicit_spans = [
        span for span in annotation.spans if span.is_normalized_implicit_intermediate
    ]

    explicit_counts_by_label = Counter(span.label for span in explicit_spans)
    all_node_counts_by_label = Counter(span.label for span in annotation.spans)
    word_counts_by_label = Counter()
    char_lengths_by_label: Dict[str, List[int]] = defaultdict(list)
    word_lengths_by_label: Dict[str, List[int]] = defaultdict(list)
    all_char_lengths: List[int] = []
    all_word_lengths: List[int] = []
    for span in explicit_spans:
        words = count_words(span.text)
        word_counts_by_label[span.label] += words
        char_lengths_by_label[span.label].append(span.char_length)
        word_lengths_by_label[span.label].append(words)
        all_char_lengths.append(span.char_length)
        all_word_lengths.append(words)

    undirected_adjacency = {node_id: set() for node_id in spans_by_id}
    out_adjacency = {node_id: set() for node_id in spans_by_id}
    inverted_adjacency = {node_id: set() for node_id in spans_by_id}
    indegree = Counter()
    outdegree = Counter()

    for edge in edges:
        undirected_adjacency[edge.source_id].add(edge.target_id)
        undirected_adjacency[edge.target_id].add(edge.source_id)
        out_adjacency[edge.source_id].add(edge.target_id)
        inverted_adjacency[edge.target_id].add(edge.source_id)
        indegree[edge.target_id] += 1
        outdegree[edge.source_id] += 1

    components = _weakly_connected_components(undirected_adjacency)
    nontrivial_components = [
        component for component in components if _component_edge_count(component, edges) > 0
    ]
    disconnected_count = sum(
        1 for node_id, neighbors in undirected_adjacency.items() if not neighbors
    )

    depths: List[int] = []
    branching_values: List[int] = []
    max_depth = 0
    for component in nontrivial_components:
        component_roots = _select_component_roots(component, spans_by_id, outdegree)
        component_depths = _compute_component_depths(component_roots, component, inverted_adjacency)
        depths.extend(component_depths.values())
        if component_depths:
            max_depth = max(max_depth, max(component_depths.values()))
        for node_id in component:
            child_count = len(inverted_adjacency[node_id] & component)
            if child_count > 0:
                branching_values.append(child_count)

    average_depth = statistics.mean(depths) if depths else None
    branching_factor = statistics.mean(branching_values) if branching_values else 0.0

    return {
        "node_by_id": spans_by_id,
        "edge_count": len(edges),
        "tree_count": len(nontrivial_components),
        "disconnected_count": disconnected_count,
        "implicit_count": len(implicit_spans),
        "explicit_span_count": len(explicit_spans),
        "node_count": len(annotation.spans),
        "explicit_counts_by_label": dict(explicit_counts_by_label),
        "all_node_counts_by_label": dict(all_node_counts_by_label),
        "word_counts_by_label": dict(word_counts_by_label),
        "span_char_lengths": all_char_lengths,
        "span_word_lengths": all_word_lengths,
        "span_char_lengths_by_label": dict(char_lengths_by_label),
        "span_word_lengths_by_label": dict(word_lengths_by_label),
        "average_depth": average_depth,
        "max_depth": max_depth,
        "branching_factor": branching_factor,
    }


def summarize_lengths(lengths: Sequence[int]) -> Dict[str, float | None]:
    if not lengths:
        return {"average": None, "median": None}
    return {
        "average": float(statistics.mean(lengths)),
        "median": float(statistics.median(lengths)),
    }


def summarize_lengths_by_label(
    lengths_by_label: Dict[str, Sequence[int]]
) -> Dict[str, Dict[str, float | None]]:
    return {
        label: summarize_lengths(lengths)
        for label, lengths in sorted(lengths_by_label.items())
    }


def _weakly_connected_components(
    adjacency: Dict[str, Set[str]]
) -> List[Set[str]]:
    remaining = set(adjacency)
    components: List[Set[str]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = {start}
        while stack:
            node_id = stack.pop()
            for neighbor in adjacency[node_id]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _component_edge_count(component: Set[str], edges: Sequence[RelationEdge]) -> int:
    return sum(
        1 for edge in edges if edge.source_id in component and edge.target_id in component
    )


def _select_component_roots(
    component: Set[str],
    spans_by_id: Dict[str, AnnotationSpan],
    outdegree: Counter,
) -> List[str]:
    sinks = [node_id for node_id in component if outdegree.get(node_id, 0) == 0]
    preferred = [
        node_id
        for node_id in sinks
        if spans_by_id[node_id].label.strip().lower() == "conclusion"
    ]
    roots = preferred or sinks or sorted(component)
    return sorted(roots)


def _compute_component_depths(
    roots: Sequence[str],
    component: Set[str],
    inverted_adjacency: Dict[str, Set[str]],
) -> Dict[str, int]:
    depths = {root: 0 for root in roots}
    stack: List[Tuple[str, int]] = [(root, 0) for root in roots]
    while stack:
        node_id, depth = stack.pop()
        for child_id in sorted(inverted_adjacency[node_id] & component):
            next_depth = depth + 1
            if next_depth > depths.get(child_id, -1):
                depths[child_id] = next_depth
                stack.append((child_id, next_depth))
    return depths
