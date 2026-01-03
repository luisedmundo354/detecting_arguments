"""Utilities for rendering Label Studio annotation trees with Graphviz."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, Optional

import graphviz

from html_utils import html_label

DEFAULT_OUTPUT_DIR = "annotations_tree_visualization"


def build_graph(
    data: Dict[str, Any],
    *,
    invert_relations: bool = True,
    rankdir: str = "TB",
) -> graphviz.Digraph:
    """Build a Graphviz graph from a Label Studio annotation export."""
    dot = graphviz.Digraph(comment="Tree from Label Studio")
    dot.attr(size="8.5,11!", page="8.5,11")
    dot.attr(rankdir=rankdir)
    dot.attr("node", shape="none")
    dot.attr("edge", arrowhead="none")

    nodes = list(_iter_nodes(data))
    for item in nodes:
        node_id = item.get("id")
        if not node_id:
            continue
        labels = item.get("value", {}).get("labels", [])
        title = labels[0] if labels else "Node"
        content = item.get("value", {}).get("text", "")
        dot.node(node_id, label=html_label(title, content))

    for src, dst, edge_label in _iter_edges(
        data, invert_relations=invert_relations
    ):
        if edge_label:
            dot.edge(src, dst, label=edge_label)
        else:
            dot.edge(src, dst)

    return dot


def render_annotation_tree(
    data: Dict[str, Any],
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    output_stem: Optional[str] = None,
    fmt: str = "pdf",
    view: bool = False,
    invert_relations: bool = True,
    rankdir: str = "TB",
) -> str:
    """Render a Graphviz visualization and return the generated path."""
    graph = build_graph(
        data,
        invert_relations=invert_relations,
        rankdir=rankdir,
    )
    graph.format = fmt

    os.makedirs(output_dir, exist_ok=True)

    stem = output_stem or default_output_stem(data)
    filename = os.path.join(output_dir, stem)

    return graph.render(filename=filename, view=view)


def default_output_stem(data: Dict[str, Any]) -> str:
    """Generate a safe filename stem for an annotation tree."""
    case_content = (data.get("task", {}).get("data", {}).get("case_content") or "").strip()
    annotator_email = (data.get("completed_by", {}).get("email") or "").strip()
    annotator_prefix = annotator_email.split("@", 1)[0]
    annotation_id = data.get("id")

    parts = [
        _slugify(case_content[:40]),
        annotator_prefix or None,
        str(annotation_id) if annotation_id is not None else None,
    ]

    filtered = [part for part in parts if part]
    return "_".join(filtered) if filtered else "annotation_tree"


def _iter_nodes(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for item in data.get("result", []):
        if item.get("type") == "labels":
            yield item


def _iter_edges(
    data: Dict[str, Any],
    *,
    invert_relations: bool,
) -> Iterable[tuple[str, str, Optional[str]]]:
    for item in data.get("result", []):
        if item.get("type") != "relation":
            continue
        src = item.get("from_id") or item.get("from")
        dst = item.get("to_id") or item.get("to")
        if not src or not dst:
            continue
        if invert_relations:
            src, dst = dst, src
        labels = item.get("labels") or []
        edge_label = ",".join(labels) if labels else None
        yield src, dst, edge_label


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-z0-9_-]", "", value)
    return value
