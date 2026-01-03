"""Utilities to build HTML labels for Graphviz tree nodes."""

from __future__ import annotations

import html
import textwrap


LABEL_COLORS = {
    "Rule": "#b6e3b6",
    "Analysis": "#add8ff",
    "Conclusion": "#ffb3b3",
    "Background Facts": "#d0bdf4",
    "Procedural History": "#fff5ba",
}


def html_label(title: str, content: str, *, width: int = 150, wrap_width: int = 40) -> str:
    """Return an HTML table label that keeps long text readable."""
    wrapped = textwrap.fill(content or "", wrap_width)
    wrapped_html = html.escape(wrapped, quote=True).replace("\n", "<BR ALIGN='left'/>")
    safe_title = html.escape(title or "Node", quote=True)
    header_color = LABEL_COLORS.get(title, "#add8ff")
    return (
        "<\n"
        f"<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"4\">\n"
        f"  <TR><TD BGCOLOR=\"{header_color}\" WIDTH=\"{width}\"><B>{safe_title}</B></TD></TR>\n"
        f"  <TR><TD ALIGN=\"left\" WIDTH=\"{width}\">{wrapped_html}</TD></TR>\n"
        "</TABLE>\n"
        ">"
    )
