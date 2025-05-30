#!/usr/bin/env python3
"""
Generate a tree visualization from a Label Studio JSON export.
Reads annotation nodes and relations from a JSON file and renders a graph using Graphviz.
"""
import json
import argparse
import textwrap
import graphviz

def html_label(title, content, width=150, wrap_width=40):
    """
    Create an HTML-like label for a node with a title and content.
    Wraps content text and embeds in a simple table.
    """
    # Normalize title if needed
    wrapped = textwrap.fill(content, wrap_width)
    wrapped_html = wrapped.replace("\n", "<BR ALIGN='left'/>")
    return f'''<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
  <TR><TD BGCOLOR="lightblue" WIDTH="{width}"><B>{title}</B></TD></TR>
  <TR><TD ALIGN="left" WIDTH="{width}">{wrapped_html}</TD></TR>
</TABLE>
>'''

def build_graph(data):
    """
    Build a Graphviz Digraph from loaded JSON data.
    Expects data to have a 'result' list with items of type 'labels' or 'relation'.
    """
    dot = graphviz.Digraph(comment='Tree from JSON')
    dot.attr(size="8.5,11!", page="8.5,11")
    # Top-down orientation: parents above children
    dot.attr(rankdir="TB")
    dot.attr('node', shape='none')

    # First, create nodes
    for item in data.get('result', []):
        if item.get('type') == 'labels':
            node_id = item.get('id')
            # Title from labels list, fallback to type
            labels = item.get('value', {}).get('labels', []) or []
            title = labels[0] if labels else 'Node'
            content = item.get('value', {}).get('text', '')
            label = html_label(title, content)
            dot.node(node_id, label=label)

    # Then, create edges
    for item in data.get('result', []):
        if item.get('type') == 'relation':
            src = item.get('from_id') or item.get('from')
            dst = item.get('to_id') or item.get('to')
            if not src or not dst:
                continue
            lbls = item.get('labels') or []
            # Combine multiple labels if present
            edge_label = ','.join(lbls) if lbls else None
            if edge_label:
                dot.edge(src, dst, label=edge_label)
            else:
                dot.edge(src, dst)
    return dot

def main():
    parser = argparse.ArgumentParser(
        description='Generate a tree diagram from a Label Studio JSON file.'
    )
    parser.add_argument('json_file', help='Path to the Label Studio JSON file')
    parser.add_argument('-o', '--output', default='tree_output',
                        help='Output file prefix (default: tree_output)')
    parser.add_argument('-f', '--format', default='pdf',
                        help='Output format (e.g., pdf, png, svg)')
    parser.add_argument('--view', action='store_true',
                        help='Open the rendered file with the default viewer')
    args = parser.parse_args()

    # Load JSON data
    with open(args.json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dot = build_graph(data)
    # Render output
    dot.format = args.format
    dot.render(filename=args.output, view=args.view)

if __name__ == '__main__':  # noqa: C901
    main()