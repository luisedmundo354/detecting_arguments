#!/usr/bin/env python3
"""
Generate a tree visualization from a Label Studio JSON export.
Reads annotation nodes and relations from a JSON file and renders a graph using Graphviz.
"""
import json
import argparse
import textwrap
import graphviz
import html

def html_label(title, content, width=150, wrap_width=40):
    wrapped = textwrap.fill(content, wrap_width)
    wrapped_html = html.escape(wrapped, quote=True).replace("\n", "<BR ALIGN='left'/>")
    colors = {
        'Rule': 'green',
        'Analysis': 'lightblue',
        'Conclusion': 'red',
        'Background Facts': 'purple',
        'Procedural History': 'yellow',
    }
    header_color = colors.get(title, 'lightblue')
    safe_title = html.escape(title, quote=True)
    return f'''<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
  <TR><TD BGCOLOR="{header_color}" WIDTH="{width}"><B>{safe_title}</B></TD></TR>
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
    dot.attr(rankdir="TB")
    dot.attr('node', shape='none')

    for item in data.get('result', []):
        if item.get('type') == 'labels':
            node_id = item.get('id')
            labels = item.get('value', {}).get('labels', []) or []
            title = labels[0] if labels else 'Node'
            content = item.get('value', {}).get('text', '')
            label = html_label(title, content)
            dot.node(node_id, label=label)

    for item in data.get('result', []):
        if item.get('type') == 'relation':
            src = item.get('from_id') or item.get('from')
            dst = item.get('to_id') or item.get('to')
            if not src or not dst:
                continue
            lbls = item.get('labels') or []
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

    with open(args.json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dot = build_graph(data)
    dot.format = args.format

    case_content = (data.get('task', {}).get('data', {}).get('case_content') or '')
    first_name = (data.get('completed_by', {}).get('first_name') or '')
    id_value = str(data.get('id') or '')
    output_name = f"annotations_tree_visualization/{case_content[:10]}_{first_name}_{id_value}"

    dot.render(filename=output_name, view=args.view)

if __name__ == '__main__':  # noqa: C901
    main()
