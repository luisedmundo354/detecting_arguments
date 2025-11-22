#!/usr/bin/env python3
"""Render Label Studio annotation trees with the root at the top."""

import argparse
import json
import os
import sys

from tree_visualization import (
    DEFAULT_OUTPUT_DIR,
    render_annotation_tree,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Graphviz tree from a Label Studio JSON annotation, "
            "saving the output inside the annotations tree visualization folder."
        )
    )
    parser.add_argument(
        "json_file",
        help="Path to a Label Studio annotation JSON file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where rendered trees are stored (default: "
            f"{DEFAULT_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "-f",
        "--format",
        default="pdf",
        help="Output format passed to Graphviz (e.g. pdf, png, svg).",
    )
    parser.add_argument(
        "--stem",
        help="Filename stem for the generated output (defaults to an auto slug).",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open the rendered file with the default viewer once generated.",
    )
    parser.add_argument(
        "--no-invert",
        dest="invert",
        action="store_false",
        help=(
            "Keep the original relation direction instead of flipping children "
            "to point away from their parents."
        ),
    )
    parser.set_defaults(invert=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.json_file):
        print(f"File '{args.json_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.json_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as err:
        print(f"Unable to parse '{args.json_file}': {err}.", file=sys.stderr)
        sys.exit(1)
    except OSError as err:
        print(f"Unable to read '{args.json_file}': {err}.", file=sys.stderr)
        sys.exit(1)

    try:
        output_path = render_annotation_tree(
            data,
            output_dir=args.output_dir,
            output_stem=args.stem,
            fmt=args.format,
            view=args.view,
            invert_relations=args.invert,
        )
    except Exception as err:  # pragma: no cover - defensive
        print(f"Failed to render tree: {err}", file=sys.stderr)
        sys.exit(1)

    print(f"Tree visualization written to {output_path}.")


if __name__ == "__main__":
    main()
