"""Command-line interface for the SOCES IAA toolkit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from . import DEFAULT_ANNOTATION_DIR
from .corpus import build_corpus
from .models import AlphaMode
from .pipeline import evaluate_corpus
from .reporting import (
    print_header,
    print_overall_scores,
    print_per_document_scores,
)

METRIC_CHOICES = ("yujianbo", "higueramico")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute inter-annotator agreement (IAA) for SOCES spans.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR,
        help="Directory with Label Studio JSON exports (default: repository annotations folder).",
    )
    parser.add_argument(
        "--min-annotators",
        type=int,
        default=2,
        help="Minimum annotators per document to include in the evaluation.",
    )
    parser.add_argument(
        "--alpha-u",
        action="store_true",
        help="Compute Krippendorff's alpha for unitizing (character offsets).",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="For alpha-u: include background regions where no annotator labeled a span.",
    )
    parser.add_argument(
        "--metric",
        choices=METRIC_CHOICES,
        default="yujianbo",
        help="Normalized edit-distance metric used for F1 alignment.",
    )
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.1,
        help="Minimum similarity required for spans to be considered a match (0-1).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.input_dir.expanduser()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"--input-dir {root} is not a directory")

    try:
        corpus = build_corpus(root, args.min_annotators)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    alpha_mode = AlphaMode.UNITIZING if args.alpha_u else AlphaMode.NOMINAL
    scores = evaluate_corpus(
        corpus,
        alpha_mode=alpha_mode,
        include_background=args.include_background,
        metric=args.metric,
        min_sim=args.min_sim,
    )

    print_header(len(corpus.files), len(corpus.doc_spans), corpus.categories)
    print("\nPer-document results:")
    print_per_document_scores(corpus, scores, alpha_mode)
    print_overall_scores(corpus, scores, alpha_mode)


if __name__ == "__main__":  # pragma: no cover
    main()
