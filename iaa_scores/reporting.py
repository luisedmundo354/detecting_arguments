"""Console-friendly presentation helpers for IAA metrics."""

from __future__ import annotations

from .models import AlphaMode, CorpusData, IAAScores


def print_header(file_count: int, doc_count: int, categories: list[str]) -> None:
    print(f"Found {file_count} files; included {doc_count} documents")
    print("Categories:", categories)


def print_per_document_scores(
    corpus: CorpusData,
    scores: IAAScores,
    alpha_mode: AlphaMode,
) -> None:
    alpha_label = alpha_mode.label() + " by category:"
    for ref_id in sorted(corpus.doc_spans.keys()):
        annotator_count = len(corpus.doc_spans[ref_id])
        print(f"-- ref_id: {ref_id} (annotators: {annotator_count}) --")
        files = [path.name for path in corpus.doc_files.get(ref_id, [])]
        if files:
            print(f"\tFiles: {', '.join(files)}")
        print("F1 by category:")
        for cat in corpus.categories:
            val = scores.per_doc_f1.get(ref_id, {}).get(cat, float("nan"))
            print(f"\t{cat}: {val}")
        print(alpha_label)
        for cat in corpus.categories:
            val = scores.per_doc_alpha.get(ref_id, {}).get(cat, float("nan"))
            print(f"\t{cat}: {val}")


def print_overall_scores(
    corpus: CorpusData,
    scores: IAAScores,
    alpha_mode: AlphaMode,
) -> None:
    print("\nOverall across documents:")
    print("F1 by category (micro average over annotator pairs):")
    for cat in corpus.categories:
        val = scores.overall_f1.get(cat, float("nan"))
        print(f"\t{cat}: {val}")
    print(alpha_mode.label() + " by category:")
    for cat in corpus.categories:
        val = scores.overall_alpha.get(cat, float("nan"))
        print(f"\t{cat}: {val}")
