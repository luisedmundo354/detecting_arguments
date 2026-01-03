"""Orchestrates corpus evaluation for IAA metrics."""

from __future__ import annotations

from tqdm import tqdm

from .alpha import (
    compute_alpha_overall,
    compute_alpha_per_doc,
    compute_alpha_u_overall,
    compute_alpha_u_per_doc,
)
from .f1_scores import compute_f1_for_document, micro_average_f1
from .models import AlphaMode, CorpusData, IAAScores


def evaluate_corpus(
    corpus: CorpusData,
    *,
    alpha_mode: AlphaMode,
    include_background: bool,
    metric: str,
    min_sim: float,
) -> IAAScores:
    """Compute per-document and overall scores for every category."""

    per_doc_f1 = {}
    per_doc_alpha = {}
    pair_counts = {}

    for ref_id in tqdm(sorted(corpus.doc_spans.keys()), desc="Scoring documents"):
        annotations = corpus.doc_spans[ref_id]
        per_doc_f1[ref_id] = compute_f1_for_document(
            annotations,
            corpus.categories,
            metric=metric,
            min_sim=min_sim,
        )

        if alpha_mode is AlphaMode.UNITIZING:
            offsets = corpus.doc_offsets.get(ref_id, {})
            per_doc_alpha[ref_id] = compute_alpha_u_per_doc(
                offsets,
                corpus.categories,
                continuum_len=corpus.doc_lengths.get(ref_id),
                include_background=include_background,
            )
        else:
            per_doc_alpha[ref_id] = compute_alpha_per_doc(
                annotations,
                corpus.categories,
            )

        annotator_count = len(annotations)
        pair_counts[ref_id] = (annotator_count * (annotator_count - 1)) // 2

    if alpha_mode is AlphaMode.UNITIZING:
        overall_alpha = compute_alpha_u_overall(
            corpus.doc_offsets,
            corpus.categories,
            include_background=include_background,
            doc_lengths=corpus.doc_lengths,
        )
    else:
        overall_alpha = compute_alpha_overall(
            corpus.doc_spans,
            corpus.categories,
        )

    overall_f1 = micro_average_f1(per_doc_f1, pair_counts, corpus.categories)

    return IAAScores(
        per_doc_f1=per_doc_f1,
        per_doc_alpha=per_doc_alpha,
        overall_f1=overall_f1,
        overall_alpha=overall_alpha,
        doc_pair_counts=pair_counts,
    )
