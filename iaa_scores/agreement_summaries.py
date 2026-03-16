"""Simple descriptive summaries for IAA result artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Mapping


def summarize_explicit_spans_from_pair_cache(
    pair_cache: Mapping[str, object],
) -> Dict[str, object]:
    """Summarize explicit span counts by label from a persisted pair cache."""

    overall_by_annotator: Dict[str, Counter[str]] = defaultdict(Counter)
    per_doc: Dict[object, Dict[str, Dict[str, int]]] = {}

    for document_record in pair_cache.get("documents", []):
        ref_id = document_record.get("ref_id")
        annotators = document_record.get("annotators")
        counts_by_annotator = document_record.get("span_counts_by_annotator")
        if not isinstance(annotators, list) or len(annotators) != 2:
            raise ValueError("Pair cache document record must contain exactly two annotators.")
        if not isinstance(counts_by_annotator, dict):
            raise ValueError("Pair cache document record is missing span_counts_by_annotator.")

        per_doc[ref_id] = {}
        for annotator in annotators:
            label_counts = counts_by_annotator.get(annotator)
            if not isinstance(label_counts, dict):
                raise ValueError(
                    f"Pair cache document record has malformed span counts for annotator {annotator!r}."
                )
            normalized_counts = {str(label): int(count) for label, count in label_counts.items()}
            per_doc[ref_id][annotator] = dict(sorted(normalized_counts.items()))
            overall_by_annotator[annotator].update(normalized_counts)

    overall_combined = Counter()
    for annotator_counts in overall_by_annotator.values():
        overall_combined.update(annotator_counts)

    return {
        "overall_by_annotator": {
            annotator: dict(sorted(label_counts.items()))
            for annotator, label_counts in sorted(overall_by_annotator.items())
        },
        "overall_combined": dict(sorted(overall_combined.items())),
        "per_doc": per_doc,
    }
