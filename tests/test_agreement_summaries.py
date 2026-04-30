from __future__ import annotations

import unittest

from iaa_scores.agreement_summaries import summarize_explicit_spans_from_pair_cache


class AgreementSummaryTests(unittest.TestCase):
    def test_summarize_explicit_spans_from_pair_cache(self) -> None:
        pair_cache = {
            "documents": [
                {
                    "ref_id": 1,
                    "annotators": ["a@example.com", "b@example.com"],
                    "span_counts_by_annotator": {
                        "a@example.com": {"Analysis": 2, "Rule": 1},
                        "b@example.com": {"Analysis": 1, "Conclusion": 1},
                    },
                },
                {
                    "ref_id": 2,
                    "annotators": ["a@example.com", "b@example.com"],
                    "span_counts_by_annotator": {
                        "a@example.com": {"Rule": 3},
                        "b@example.com": {"Analysis": 2, "Rule": 1},
                    },
                },
            ]
        }

        summary = summarize_explicit_spans_from_pair_cache(pair_cache)

        self.assertEqual(
            summary["overall_by_annotator"]["a@example.com"],
            {"Analysis": 2, "Rule": 4},
        )
        self.assertEqual(
            summary["overall_by_annotator"]["b@example.com"],
            {"Analysis": 3, "Conclusion": 1, "Rule": 1},
        )
        self.assertEqual(
            summary["overall_combined"],
            {"Analysis": 5, "Conclusion": 1, "Rule": 5},
        )
        self.assertEqual(
            summary["per_doc"][1]["a@example.com"],
            {"Analysis": 2, "Rule": 1},
        )


if __name__ == "__main__":
    unittest.main()
