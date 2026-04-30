from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from iaa_scores.cohere_api import load_repo_cohere_api_key
from iaa_scores.semantic_cache import (
    build_or_load_semantic_embedding_cache,
    load_semantic_embedding_cache,
)


def _write_export(
    root: Path,
    filename: str,
    *,
    export_id: int,
    ref_id: int,
    annotator: str,
    case_text: str,
    results: list[dict],
) -> None:
    payload = {
        "id": export_id,
        "completed_by": {"email": annotator},
        "task": {
            "id": export_id,
            "data": {
                "assigned_to": annotator,
                "case_content": case_text,
                "ref_id": ref_id,
            },
        },
        "result": results,
    }
    (root / filename).write_text(json.dumps(payload), encoding="utf-8")


def _span(node_id: str, label: str, text: str, *, start: int, end: int) -> dict:
    return {
        "id": node_id,
        "type": "labels",
        "from_name": "label",
        "to_name": "text",
        "origin": "manual",
        "value": {
            "start": start,
            "end": end,
            "text": text,
            "labels": [label],
        },
    }


def _span_for_substring(case_text: str, node_id: str, label: str, substring: str) -> dict:
    start = case_text.index(substring)
    end = start + len(substring)
    return _span(node_id, label, substring, start=start, end=end)


class SemanticCacheTests(unittest.TestCase):
    def test_load_repo_cohere_api_key_uses_only_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / ".env").write_text("COHERE_API_KEY=test-key\n", encoding="utf-8")
            previous = os.environ.get("COHERE_API_KEY")
            os.environ["COHERE_API_KEY"] = "wrong-key"
            try:
                self.assertEqual(load_repo_cohere_api_key(repo_root), "test-key")
            finally:
                if previous is None:
                    os.environ.pop("COHERE_API_KEY", None)
                else:
                    os.environ["COHERE_API_KEY"] = previous

    def test_load_repo_cohere_api_key_fails_when_missing_from_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / ".env").write_text("OTHER_KEY=abc\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "COHERE_API_KEY is missing or empty"):
                load_repo_cohere_api_key(repo_root)

    def test_semantic_embedding_cache_roundtrip_and_fingerprint_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            input_dir = repo_root / "annotations" / "final_annotations_iaa_set"
            input_dir.mkdir(parents=True, exist_ok=True)
            case_text = (
                "Alpha establishes the governing rule. "
                "Beta applies that rule to the record."
            )
            results = [
                _span_for_substring(
                    case_text,
                    "r1",
                    "Rule",
                    "Alpha establishes the governing rule.",
                ),
                _span_for_substring(
                    case_text,
                    "a1",
                    "Analysis",
                    "Beta applies that rule to the record.",
                ),
            ]
            _write_export(
                input_dir,
                "101_a.json",
                export_id=101,
                ref_id=1,
                annotator="a@example.com",
                case_text=case_text,
                results=results,
            )
            _write_export(
                input_dir,
                "102_b.json",
                export_id=102,
                ref_id=1,
                annotator="b@example.com",
                case_text=case_text,
                results=results,
            )

            def fake_embedder(texts, **kwargs):
                return [[float(index + 1)] * 4 for index, _ in enumerate(texts)]

            cache = build_or_load_semantic_embedding_cache(
                repo_root=repo_root,
                input_dir=input_dir,
                overwrite=True,
                output_dimension=4,
                embedder=fake_embedder,
            )
            self.assertEqual(cache["embeddings"].shape, (2, 4))
            self.assertIn(
                "Alpha establishes the governing rule.",
                cache["text_to_embedding"],
            )

            loaded = load_semantic_embedding_cache(
                repo_root=repo_root,
                input_dir=input_dir,
                output_dimension=4,
            )
            self.assertEqual(loaded["embeddings"].shape, (2, 4))

            modified_case_text = (
                "Alpha establishes the governing rule. "
                "Beta applies that rule to a changed record."
            )
            _write_export(
                input_dir,
                "102_b.json",
                export_id=102,
                ref_id=1,
                annotator="b@example.com",
                case_text=modified_case_text,
                results=[
                    _span_for_substring(
                        modified_case_text,
                        "r1",
                        "Rule",
                        "Alpha establishes the governing rule.",
                    ),
                    _span_for_substring(
                        modified_case_text,
                        "a1",
                        "Analysis",
                        "Beta applies that rule to a changed record.",
                    ),
                ],
            )

            with self.assertRaisesRegex(ValueError, "fingerprint does not match"):
                load_semantic_embedding_cache(
                    repo_root=repo_root,
                    input_dir=input_dir,
                    output_dimension=4,
                )


if __name__ == "__main__":
    unittest.main()
