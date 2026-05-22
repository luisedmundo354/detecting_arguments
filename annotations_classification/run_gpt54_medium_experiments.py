"""Run GPT-5.4 medium-reasoning classification experiments.

This script reuses the existing classification pipeline and only overrides the
GPT model/reasoning settings for this experiment.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from gpt5_classifier import GPT5Config
from main import run_linear_svc_classification
from reporting import export_run_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT_ROOT / "annotations_classification"

ANNOTATION_DIR = PROJECT_ROOT / "annotations" / "final_annotations_gold"
RESULTS_ROOT = MODULE_DIR / "results"
CHECKPOINT_ROOT = MODULE_DIR / "checkpoints"

MODEL = "gpt-5.4"
REASONING_EFFORT = "medium"

EMBEDDINGS = ("tfidf", "sbert", "legal-bert", "modern-bert")
USE_MODERN_BERT = False
INCLUDE_BASELINES = True
GPT5_WORKERS = 10

N_SPLITS = 5
RANDOM_STATE = 42
BATCH_SIZE = 16
FILTER_IMPLICIT_CONCLUSIONS = True

RUNS = (
    {
        "name": "gold_5_class_gpt54_medium_grouped_cv",
        "checkpoint": CHECKPOINT_ROOT / "gpt54_medium_predictions_gold_5_classes.jsonl",
        "combine_analysis_conclusion": False,
    },
    {
        "name": "gold_4_class_gpt54_medium_grouped_cv",
        "checkpoint": CHECKPOINT_ROOT / "gpt54_medium_predictions_gold_4_classes.jsonl",
        "combine_analysis_conclusion": True,
    },
)


def load_environment() -> None:
    """Load API credentials from dotenv without overriding shell variables."""

    dotenv_paths = (PROJECT_ROOT / ".env", MODULE_DIR / ".env")
    for dotenv_path in dotenv_paths:
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path, override=False)

    if not os.getenv("OPENAI_API_KEY"):
        searched = ", ".join(str(path) for path in dotenv_paths)
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your shell environment or one "
            f"of these dotenv files: {searched}"
        )


def run_experiment(run_spec: Dict[str, Any]) -> Dict[str, str]:
    """Execute one configured classification run and export its artifacts."""

    gpt5_config = GPT5Config(
        model=MODEL,
        reasoning_effort=REASONING_EFFORT,
    )

    results = run_linear_svc_classification(
        ANNOTATION_DIR,
        embeddings=EMBEDDINGS,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        batch_size=BATCH_SIZE,
        include_baselines=INCLUDE_BASELINES,
        use_modern_bert=USE_MODERN_BERT,
        gpt5=True,
        gpt5_config=gpt5_config,
        test_mode=False,
        gpt5_checkpoint_path=run_spec["checkpoint"],
        gpt5_workers=GPT5_WORKERS,
        combine_analysis_conclusion=run_spec["combine_analysis_conclusion"],
        filter_implicit_conclusions=FILTER_IMPLICIT_CONCLUSIONS,
    )

    artifact_paths = export_run_artifacts(
        results=results,
        output_root=RESULTS_ROOT,
        run_name=run_spec["name"],
        overwrite=True,
    )
    artifact_paths["run_metadata_json"] = write_run_metadata(
        artifact_paths=artifact_paths,
        run_spec=run_spec,
        gpt5_config=gpt5_config,
        results=results,
    )
    return artifact_paths


def write_run_metadata(
    *,
    artifact_paths: Dict[str, str],
    run_spec: Dict[str, Any],
    gpt5_config: GPT5Config,
    results: Dict[str, Any],
) -> str:
    """Write explicit GPT settings because checkpoint rows only store predictions."""

    run_dir = Path(artifact_paths["run_dir"])
    metadata_path = run_dir / "gpt54_medium_run_metadata.json"
    metadata = {
        "run_name": run_spec["name"],
        "annotation_dir": str(ANNOTATION_DIR),
        "requested_checkpoint_path": str(run_spec["checkpoint"]),
        "resolved_checkpoint_path": results.get("gpt5", {}).get("checkpoint_path"),
        "combine_analysis_conclusion": run_spec["combine_analysis_conclusion"],
        "filter_implicit_conclusions": FILTER_IMPLICIT_CONCLUSIONS,
        "gpt5_config": asdict(gpt5_config),
        "gpt5_workers": GPT5_WORKERS,
        "embeddings": list(EMBEDDINGS),
        "use_modern_bert": USE_MODERN_BERT,
        "include_baselines": INCLUDE_BASELINES,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "batch_size": BATCH_SIZE,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf8")
    return str(metadata_path)


def main() -> None:
    load_environment()
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    all_artifacts = {}
    for run_spec in RUNS:
        all_artifacts[run_spec["name"]] = run_experiment(run_spec)

    print(json.dumps(all_artifacts, indent=2))


if __name__ == "__main__":
    main()
