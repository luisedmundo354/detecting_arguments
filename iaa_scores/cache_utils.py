"""Shared helpers for strict IAA cache management."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def compute_dataset_fingerprint(input_dir: Path) -> str:
    """Return a stable content hash for a directory of JSON exports."""

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

    hasher = hashlib.sha256()
    files = sorted(path for path in input_dir.glob("*.json") if path.is_file())
    if not files:
        raise ValueError(f"No JSON files found in {input_dir}")

    for path in files:
        hasher.update(path.name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def dump_json(data: Any, path: Path) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def float_slug(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return text.replace("-", "neg").replace(".", "p")


def cache_root(repo_root: Path) -> Path:
    return repo_root / "iaa_scores" / "cache"


def repo_relative_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def validate_required_keys(mapping: dict[str, Any], required: Iterable[str], *, context: str) -> None:
    missing = [key for key in required if key not in mapping]
    if missing:
        raise ValueError(f"Missing required keys in {context}: {missing}")
