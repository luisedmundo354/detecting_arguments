"""Package scaffold for SOCES inter-annotator agreement utilities."""

from pathlib import Path

DEFAULT_ANNOTATION_DIR = Path(__file__).resolve().parent.parent / "annotations"

__all__ = ["DEFAULT_ANNOTATION_DIR"]
