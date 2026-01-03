from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


IIC_PREFIX = "Implicit Intermediate Conclusion"
IIC_RE = re.compile(r"^Implicit Intermediate Conclusion\b")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")
    tmp_path.replace(path)


def _iter_json_files(dir_path: Path) -> Iterable[Path]:
    for path in sorted(dir_path.glob("*.json")):
        if path.is_file():
            yield path


def _is_iic_span_text(text: str) -> bool:
    return bool(IIC_RE.match(text.strip()))


def _normalize_span_text(text: str) -> tuple[str, bool]:
    """
    Some exports contain literal escape sequences like ``\\n`` inside value.text instead of real newlines.

    For matching and validation, normalize those to their intended characters.
    """
    normalized = text.replace("\\r\\n", "\r\n").replace("\\n", "\n").replace("\\t", "\t")
    return normalized, normalized != text


def _find_all_occurrences(haystack: str, needle: str) -> list[int]:
    if not needle:
        return []
    starts: list[int] = []
    idx = haystack.find(needle)
    while idx != -1:
        starts.append(idx)
        idx = haystack.find(needle, idx + 1)
    return starts


def _build_boundary_map(src: str, dst: str) -> list[int]:
    """
    Build a monotonic mapping from boundary indices in `src` to boundary indices in `dst`.

    The mapping is sized `len(src)+1` where each entry is a boundary index in `dst` (0..len(dst)).
    """
    matcher = SequenceMatcher(None, src, dst, autojunk=False)
    boundary_map: list[int | None] = [None] * (len(src) + 1)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        src_len = i2 - i1
        dst_len = j2 - j1

        if tag == "equal" or (tag == "replace" and src_len == dst_len):
            # For equal (and same-length replace), map every boundary in the segment 1:1.
            for k in range(0, src_len + 1):
                boundary_map[i1 + k] = j1 + k
            continue

        # Otherwise, at least map the segment boundaries.
        boundary_map[i1] = j1
        boundary_map[i2] = j2

    # Fill any gaps by carrying forward the last known mapping. This preserves monotonicity and
    # is sufficient for spans whose boundaries are in aligned regions.
    out: list[int] = [0] * (len(src) + 1)
    last = 0
    for i, v in enumerate(boundary_map):
        if v is None:
            out[i] = last
        else:
            out[i] = v
            last = v
    return out


@dataclass(frozen=True)
class FixStats:
    files_total: int = 0
    files_written: int = 0
    spans_total: int = 0
    spans_iic_nulled: int = 0
    spans_text_normalized: int = 0
    spans_text_overwritten: int = 0
    spans_unmatched_nulled: int = 0
    spans_remapped: int = 0
    spans_verified_ok: int = 0
    spans_failed: int = 0


def _load_clean_text_by_ref_id(clean_tasks_dir: Path) -> dict[int, str]:
    clean_by_ref: dict[int, str] = {}
    for path in _iter_json_files(clean_tasks_dir):
        task = _load_json(path)
        if not isinstance(task, dict) or "data" not in task:
            raise ValueError(f"Unexpected task format in {path}")
        data = task["data"]
        if not isinstance(data, dict) or "ref_id" not in data or "case_content" not in data:
            raise ValueError(f"Missing data.ref_id or data.case_content in {path}")
        ref_id = int(data["ref_id"])
        case_content = data["case_content"]
        if not isinstance(case_content, str):
            raise ValueError(f"data.case_content is not a string in {path}")
        if ref_id in clean_by_ref:
            raise ValueError(f"Duplicate ref_id {ref_id} in clean tasks: {path}")
        clean_by_ref[ref_id] = case_content
    return clean_by_ref


def _remap_span(
    *,
    text: str,
    src_start: int,
    src_end: int,
    boundary_map: list[int],
    dst_text: str,
) -> tuple[int, int] | None:
    if not (0 <= src_start <= src_end <= len(boundary_map) - 1):
        return None

    mapped_start = boundary_map[src_start]
    mapped_end = boundary_map[src_end]
    if 0 <= mapped_start <= mapped_end <= len(dst_text) and dst_text[mapped_start:mapped_end] == text:
        return mapped_start, mapped_end

    # Fallback: exact search in destination text (prefer a match close to the mapped start).
    candidates = _find_all_occurrences(dst_text, text)
    if not candidates:
        return None
    best = min(candidates, key=lambda idx: abs(idx - mapped_start))
    return best, best + len(text)


def fix_annotations(
    *,
    annotations_dir: Path,
    clean_tasks_dir: Path,
    backup_dir: Path | None,
    no_backup: bool,
    dry_run: bool,
) -> FixStats:
    clean_by_ref_id = _load_clean_text_by_ref_id(clean_tasks_dir)

    annotation_paths = list(_iter_json_files(annotations_dir))
    stats = FixStats(files_total=len(annotation_paths))

    if not dry_run:
        if backup_dir is not None and no_backup:
            raise ValueError("backup_dir and no_backup are mutually exclusive")
        if not no_backup:
            if backup_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dir = annotations_dir.with_name(f"{annotations_dir.name}_backup_{timestamp}")
            backup_dir.mkdir(parents=True, exist_ok=False)
            for path in annotation_paths:
                shutil.copy2(path, backup_dir / path.name)

    files_written = 0
    spans_total = 0
    spans_iic_nulled = 0
    spans_text_normalized = 0
    spans_text_overwritten = 0
    spans_unmatched_nulled = 0
    spans_remapped = 0
    spans_verified_ok = 0
    spans_failed = 0

    for path in annotation_paths:
        ann = _load_json(path)
        if not isinstance(ann, dict):
            raise ValueError(f"Unexpected annotation format in {path}")

        task = ann.get("task")
        if not isinstance(task, dict):
            raise ValueError(f"Missing/invalid task in {path}")
        data = task.get("data")
        if not isinstance(data, dict):
            raise ValueError(f"Missing/invalid task.data in {path}")

        ref_id_raw = data.get("ref_id")
        if ref_id_raw is None:
            raise ValueError(f"Missing task.data.ref_id in {path}")
        ref_id = int(ref_id_raw)
        if ref_id not in clean_by_ref_id:
            raise ValueError(f"ref_id {ref_id} not found in {clean_tasks_dir} (needed by {path})")

        clean_text = clean_by_ref_id[ref_id]
        src_text = data.get("case_content")
        if not isinstance(src_text, str):
            raise ValueError(f"Missing/invalid task.data.case_content in {path}")

        boundary_map = _build_boundary_map(src_text, clean_text)

        # Normalize the embedded task text to the canonical clean version so offsets are consistent.
        data["case_content"] = clean_text

        results = ann.get("result")
        if not isinstance(results, list):
            raise ValueError(f"Missing/invalid result list in {path}")

        changed = False
        for r in results:
            if not isinstance(r, dict) or r.get("type") != "labels":
                continue
            value = r.get("value")
            if not isinstance(value, dict):
                continue

            if "start" not in value or "end" not in value or "text" not in value:
                continue

            spans_total += 1

            span_text = value.get("text")
            if not isinstance(span_text, str):
                spans_failed += 1
                continue

            if _is_iic_span_text(span_text):
                if value.get("start") is not None or value.get("end") is not None:
                    value["start"] = None
                    value["end"] = None
                    changed = True
                spans_iic_nulled += 1
                continue

            normalized_text, did_normalize = _normalize_span_text(span_text)
            if did_normalize:
                value["text"] = normalized_text
                span_text = normalized_text
                spans_text_normalized += 1
                changed = True

            src_start = value.get("start")
            src_end = value.get("end")
            if src_start is None and src_end is None:
                spans_unmatched_nulled += 1
                continue
            if not isinstance(src_start, int) or not isinstance(src_end, int):
                spans_failed += 1
                continue

            mapped = _remap_span(
                text=span_text,
                src_start=src_start,
                src_end=src_end,
                boundary_map=boundary_map,
                dst_text=clean_text,
            )
            if mapped is None:
                # If we can’t locate the span text exactly (rare), still preserve an aligned region by using
                # the boundary mapping, and overwrite the span text to match the canonical text.
                mapped_start = boundary_map[src_start]
                mapped_end = boundary_map[src_end]
                if mapped_end > mapped_start:
                    value["start"] = mapped_start
                    value["end"] = mapped_end
                    value["text"] = clean_text[mapped_start:mapped_end]
                    spans_text_overwritten += 1
                    spans_remapped += 1
                    spans_verified_ok += 1
                    changed = True
                else:
                    # Degenerate/unrecoverable span (e.g., start==end with non-empty text). Keep it for
                    # relations/inspection, but make it non-span-like.
                    value["start"] = None
                    value["end"] = None
                    spans_unmatched_nulled += 1
                    changed = True
                continue

            new_start, new_end = mapped
            if new_start != src_start or new_end != src_end:
                value["start"] = new_start
                value["end"] = new_end
                changed = True
            spans_remapped += 1

            if clean_text[new_start:new_end] == span_text:
                spans_verified_ok += 1
            else:
                # Keep offsets authoritative and make the stored text consistent with the canonical source.
                value["text"] = clean_text[new_start:new_end]
                spans_text_overwritten += 1
                spans_verified_ok += 1
                changed = True

        if changed and not dry_run:
            _atomic_write_json(path, ann)
            files_written += 1

    return FixStats(
        files_total=stats.files_total,
        files_written=files_written,
        spans_total=spans_total,
        spans_iic_nulled=spans_iic_nulled,
        spans_text_normalized=spans_text_normalized,
        spans_text_overwritten=spans_text_overwritten,
        spans_unmatched_nulled=spans_unmatched_nulled,
        spans_remapped=spans_remapped,
        spans_verified_ok=spans_verified_ok,
        spans_failed=spans_failed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fix span offsets in annotations/final_annotations_iaa_set using clean tasks in label_studio_taks/overlapping."
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("annotations/final_annotations_iaa_set"),
        help="Directory containing Label Studio annotation JSON exports.",
    )
    parser.add_argument(
        "--clean-tasks-dir",
        type=Path,
        default=Path("label_studio_taks/overlapping"),
        help="Directory containing clean Label Studio task JSON (data.ref_id + data.case_content).",
    )
    backup_group = parser.add_mutually_exclusive_group()
    backup_group.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Where to copy the original annotation files before editing (defaults to a timestamped sibling folder).",
    )
    backup_group.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a backup copy before editing in-place.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; only compute and validate remapping.",
    )

    args = parser.parse_args(argv)

    stats = fix_annotations(
        annotations_dir=args.annotations_dir,
        clean_tasks_dir=args.clean_tasks_dir,
        backup_dir=args.backup_dir,
        no_backup=args.no_backup,
        dry_run=args.dry_run,
    )

    print(json.dumps(stats.__dict__, indent=2, sort_keys=True))

    if stats.spans_failed > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
