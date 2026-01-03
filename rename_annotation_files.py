#!/usr/bin/env python3
"""Rename annotation files to include annotator email prefix."""

import argparse
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rename every JSON annotation file in a folder to '<id>_<emailprefix>'."
        )
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="annotations",
        help="Folder containing annotation files (default: annotations).",
    )
    return parser.parse_args()


def ensure_folder(path: str):
    if not os.path.exists(path):
        print(f"Folder '{path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(path):
        print(f"Path '{path}' is not a folder.", file=sys.stderr)
        sys.exit(1)


def load_annotation(path: str):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as err:
        print(f"Skipping '{path}': invalid JSON ({err}).", file=sys.stderr)
    except OSError as err:
        print(f"Skipping '{path}': error reading file ({err}).", file=sys.stderr)
    return None


def email_prefix(email: str) -> str:
    return email.split("@", 1)[0]


def build_new_name(data, original_name: str):
    file_id = data.get("id")
    if file_id is None:
        print(
            f"Skipping '{original_name}': missing 'id' field.",
            file=sys.stderr,
        )
        return None

    completed_by = data.get("completed_by") or {}
    email = completed_by.get("email")
    if not email:
        print(
            f"Skipping '{original_name}': missing annotator email.",
            file=sys.stderr,
        )
        return None

    prefix = email_prefix(email)
    if not prefix:
        print(
            f"Skipping '{original_name}': annotator email '{email}' has empty prefix.",
            file=sys.stderr,
        )
        return None

    base, ext = os.path.splitext(original_name)
    return f"{file_id}_{prefix}{ext}.json"


def rename_files(folder: str):
    entries = os.listdir(folder)
    if not entries:
        print(f"Folder '{folder}' is empty.")
        return

    renamed = 0
    skipped = 0

    for name in sorted(entries):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue

        data = load_annotation(path)
        if data is None:
            skipped += 1
            continue

        new_name = build_new_name(data, name)
        if not new_name:
            skipped += 1
            continue

        if name == new_name:
            continue

        target_path = os.path.join(folder, new_name)
        if os.path.exists(target_path):
            print(
                f"Skipping '{name}': '{new_name}' already exists.",
                file=sys.stderr,
            )
            skipped += 1
            continue

        try:
            os.rename(path, target_path)
            renamed += 1
            print(f"Renamed '{name}' → '{new_name}'.")
        except OSError as err:
            print(
                f"Failed to rename '{name}' to '{new_name}': {err}.",
                file=sys.stderr,
            )
            skipped += 1

    if renamed == 0:
        print("No files renamed.")
    else:
        suffix = "s" if renamed != 1 else ""
        print(f"Renamed {renamed} file{suffix} in '{folder}'.")

    if skipped:
        suffix = "s" if skipped != 1 else ""
        print(f"Skipped {skipped} file{suffix} due to errors.", file=sys.stderr)


def main():
    args = parse_args()
    ensure_folder(args.folder)
    rename_files(args.folder)


if __name__ == "__main__":
    main()
