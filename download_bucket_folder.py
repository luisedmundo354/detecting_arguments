#!/usr/bin/env python3
"""Download every object under an S3 prefix to a local directory."""

import argparse
import os
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from download_bucket_file import download_from_s3, parse_s3_path


def iter_s3_objects(bucket: str, prefix: str):
    """Yield S3 object keys under the given prefix."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj.get("Key")
                if not key:
                    continue
                if key.endswith("/"):
                    continue
                yield key
    except (BotoCoreError, ClientError) as err:
        print(
            f"Error listing objects for s3://{bucket}/{prefix}: {err}",
            file=sys.stderr,
        )
        sys.exit(1)


def normalize_prefix(prefix: str) -> str:
    prefix = prefix.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix = f"{prefix}/"
    return prefix


def ensure_destination(dest: str, parser: argparse.ArgumentParser):
    if os.path.exists(dest) and not os.path.isdir(dest):
        parser.error(f"Destination '{dest}' exists and is not a directory.")
    try:
        os.makedirs(dest, exist_ok=True)
    except OSError as err:
        parser.error(f"Unable to create destination '{dest}': {err}")


def main():
    parser = argparse.ArgumentParser(
        description="Download all objects from an S3 folder into a local directory.",
    )
    parser.add_argument(
        "s3_folder",
        nargs="?",
        default="s3://legal-cases-bucket240797/target/",
        help=(
            "S3 folder to download (bucket/prefix). Defaults to "
            "'s3://legal-cases-bucket240797/target/'."
        ),
    )
    parser.add_argument(
        "-d",
        "--dest",
        default="annotations",
        help="Local directory to store downloads (default: annotations).",
    )
    args = parser.parse_args()

    try:
        bucket, prefix = parse_s3_path(args.s3_folder)
    except ValueError as err:
        parser.error(str(err))

    prefix = normalize_prefix(prefix)
    dest_dir = args.dest

    ensure_destination(dest_dir, parser)

    display_prefix = prefix or ""
    print(f"Fetching object list from s3://{bucket}/{display_prefix} …")

    count = 0
    for key in iter_s3_objects(bucket, prefix):
        count += 1
        download_from_s3(bucket, key, dest_dir)

    if count == 0:
        print(f"No objects found under s3://{bucket}/{display_prefix}.")
    else:
        plural = "s" if count != 1 else ""
        print(f"Downloaded {count} object{plural} to '{dest_dir}'.")


if __name__ == "__main__":
    main()
