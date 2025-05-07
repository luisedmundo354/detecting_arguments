"""
Usage examples
--------------
# upload or update a single file
python upload_to_s3.py  path/to/file.txt   my-bucket

# add only the missing files inside a folder (non-recursive)
python upload_to_s3.py  path/to/folder     my-bucket
"""
import argparse
import os
import sys
import boto3
from botocore.exceptions import ClientError

s3_client = boto3.client("s3")


def list_bucket_keys(bucket: str) -> set[str]:
    """Return a **set** with every object key currently in *bucket*."""
    paginator = s3_client.get_paginator("list_objects_v2")
    keys: set[str] = set()
    for page in paginator.paginate(Bucket=bucket):
        keys.update(obj["Key"] for obj in page.get("Contents", []))
    return keys


def upload_file(file_path: str, bucket: str, object_name: str | None = None) -> bool:
    """Upload *file_path* to S3 (overwrite allowed)."""
    if object_name is None:
        object_name = os.path.basename(file_path)
    try:
        s3_client.upload_file(file_path, bucket, object_name)
        print(f"✔ Uploaded {file_path} → s3://{bucket}/{object_name}")
        return True
    except ClientError as e:
        print(f"✗ Failed to upload {file_path}: {e}")
        return False


def upload_folder(folder: str, bucket: str) -> None:
    """Add every file in *folder* that is **not yet** in *bucket* (non-recursive)."""
    existing = list_bucket_keys(bucket)
    for entry in os.listdir(folder):
        path = os.path.join(folder, entry)
        if os.path.isfile(path):
            if entry in existing:
                print(f"• Skipping {entry} (already in bucket)")
            else:
                upload_file(path, bucket, entry)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add or update a file, or add missing files from a folder, in an S3 bucket."
    )
    parser.add_argument("path", help="File or folder to upload")
    parser.add_argument("bucket", help="Destination S3 bucket name")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        sys.exit(f"Path does not exist: {args.path}")

    if os.path.isdir(args.path):
        upload_folder(args.path, args.bucket)
    else:
        upload_file(args.path, args.bucket)


if __name__ == "__main__":
    main()
