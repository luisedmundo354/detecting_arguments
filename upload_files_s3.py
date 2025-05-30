from __future__ import annotations
import argparse, os, sys, boto3
from botocore.exceptions import ClientError

s3_client = boto3.client("s3")

# ─────────── yes/no prompt ──────────────────────────────────────────
def yes_no(question: str) -> bool:
    while True:
        reply = input(f"{question} [y/N]: ").strip().lower()
        if reply in {"y", "yes"}:
            return True
        if reply in {"", "n", "no"}:
            return False
        print("Please type y or n…")

# ─────────── existence helpers ─────────────────────────────────────
def bucket_exists(bucket: str) -> bool:
    try:
        s3_client.head_bucket(Bucket=bucket)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in {"404", "403", "400", "NoSuchBucket"}:
            return False
        raise

def prefix_exists(bucket: str, prefix: str) -> bool:
    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return "Contents" in resp

def object_exists(bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in {"404", "403", "400", "NoSuchKey"}:
            return False
        raise

def create_prefix_object(bucket: str, prefix: str) -> None:
    key = prefix if prefix.endswith("/") else prefix + "/"
    s3_client.put_object(Bucket=bucket, Key=key, Body=b"")
    print(f"✔ Created folder s3://{bucket}/{key}")

# ─────────── upload helpers ────────────────────────────────────────
def upload_file(file_path: str, bucket: str, object_prefix: str = "", *,
                force: bool = False, skip: bool = False) -> bool:
    object_name = f"{object_prefix}{os.path.basename(file_path)}"

    if not force and not skip and object_exists(bucket, object_name):
        if not yes_no(f"Object s3://{bucket}/{object_name} exists – replace?"):
            print(f"• Skipped {file_path}")
            return False

    if skip and object_exists(bucket, object_name):
        print(f"• Skipped {file_path} (already in bucket)")
        return False

    try:
        s3_client.upload_file(file_path, bucket, object_name)
        print(f"✔ Uploaded {file_path} → s3://{bucket}/{object_name}")
        return True
    except ClientError as e:
        print(f"✗ Failed to upload {file_path}: {e}")
        return False

def upload_folder(folder: str, bucket: str, base_prefix: str = "", *,
                  force: bool = False, skip: bool = False) -> None:
    prefix = base_prefix
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    if not prefix_exists(bucket, prefix):
        if yes_no(f"Folder “{prefix}” does not exist in {bucket}. Create it?"):
            create_prefix_object(bucket, prefix)
        else:
            sys.exit("Aborted by user.")

    for entry in os.listdir(folder):
        path = os.path.join(folder, entry)
        if os.path.isfile(path):
            upload_file(path, bucket, prefix, force=force, skip=skip)

# ─────────── CLI ───────────────────────────────────────────────────
def parse_bucket_and_prefix(raw: str) -> tuple[str, str]:
    bucket, _, rest = raw.partition("/")
    if not bucket:
        sys.exit("Bucket name missing before slash in argument")
    prefix = (rest.rstrip("/") + "/") if rest else ""
    return bucket, prefix

def main() -> None:
    p = argparse.ArgumentParser(description="Upload a file or sync a folder to S3.")
    p.add_argument("path", help="Local file or folder to upload")
    p.add_argument("bucket", help="Destination S3 bucket (optionally with /prefix)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--force", action="store_true",
                       help="overwrite objects without asking")
    group.add_argument("--skip-existing", action="store_true",
                       help="never overwrite existing objects")
    args = p.parse_args()

    if not os.path.exists(args.path):
        sys.exit(f"Path does not exist: {args.path}")

    bucket, prefix = parse_bucket_and_prefix(args.bucket)

    if not bucket_exists(bucket):
        if yes_no(f"Bucket “{bucket}” does not exist. Create it?"):
            s3_client.create_bucket(Bucket=bucket)
            print(f"✔ Bucket {bucket} created")
        else:
            sys.exit("Aborted by user.")

    if os.path.isdir(args.path):
        upload_folder(args.path, bucket, prefix,
                      force=args.force, skip=args.skip_existing)
    else:
        upload_file(args.path, bucket, prefix,
                    force=args.force, skip=args.skip_existing)

if __name__ == "__main__":
    main()


