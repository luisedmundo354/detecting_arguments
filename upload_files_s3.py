import argparse, os, sys
import boto3
from botocore.exceptions import ClientError

s3_client = boto3.client("s3")

def yes_no(question: str) -> bool:
    """Reusable interactive prompt that only accepts y/n."""
    while True:
        reply = input(f"{question} [y/N]: ").strip().lower()
        if reply in {"y", "yes"}:
            return True
        if reply in {"", "n", "no"}:
            return False
        print("Please type y or n…")

# ---------- bucket / prefix helpers -----------------------------------------

def bucket_exists(bucket: str) -> bool:
    """HEAD-request the bucket; returns False on 404 / 403."""
    try:
        s3_client.head_bucket(Bucket=bucket)          # 200 OK ⇒ exists
        return True
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "403", "400", "NoSuchBucket"):
            return False
        raise                                           # something else

def prefix_exists(bucket: str, prefix: str) -> bool:
    """Does *any* object already start with this prefix?"""
    resp = s3_client.list_objects_v2(
        Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return "Contents" in resp

def create_prefix_object(bucket: str, prefix: str) -> None:
    """Create the zero-byte object that makes S3 show a folder."""
    key = prefix if prefix.endswith("/") else prefix + "/"
    s3_client.put_object(Bucket=bucket, Key=key, Body=b"")
    print(f"✔ Created folder s3://{bucket}/{key}")

# ---------- your original upload helpers ------------------------------------

def upload_file(file_path: str, bucket: str, object_name: str | None = None) -> bool:
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
    """Upload every file in *folder* that is **not yet** in the bucket (flat)."""
    prefix = os.path.basename(folder.rstrip("/")) + "/"
    if not prefix_exists(bucket, prefix):
        if yes_no(f"Folder “{prefix}” does not exist in {bucket}. Create it?"):
            create_prefix_object(bucket, prefix)
        else:
            sys.exit("Aborted by user.")
    existing = {
        obj["Key"].split("/")[-1]                       # just the filename
        for obj in s3_client.list_objects_v2(
            Bucket=bucket, Prefix=prefix).get("Contents", [])
    }
    for entry in os.listdir(folder):
        path = os.path.join(folder, entry)
        if os.path.isfile(path):
            if entry in existing:
                print(f"• Skipping {entry} (already in bucket)")
            else:
                upload_file(path, bucket, prefix + entry)

# ---------- CLI -------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Upload a file or sync a one-level folder to S3.")
    p.add_argument("path", help="File or folder to upload")
    p.add_argument("bucket", help="Destination S3 bucket name")
    args = p.parse_args()

    if not os.path.exists(args.path):
        sys.exit(f"Path does not exist: {args.path}")

    if not bucket_exists(args.bucket):
        if yes_no(f"Bucket “{args.bucket}” does not exist. Create it?"):
            s3_client.create_bucket(Bucket=args.bucket)
            print(f"✔ Bucket {args.bucket} created")
        else:
            sys.exit("Aborted by user.")

    if os.path.isdir(args.path):
        upload_folder(args.path, args.bucket)
    else:
        upload_file(args.path, args.bucket)

if __name__ == "__main__":
    main()

