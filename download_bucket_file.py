import argparse
import os
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError

def parse_s3_path(s3_path: str):
    """
    Parse an S3 path into bucket and key.
    Accepts formats like:
      - my-bucket/path/to/file.txt
      - s3://my-bucket/path/to/file.txt
    """
    # Strip s3:// if present
    if s3_path.startswith("s3://"):
        s3_path = s3_path[5:]
    # Split into bucket and key
    parts = s3_path.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid S3 path: {s3_path!r}. Must be bucket/key")
    bucket, key = parts
    return bucket, key

def download_from_s3(bucket: str, key: str, destination: str):
    """
    Download an object from S3 to the given local path.
    """
    s3 = boto3.client("s3")
    try:
        # If destination is a directory, keep original filename
        if os.path.isdir(destination):
            filename = os.path.basename(key)
            destination = os.path.join(destination, filename)
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        print(f"Downloading s3://{bucket}/{key} → {destination} …")
        s3.download_file(bucket, key, destination)
        print("Download complete.")
    except (BotoCoreError, ClientError) as err:
        print(f"Error downloading file: {err}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Download a single file from S3 given a full S3 path."
    )
    parser.add_argument(
        "s3_path",
        help="Full S3 path to the object, e.g. "
             "`my-bucket/path/to/file.txt` or `s3://my-bucket/path/to/file.txt`",
    )
    parser.add_argument(
        "-d", "--dest",
        default=".",
        help="Local destination file or directory (default: current directory)",
    )
    args = parser.parse_args()

    try:
        bucket, key = parse_s3_path(args.s3_path)
    except ValueError as e:
        parser.error(str(e))

    download_from_s3(bucket, key, args.dest)

if __name__ == "__main__":
    main()
