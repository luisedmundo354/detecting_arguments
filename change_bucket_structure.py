import boto3
from botocore.exceptions import ClientError

def setup_folders_and_move_root_items(bucket_name, region=None):
    """
    Creates two “folders” (source/ and target/) in the given S3 bucket
    then moves every object at the root level into source/.
    """
    # Use the client for listing and the resource for convenience methods
    s3_client = boto3.client('s3', region_name=region)
    s3 = boto3.resource('s3', region_name=region)
    bucket = s3.Bucket(bucket_name)

    # 1) Create the two folder placeholders
    for prefix in ('source/', 'target/'):
        try:
            bucket.put_object(Key=prefix)
            print(f"Created folder {prefix}")
        except ClientError as e:
            print(f"Error creating folder {prefix}: {e}")

    # 2) List *all* objects in the bucket root (no prefix)
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Delimiter='/', Prefix='')

    moved = 0
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            # skip our folder placeholders and anything already under source/ or target/
            if key.endswith('/') or key.startswith('source/') or key.startswith('target/'):
                continue

            new_key = f"source/{key}"
            copy_source = {'Bucket': bucket_name, 'Key': key}

            # 3) Copy the object into source/
            try:
                bucket.copy(copy_source, new_key)
                # 4) Delete the original
                s3.Object(bucket_name, key).delete()
                print(f"Moved {key} → {new_key}")
                moved += 1
            except ClientError as e:
                print(f"Failed to move {key}: {e}")

    print(f"\nDone. Moved {moved} object(s) into 'source/'.")

if __name__ == '__main__':
    # Replace with your actual bucket name and, if needed, region
    setup_folders_and_move_root_items('legal-cases-bucket240797', region='us-east-1')
