"""
Uploads the photo of every item in api/bundle/ to a Tigris bucket (Fly's
S3-compatible object storage), so the API can return reliable image URLs
instead of retailer links. Skips photos already in the bucket.

Credentials come from `fly storage create`, which prints them:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME,
    AWS_ENDPOINT_URL_S3 (https://fly.storage.tigris.dev)

Usage:
    python api/upload_photos.py
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "product_images" / "product_images"
BUNDLE = Path(__file__).resolve().parent / "bundle"


def main() -> None:
    missing = [v for v in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "BUCKET_NAME") if not os.environ.get(v)]
    if missing:
        sys.exit(f"Set {', '.join(missing)} (printed by `fly storage create`).")
    bucket = os.environ["BUCKET_NAME"]
    s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3", "https://fly.storage.tigris.dev"))

    keys = pd.read_parquet(BUNDLE / "items.parquet", columns=["photo_key"])["photo_key"]
    keys = sorted(set(keys[keys != ""]))
    existing = {o["Key"] for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket)
                for o in page.get("Contents", [])}
    todo = [k for k in keys if k not in existing]
    print(f"{len(keys):,} photos, {len(existing):,} already in {bucket}, {len(todo):,} to upload")

    def put(key: str) -> None:
        s3.upload_file(str(IMAGES_DIR / key), bucket, key, ExtraArgs={
            "ContentType": "image/jpeg", "CacheControl": "public, max-age=604800"})

    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, _ in enumerate(pool.map(put, todo), 1):
            if i % 1000 == 0 or i == len(todo):
                print(f"  {i:,}/{len(todo):,}", flush=True)
    print(f"done. Set PHOTO_BASE_URL=https://{bucket}.fly.storage.tigris.dev in fly.toml")


if __name__ == "__main__":
    main()
