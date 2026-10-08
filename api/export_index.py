"""
Packages the production search index into api/bundle/, which the Docker image
ships as-is (the index is read-only between rebuilds, so it's versioned with
the deploy instead of living in a database; see api/README.md).

    bundle/photo_vectors.npy   SigLIP 2 photo embeddings   (float16, n x 768)
    bundle/title_vectors.npy   MiniLM title embeddings     (float16, n x 384)
    bundle/items.parquet       one row per vector: metadata + color / category
    bundle/manifest.json       models, counts, fusion weight, build info

Usage (after image_search/build_index.py and extract_attributes.py):
    python api/export_index.py
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "image_search"))

from attributes import PALETTE, load_attributes  # noqa: E402
from encoders import MODELS  # noqa: E402
from fusion import PHOTO_WEIGHT  # noqa: E402
from settings import GALLERY_PARQUET, emb_path  # noqa: E402

BUNDLE = Path(__file__).resolve().parent / "bundle"
PHOTO_MODEL, TITLE_MODEL = "siglip2-b16", "minilm"


def clean_price(p) -> float | None:
    return None if p is None or (isinstance(p, float) and math.isnan(p)) or p <= 0 or p > 100_000 else float(p)


def main() -> None:
    gallery = pd.read_parquet(GALLERY_PARQUET)
    attrs = load_attributes(gallery)
    photo = np.load(emb_path(PHOTO_MODEL, "image"))
    title = np.load(emb_path(TITLE_MODEL, "title"))
    assert len(photo) == len(title) == len(gallery), "embeddings and gallery are out of sync"

    items = pd.DataFrame({
        "product_id": gallery["product_id"],
        "title": gallery["title"],
        "merchant": gallery["merchant"],
        "price": gallery["price"].map(clean_price),
        "url": gallery["url"],
        "retailer_image_url": gallery["image_url"],
        "photo_key": gallery["local_image"],     # object key in the photo bucket
        "color": attrs["color"],
        "color_name": attrs["color_name"],
        "category": attrs["category"],
    })

    BUNDLE.mkdir(exist_ok=True)
    np.save(BUNDLE / "photo_vectors.npy", photo.astype(np.float16))
    np.save(BUNDLE / "title_vectors.npy", title.astype(np.float16))
    items.to_parquet(BUNDLE / "items.parquet", index=False)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=ROOT).stdout.strip()
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "items": len(items),
        "with_attributes": int((items["color"] != "").sum()),
        "photo_model": MODELS[PHOTO_MODEL][0],
        "title_model": MODELS[TITLE_MODEL][0],
        "photo_weight": PHOTO_WEIGHT,
        "palette": list(PALETTE),
    }
    (BUNDLE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    size = sum(f.stat().st_size for f in BUNDLE.iterdir()) / 1e6
    print(json.dumps(manifest, indent=2))
    print(f"bundle: {size:.0f} MB -> {BUNDLE}")


if __name__ == "__main__":
    main()
