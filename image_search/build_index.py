"""
Builds the image-search gallery and embeds it with every model in encoders.MODELS.

The gallery is every catalog product that has a local image, de-duplicated so
the same photo or title doesn't appear under several product ids. For each
model we cache:
    <model>_image.npy   image embeddings  (shared + image-only models)
    <model>_title.npy   title embeddings  (shared + text-only models)

Usage:
    python image_search/build_index.py                       # build what's missing
    python image_search/build_index.py --models clip-b32     # only some models
    python image_search/build_index.py --rebuild             # rebuild everything
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from encoders import MODELS, Encoder, device
from settings import ARTIFACTS_DIR, GALLERY_PARQUET, IMAGES_DIR, SOURCE_CORPUS, emb_path


def _readable(name: str) -> bool:
    try:
        with Image.open(IMAGES_DIR / name) as im:
            im.verify()
        return True
    except Exception:
        return False


def build_gallery() -> pd.DataFrame:
    if not SOURCE_CORPUS.exists():
        raise SystemExit("Missing catalog - run `python search_app/build_index.py` first.")
    corpus = pd.read_parquet(SOURCE_CORPUS)
    g = corpus[corpus["local_image"] != ""]
    n_with_image = len(g)

    g = g.drop_duplicates("local_image")
    g = g[~g["title"].str.lower().duplicated()]
    with ThreadPoolExecutor(max_workers=16) as pool:
        ok = list(pool.map(_readable, g["local_image"]))
    g = g[ok].reset_index(drop=True)

    print(f"  gallery: {len(g):,} products ({n_with_image:,} with a local image, "
          f"{ok.count(False)} unreadable, rest duplicates); "
          f"{(g.ai_description != '').sum():,} have an AI description")
    return g[["product_id", "title", "ai_description", "merchant", "price",
              "url", "image_url", "local_image"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.rebuild or not GALLERY_PARQUET.exists():
        print("building gallery ...")
        gallery = build_gallery()
        gallery.to_parquet(GALLERY_PARQUET, index=False)
    else:
        gallery = pd.read_parquet(GALLERY_PARQUET)
        print(f"loaded gallery: {len(gallery):,} products")

    paths = [IMAGES_DIR / n for n in gallery["local_image"]]
    titles = gallery["title"].tolist()

    for key in args.models:
        kind = MODELS[key][1]
        modalities = {"shared": ["image", "title"], "image": ["image"], "text": ["title"]}[kind]
        todo = [m for m in modalities if args.rebuild or not emb_path(key, m).exists()]
        if not todo:
            continue
        enc = Encoder(key)
        for modality in todo:
            out = emb_path(key, modality)
            print(f"embedding {modality}s with {enc.label} on {device()} ...")
            t0 = time.time()
            emb = enc.encode_image_files(paths, progress=True) if modality == "image" \
                else enc.encode_text(titles, progress=True)
            np.save(out, emb)
            print(f"  {emb.shape} in {time.time()-t0:.0f}s -> {out.name}")
        del enc

    print("image indexes ready.")


if __name__ == "__main__":
    main()
