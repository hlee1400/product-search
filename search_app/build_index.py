"""
Builds the search indexes over the local product catalog in ../data, using the
BM25 + dense retrieval code from search_pipeline unchanged.

Usage:
    python search_app/build_index.py            # build whatever is missing
    python search_app/build_index.py --rebuild  # rebuild everything
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import glob
import time

import numpy as np
import pandas as pd

import settings  # noqa: F401  (puts the pipeline repo on sys.path)
from settings import (
    ARTIFACTS_DIR, BM25_DIR, BM25_PIDS, CATALOG_CSV, CORPUS_PARQUET,
    DATA_DIR, DENSE_EMB, DENSE_PIDS, IMAGES_DIR,
)

from src.bm25.bm25_search import build_index, save_index as bm25_save_index
from src.dense.dense_search import (
    DEFAULT_MODEL as DENSE_MODEL,
    device,
    embed_corpus,
    load_model,
    save_index as dense_save_index,
)


def clean_title(title: str) -> str:
    # Marketplace boilerplate ("Amazon.com: ...", "... - Etsy") is on >100k
    # titles and carries no product signal, so strip it before indexing.
    t = str(title).strip()
    for prefix in ("Amazon.com: ", "Amazon.com : "):
        if t.startswith(prefix):
            t = t[len(prefix):]
    for suffix in (" - Etsy", " | Etsy"):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
    return t.strip()


def build_corpus() -> pd.DataFrame:
    print(f"loading catalog {CATALOG_CSV.name} ...")
    cat = pd.read_csv(
        CATALOG_CSV,
        low_memory=False,
        usecols=[
            "global_product_id", "advertiser_program_name", "most_recent_product_name",
            "most_recent_product_description", "linked_product_url", "most_recent_price",
            "most_recent_product_image", "outpath",
        ],
    )

    print("loading enriched descriptions ...")
    enriched = pd.concat(
        pd.read_csv(f, low_memory=False, usecols=["global_product_id", "text_description"])
        for f in sorted(glob.glob(str(DATA_DIR / "*enriched_descriptions.csv")))
    ).drop_duplicates("global_product_id")

    df = cat.merge(enriched, on="global_product_id", how="left")
    df = df.dropna(subset=["global_product_id", "most_recent_product_name"])
    df = df.drop_duplicates("global_product_id")

    local_images = set(os.listdir(IMAGES_DIR)) if IMAGES_DIR.is_dir() else set()
    img_name = df["outpath"].astype(str).str.split("/").str[-1]

    corpus = pd.DataFrame({
        "product_id": df["global_product_id"].astype("int64").astype(str),
        "title": df["most_recent_product_name"].map(clean_title),
        "description": df["most_recent_product_description"].fillna("").astype(str),
        "ai_description": df["text_description"].fillna("").astype(str),
        "merchant": df["advertiser_program_name"].fillna("").astype(str),
        "price": pd.to_numeric(df["most_recent_price"], errors="coerce"),
        "url": df["linked_product_url"].fillna("").astype(str),
        "image_url": df["most_recent_product_image"].fillna("").astype(str),
        "local_image": img_name.where(img_name.isin(local_images), ""),
    })
    corpus = corpus[corpus["title"].str.len() > 0].reset_index(drop=True)
    print(f"  corpus: {len(corpus):,} products "
          f"({(corpus.ai_description != '').sum():,} enriched, "
          f"{(corpus.local_image != '').sum():,} with local image)")
    return corpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.rebuild or not CORPUS_PARQUET.exists():
        corpus = build_corpus()
        corpus.to_parquet(CORPUS_PARQUET, index=False)
        print(f"saved -> {CORPUS_PARQUET}")
    else:
        print(f"loading cached corpus {CORPUS_PARQUET} ...")
        corpus = pd.read_parquet(CORPUS_PARQUET)

    # Title-only retrieval text, matching the pipeline's design decision.
    titles = corpus["title"].tolist()
    pids = corpus["product_id"].to_numpy()

    if args.rebuild or not BM25_PIDS.exists():
        print("building bm25 index ...")
        t0 = time.time()
        idx = build_index(titles, k1=1.2, b=0.75, show_progress=False)
        bm25_save_index(idx, pids, str(BM25_DIR), str(BM25_PIDS))
        print(f"  done in {time.time()-t0:.1f}s -> {BM25_DIR}")

    if args.rebuild or not DENSE_EMB.exists():
        print(f"embedding corpus with {DENSE_MODEL} on {device()} ...")
        t0 = time.time()
        model = load_model(DENSE_MODEL)
        emb = embed_corpus(model, titles, batch_size=256, show_progress_bar=True)
        dense_save_index(emb, pids, str(DENSE_EMB), str(DENSE_PIDS))
        print(f"  done in {time.time()-t0:.0f}s, shape={emb.shape} -> {DENSE_EMB}")

    print("indexes ready.")


if __name__ == "__main__":
    main()
