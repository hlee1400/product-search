"""
Compares six image/text search types on the product gallery.

  Same model for both modalities (CLIP / SigLIP / SigLIP 2):
    1. text  -> image   AI description finds the product photo
    2. image -> text    product photo finds the product title
    3. text  -> text    AI description finds the title, using the model's text tower
    4. image -> image   an altered copy of the photo finds the original
  A separate model for each modality:
    5. text  -> text    MiniLM (text-only)
    6. image -> image   DINOv2 (image-only)

Plus one fused row: text query -> RRF(MiniLM over titles, shared model over
images), i.e. a separate text model combined with a shared model's image side.

There are no human relevance labels for this catalog, so each query has
exactly one right answer: the product it was made from.
  - Text queries are the AI-written descriptions. They were written separately
    from the title, so text -> title is not string matching.
  - Image queries for image -> image are a random crop + flip + color shift +
    JPEG re-compression of the product photo, so the original is not a
    pixel-exact match.

Usage:
    python image_search/evaluate.py                 # 2,000 queries, all models
    python image_search/evaluate.py --n 500 --models clip-b32 minilm dinov2-b
"""
from __future__ import annotations

import argparse
import io
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps

from encoders import IMAGE_MODELS, MODELS, SHARED_MODELS, TEXT_MODELS, Encoder, device
from settings import GALLERY_PARQUET, IMAGES_DIR, RESULTS_DIR, emb_path

KS = (1, 10)
RRF_K = 60


def augment(im: Image.Image, seed: int) -> Image.Image:
    """Crop 60-90% of the area, maybe flip, shift color, re-compress as JPEG."""
    rng = random.Random(seed)
    w, h = im.size
    scale = rng.uniform(0.6, 0.9) ** 0.5
    cw, ch = int(w * scale), int(h * scale)
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    im = im.crop((x, y, x + cw, y + ch))
    if rng.random() < 0.5:
        im = ImageOps.mirror(im)
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.75, 1.25))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.7, 1.3))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=rng.randint(50, 80))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def ranks_of_targets(queries: np.ndarray, gallery: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """1-based rank of each query's target in the gallery (ties count against it)."""
    g = torch.from_numpy(gallery).to(device())
    out = []
    for i in range(0, len(queries), 512):
        q = torch.from_numpy(queries[i:i + 512]).to(device())
        t = torch.from_numpy(targets[i:i + 512]).to(device())
        scores = q @ g.T
        target_scores = scores.gather(1, t[:, None])
        out.append(((scores >= target_scores).sum(1)).cpu().numpy())
    return np.concatenate(out)


def topk(queries: np.ndarray, gallery: np.ndarray, k: int) -> np.ndarray:
    g = torch.from_numpy(gallery).to(device())
    out = []
    for i in range(0, len(queries), 512):
        q = torch.from_numpy(queries[i:i + 512]).to(device())
        out.append((q @ g.T).topk(k, dim=1).indices.cpu().numpy())
    return np.concatenate(out)


def rrf_ranks(a: np.ndarray, b: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Fuse two top-k lists per query with RRF; return the target's fused rank (inf if absent)."""
    ranks = []
    for la, lb, t in zip(a, b, targets):
        score: dict[int, float] = {}
        for lst in (la, lb):
            for r, idx in enumerate(lst):
                score[idx] = score.get(idx, 0.0) + 1.0 / (RRF_K + r + 1)
        order = sorted(score, key=score.get, reverse=True)
        ranks.append(order.index(t) + 1 if t in score else np.inf)
    return np.array(ranks, dtype=float)


def metrics(ranks: np.ndarray) -> dict:
    m = {f"R@{k}": float((ranks <= k).mean()) for k in KS}
    m["MRR@10"] = float(np.where(ranks <= 10, 1.0 / ranks, 0.0).mean())
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="number of query products")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gallery = pd.read_parquet(GALLERY_PARQUET)
    has_desc = np.flatnonzero(gallery["ai_description"].str.len() > 0)
    rng = np.random.default_rng(args.seed)
    targets = np.sort(rng.choice(has_desc, size=min(args.n, len(has_desc)), replace=False))
    text_queries = gallery["ai_description"].iloc[targets].tolist()
    image_paths = [IMAGES_DIR / n for n in gallery["local_image"].iloc[targets]]
    aug = {str(p): int(t) for p, t in zip(image_paths, targets)}

    print(f"gallery {len(gallery):,} products, {len(targets):,} queries, device {device()}")

    results: list[dict] = []
    top_lists: dict[str, np.ndarray] = {}   # for the fused row

    def record(search_type: str, family: str, key: str, ranks: np.ndarray) -> None:
        m = metrics(ranks)
        results.append({"type": search_type, "family": family, "model": key,
                        "label": " + ".join(MODELS[k][2] for k in key.split(" + ")), **m})
        print(f"  {search_type:<16} {key:<22} " + "  ".join(f"{k} {v:.3f}" for k, v in m.items()))

    for key in args.models:
        kind = MODELS[key][1]
        enc = Encoder(key)
        t0 = time.time()
        print(f"{enc.label}:")

        if enc.does_text:
            titles = np.load(emb_path(key, "title"))
            q_text = enc.encode_text(text_queries)
        if enc.does_images:
            images = np.load(emb_path(key, "image"))
            q_aug = enc.encode_image_files(
                image_paths, transform=lambda im, path: augment(im, seed=aug[str(path)]),
            )

        family = "same model" if kind == "shared" else "separate models"
        if kind == "shared":
            record("text -> image", family, key, ranks_of_targets(q_text, images, targets))
            record("image -> text", family, key, ranks_of_targets(images[targets], titles, targets))
            top_lists[f"{key}:t2i"] = topk(q_text, images, 50)
        if enc.does_text:
            record("text -> text", family, key, ranks_of_targets(q_text, titles, targets))
            top_lists[f"{key}:t2t"] = topk(q_text, titles, 50)
        if enc.does_images:
            record("image -> image", family, key, ranks_of_targets(q_aug, images, targets))
        print(f"  ({time.time()-t0:.0f}s)")
        del enc
        torch.cuda.empty_cache()

    # Separate text model over titles + shared model over images, fused.
    for text_key in [k for k in args.models if k in TEXT_MODELS]:
        for img_key in [k for k in args.models if k in SHARED_MODELS]:
            ranks = rrf_ranks(top_lists[f"{text_key}:t2t"], top_lists[f"{img_key}:t2i"], targets)
            record("text -> product (RRF)", "fused", f"{text_key} + {img_key}", ranks)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"gallery": len(gallery), "queries": len(targets), "seed": args.seed}
    (RESULTS_DIR / "results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    (RESULTS_DIR / "results.md").write_text(to_markdown(results, meta))
    print(f"\nwrote {RESULTS_DIR / 'results.md'}")


def to_markdown(results: list[dict], meta: dict) -> str:
    df = pd.DataFrame(results)
    lines = [f"Gallery: {meta['gallery']:,} products. Queries: {meta['queries']:,} "
             f"(seed {meta['seed']}). One correct answer per query.\n"]
    for family in ("same model", "separate models", "fused"):
        sub = df[df["family"] == family]
        if sub.empty:
            continue
        lines.append(f"\n### {family.capitalize()}\n")
        lines.append("| Search type | Model | R@1 | R@10 | MRR@10 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for _, r in sub.iterrows():
            lines.append(f"| {r['type']} | {r['label']} | {r['R@1']:.3f} | {r['R@10']:.3f} | {r['MRR@10']:.3f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
