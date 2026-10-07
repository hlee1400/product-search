"""
Tunes how a text query's title matches (MiniLM) and photo matches (a shared
model such as SigLIP 2) are combined.

Equal-weight RRF helped on queries written from product text but hurt on
queries written from the photo (results/generated_queries.md). This script
tries two weighted alternatives:

  weighted RRF   score = w / (60 + photo rank) + (1 - w) / (60 + title rank),
                 over each side's top 100
  score blend    score = w * z(photo similarity) + (1 - w) * z(title similarity),
                 z-normalized per query over the whole gallery

w = 0 is title search only and w = 1 is photo search only.

To avoid tuning and scoring on the same data, products are split in half:
the weight is chosen on the tune half and reported on the held-out test half.
The search engine can't tell whether a query was written from text or from a
photo, so one weight is chosen for both query types together (by mean MRR@10
across them).

Usage:
    python image_search/tune_fusion.py                          # all query files
    python image_search/tune_fusion.py --generator qwen2.5vl:7b # one writer only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmark_queries import load_queries
from encoders import MODELS, Encoder, device
from evaluate import RRF_K, metrics
from fusion import PHOTO_WEIGHT, zscore
from generate_queries import QUERIES_DIR
from settings import GALLERY_PARQUET, RESULTS_DIR, emb_path

WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 2)
RECALL_K = 100


def fused_ranks(q_title: np.ndarray, q_photo: np.ndarray, titles: np.ndarray, photos: np.ndarray,
                targets: np.ndarray) -> dict[str, np.ndarray]:
    """Rank of each query's target under every method and weight."""
    dev = device()
    T = torch.from_numpy(titles).to(dev)
    P = torch.from_numpy(photos).to(dev)
    out = {f"{m}@{w}": [] for m in ("rrf", "blend") for w in WEIGHTS}
    for i in range(0, len(targets), 256):
        t = torch.from_numpy(targets[i:i + 256]).to(dev)[:, None]
        s_title = torch.from_numpy(q_title[i:i + 256]).to(dev) @ T.T
        s_photo = torch.from_numpy(q_photo[i:i + 256]).to(dev) @ P.T

        # Score blend: exact rank over the full gallery.
        z_title, z_photo = zscore(s_title), zscore(s_photo)
        for w in WEIGHTS:
            s = w * z_photo + (1 - w) * z_title
            out[f"blend@{w}"].append((s >= s.gather(1, t)).sum(1).cpu().numpy())

        # Weighted RRF over each side's top RECALL_K, like the app does.
        n, g = s_title.shape
        r_title = torch.full((n, g), float("inf"), device=dev)
        r_photo = torch.full((n, g), float("inf"), device=dev)
        rows = torch.arange(n, device=dev)[:, None]
        ranks = torch.arange(1, RECALL_K + 1, device=dev, dtype=torch.float32).expand(n, -1)
        r_title[rows, s_title.topk(RECALL_K, 1).indices] = ranks
        r_photo[rows, s_photo.topk(RECALL_K, 1).indices] = ranks
        for w in WEIGHTS:
            s = w / (RRF_K + r_photo) + (1 - w) / (RRF_K + r_title)   # 1/inf = 0
            target = s.gather(1, t)
            rank = (s >= target).sum(1).float()
            rank[target[:, 0] == 0] = float("inf")   # target in neither top list
            out[f"rrf@{w}"].append(rank.cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", nargs="+", type=Path, default=sorted(QUERIES_DIR.glob("queries_*.jsonl")))
    ap.add_argument("--generator", help="only use queries written by this model")
    ap.add_argument("--title-model", default="minilm")
    ap.add_argument("--photo-model", default="siglip2-b16")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gallery = pd.read_parquet(GALLERY_PARQUET)
    q = load_queries(args.queries, {pid: i for i, pid in enumerate(gallery["product_id"])})
    if args.generator:
        q = q[q["generator"] == args.generator].reset_index(drop=True)
    print(f"{len(q):,} queries, {q.product_id.nunique()} products, "
          f"writers/sources: {q.groupby(['generator', 'source']).size().to_dict()}")

    # Split by product so a product's queries are all on one side.
    products = q["product_id"].unique()
    rng = np.random.default_rng(args.seed)
    tune_products = set(rng.choice(products, size=len(products) // 2, replace=False))
    q["split"] = np.where(q["product_id"].isin(tune_products), "tune", "test")

    texts, targets = q["query"].tolist(), q["target"].to_numpy()
    q_title = Encoder(args.title_model).encode_text(texts)
    q_photo = Encoder(args.photo_model).encode_text(texts)
    ranks = fused_ranks(q_title, q_photo, np.load(emb_path(args.title_model, "title")),
                        np.load(emb_path(args.photo_model, "image")), targets)

    sources = sorted(q["source"].unique())

    def score(key: str, split: str) -> dict:
        res = {}
        for src in sources:
            m = ((q["split"] == split) & (q["source"] == src)).to_numpy()
            res[src] = metrics(ranks[key][m])
        res["mean MRR@10"] = float(np.mean([res[s]["MRR@10"] for s in sources]))
        return res

    # Pick the weight per method on the tune half.
    sweep = {k: score(k, "tune") for k in ranks}
    best = {m: max((k for k in ranks if k.startswith(m)), key=lambda k: sweep[k]["mean MRR@10"])
            for m in ("rrf", "blend")}

    title_label, photo_label = MODELS[args.title_model][2], MODELS[args.photo_model][2]
    rows = [
        (f"title only ({title_label})", "blend@0.0"),
        (f"photo only ({photo_label})", "blend@1.0"),
        ("equal-weight RRF (old app)", "rrf@0.5"),
        (f"weighted RRF, w={best['rrf'].split('@')[1]}", best["rrf"]),
        (f"score blend, w={best['blend'].split('@')[1]} (best on tune half)", best["blend"]),
    ]
    if f"blend@{PHOTO_WEIGHT}" != best["blend"]:
        rows.append((f"score blend, w={PHOTO_WEIGHT} (fusion.PHOTO_WEIGHT)", f"blend@{PHOTO_WEIGHT}"))
    test = {name: score(key, "test") for name, key in rows}

    md = [f"Title model: {title_label}. Photo model: {photo_label}. "
          f"{len(q):,} queries; weights chosen on {len(tune_products)} products, "
          f"scored on the other {len(products) - len(tune_products)}.\n",
          "w is the weight on the photo side (0 = titles only, 1 = photos only).\n",
          "\n### Held-out test half: MRR@10 [R@10]\n",
          "| Method | " + " | ".join(f"from {s}" for s in sources) + " | mean MRR@10 |",
          "| --- |" + " --- |" * (len(sources) + 1)]
    for name, res in test.items():
        md.append(f"| {name} | " + " | ".join(
            f"{res[s]['MRR@10']:.3f} [{res[s]['R@10']:.2f}]" for s in sources)
            + f" | {res['mean MRR@10']:.3f} |")
    md += ["\n### Weight sweep on the tune half: mean MRR@10\n",
           "| w | " + " | ".join(str(w) for w in WEIGHTS) + " |",
           "| --- |" + " --- |" * len(WEIGHTS)]
    for m in ("rrf", "blend"):
        md.append(f"| {m} | " + " | ".join(f"{sweep[f'{m}@{w}']['mean MRR@10']:.3f}" for w in WEIGHTS) + " |")
    md = "\n".join(md) + "\n"

    tag = f"_{args.generator.replace(':', '-')}" if args.generator else ""
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"fusion{tag}.md").write_text(md, encoding="utf-8")
    (RESULTS_DIR / f"fusion{tag}.json").write_text(json.dumps(
        {"test": test, "tune_sweep": sweep, "best": best}, indent=2))
    print("\n" + md)
    print(f"wrote {RESULTS_DIR / f'fusion{tag}.md'}")


if __name__ == "__main__":
    main()
