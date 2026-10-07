"""
Does the classified color attribute improve search?

For each benchmark query that names a color ("red summer dress"), products
whose classified color matches get a boost on top of the blended score:

    score = blend(title, photo) + b * [product color in query colors]

b is in z-score units (the blend's scale). b = inf means a hard filter:
only matching products are returned. The boost is chosen on half the products
and reported on the other half, the same split as tune_fusion.py.

It also reports how often the classified color of a query's target product
matches the color the query names, a rough accuracy check of the color
labels by an independent writer.

Usage:
    python image_search/tune_color_boost.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from attributes import load_attributes, query_colors
from benchmark_queries import load_queries
from encoders import Encoder, device
from evaluate import metrics
from fusion import PHOTO_WEIGHT, blend
from generate_queries import QUERIES_DIR
from settings import GALLERY_PARQUET, RESULTS_DIR, emb_path

GENERATOR = "deepseek-4.1-flash"
BOOSTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, float("inf")]


def main() -> None:
    gallery = pd.read_parquet(GALLERY_PARQUET)
    attrs = load_attributes(gallery)
    labeled = (attrs["color"] != "").mean()
    q = load_queries(sorted(QUERIES_DIR.glob("queries_*.jsonl")),
                     {pid: i for i, pid in enumerate(gallery["product_id"])})
    q = q[q["generator"] == GENERATOR].reset_index(drop=True)

    # Same product split as tune_fusion.py (seed 0).
    products = q["product_id"].unique()
    tune = set(np.random.default_rng(0).choice(products, size=len(products) // 2, replace=False))
    q["split"] = np.where(q["product_id"].isin(tune), "tune", "test")

    q["colors"] = q["query"].map(query_colors)
    q["target_color"] = attrs["color"].to_numpy()[q["target"].to_numpy()]
    cq = q[q["colors"].map(len) > 0].reset_index(drop=True)
    cq["label_agrees"] = [t in c for t, c in zip(cq["target_color"], cq["colors"])]
    print(f"{labeled:.1%} of gallery has a color; {len(cq):,} of {len(q):,} queries name a color")

    texts, targets = cq["query"].tolist(), cq["target"].to_numpy()
    dev = device()
    s_title = torch.from_numpy(Encoder("minilm").encode_text(texts)).to(dev) @ \
        torch.from_numpy(np.load(emb_path("minilm", "title"))).to(dev).T
    s_photo = torch.from_numpy(Encoder("siglip2-b16").encode_text(texts)).to(dev) @ \
        torch.from_numpy(np.load(emb_path("siglip2-b16", "image"))).to(dev).T
    base = blend(s_title, s_photo, PHOTO_WEIGHT)

    color_ids = {c: i for i, c in enumerate(sorted(set(attrs["color"]) - {""}))}
    prod_color = torch.tensor([color_ids.get(c, -1) for c in attrs["color"]], device=dev)
    match = torch.zeros_like(base, dtype=torch.bool)
    for row, colors in enumerate(cq["colors"]):
        for c in colors:
            if c in color_ids:
                match[row] |= prod_color == color_ids[c]

    t = torch.from_numpy(targets).to(dev)[:, None]
    ranks = {}
    for b in BOOSTS:
        s = base.masked_fill(~match, float("-inf")) if b == float("inf") else base + b * match
        target = s.gather(1, t)
        r = (s >= target).sum(1).float()
        r[~torch.isfinite(target[:, 0])] = float("inf")   # filtered out
        ranks[b] = r.cpu().numpy()

    sources = sorted(cq["source"].unique())

    def score(b, split):
        res = {}
        for src in sources:
            m = ((cq["split"] == split) & (cq["source"] == src)).to_numpy()
            res[src] = {**metrics(ranks[b][m]), "n": int(m.sum())}
        res["mean MRR@10"] = float(np.mean([res[s]["MRR@10"] for s in sources]))
        return res

    sweep = {b: score(b, "tune") for b in BOOSTS}
    best = max(BOOSTS, key=lambda b: sweep[b]["mean MRR@10"])
    test = {b: score(b, "test") for b in BOOSTS}

    name = lambda b: "filter (only matching color)" if b == float("inf") else f"boost {b}"
    md = [f"Color queries by `{GENERATOR}`: {len(cq):,} of {len(q):,} queries name a color "
          f"({', '.join(f'{s}: {(cq.source == s).sum()}' for s in sources)}). "
          f"Base score: blend with w = {PHOTO_WEIGHT}.\n",
          "**Label check:** the target product's classified color matches the color the query "
          "names in " + ", ".join(f"{(cq[cq.source == s].label_agrees.mean()):.0%} of *from {s}* queries"
                                  for s in sources) + ".\n",
          f"Best on the tune half: **{name(best)}**.\n",
          "\n### Held-out test half, color queries only: MRR@10 [R@10]\n",
          "| Method | " + " | ".join(f"from {s}" for s in sources) + " | mean MRR@10 |",
          "| --- |" + " --- |" * (len(sources) + 1)]
    for b in BOOSTS:
        res = test[b]
        md.append(f"| {name(b)}{' **(best on tune)**' if b == best else ''} | " + " | ".join(
            f"{res[s]['MRR@10']:.3f} [{res[s]['R@10']:.2f}]" for s in sources) + f" | {res['mean MRR@10']:.3f} |")
    md = "\n".join(md) + "\n"
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "color_boost.md").write_text(md, encoding="utf-8")
    (RESULTS_DIR / "color_boost.json").write_text(json.dumps(
        {"best": str(best), "test": {str(k): v for k, v in test.items()},
         "tune": {str(k): v for k, v in sweep.items()}}, indent=2))
    print("\n" + md)


if __name__ == "__main__":
    main()
