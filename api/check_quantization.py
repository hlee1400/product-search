"""
Does int8 quantization of the query encoders hurt search quality?

The index vectors stay full precision; only the models that encode incoming
queries are quantized (to fit a 2 GB machine that can suspend). This compares
full-precision vs int8 encoders, both on CPU, on:
  - text: the 3,704 LLM-written benchmark queries, ranked with the production
    blend (w = 0.7), MRR@10 by query source
  - photos: 300 altered copies of product photos (crop, flip, color shift, JPEG),
    ranked by photo similarity, MRR@10
plus how far the query vectors move (cosine) and per-query CPU time.

Runs inside the production image, with the repo mounted read-only:
    docker run --rm -v "<repo>:/repo:ro" -e TORCH_THREADS=2 product-search-api \
        python /repo/api/check_quantization.py
"""
from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "image_search"))

from encoders import Encoder, load_image  # noqa: E402
from evaluate import augment, metrics  # noqa: E402
from fusion import blend  # noqa: E402

BUNDLE = REPO / "api" / "bundle"
IMAGES = REPO / "data" / "product_images" / "product_images"
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
torch.set_grad_enabled(False)


def ranks(scores: torch.Tensor, targets: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(targets)[:, None]
    return (scores >= scores.gather(1, t)).sum(1).numpy()


def main() -> None:
    items = pd.read_parquet(BUNDLE / "items.parquet")
    row = {pid: i for i, pid in enumerate(items["product_id"])}
    photo = torch.from_numpy(np.load(BUNDLE / "photo_vectors.npy").astype(np.float32))
    title = torch.from_numpy(np.load(BUNDLE / "title_vectors.npy").astype(np.float32))

    q = pd.read_json(REPO / "image_search/queries/queries_api.jsonl", lines=True, dtype={"product_id": str})
    q = q[q["product_id"].isin(row)].reset_index(drop=True)
    texts, targets = q["query"].tolist(), q["product_id"].map(row).to_numpy()

    rng = np.random.default_rng(0)
    img_rows = np.sort(rng.choice(len(items), size=300, replace=False))
    img_queries = [augment(load_image(IMAGES / items["photo_key"].iloc[i]), seed=int(i)) for i in img_rows]

    out = {}
    for name, quant in (("full precision", False), ("int8", True)):
        t0 = time.time()
        siglip, minilm = Encoder("siglip2-b16", quantize=quant), Encoder("minilm", quantize=quant)
        load_s = time.time() - t0

        t0 = time.time()
        qp = torch.from_numpy(siglip.encode_text(texts, batch_size=64))
        qt = torch.from_numpy(minilm.encode_text(texts, batch_size=64))
        text_ms = (time.time() - t0) / len(texts) * 1000
        r_text = np.concatenate([ranks(blend(qt[i:i + 256] @ title.T, qp[i:i + 256] @ photo.T),
                                       targets[i:i + 256]) for i in range(0, len(texts), 256)])

        t0 = time.time()
        qi = torch.from_numpy(np.concatenate([siglip.encode_images(img_queries[i:i + 16])
                                              for i in range(0, len(img_queries), 16)]))
        img_ms = (time.time() - t0) / len(img_queries) * 1000
        r_img = ranks(qi @ photo.T, img_rows)

        out[name] = {"qp": qp, "qi": qi, "load_s": load_s, "text_ms": text_ms, "img_ms": img_ms,
                     "text": {s: metrics(r_text[(q["source"] == s).to_numpy()]) for s in sorted(q["source"].unique())},
                     "image": metrics(r_img)}
        del siglip, minilm

    fp, i8 = out["full precision"], out["int8"]
    print(f"{len(texts):,} text queries, {len(img_queries)} photo queries, "
          f"{torch.get_num_threads()} CPU threads\n")
    print("| | " + " | ".join(f"text from {s}" for s in fp["text"]) + " | photo → photo | text ms/query | photo ms/query |")
    print("| --- |" + " --- |" * (len(fp["text"]) + 3))
    for name, r in out.items():
        print(f"| {name} | " + " | ".join(f"{m['MRR@10']:.3f}" for m in r["text"].values())
              + f" | {r['image']['MRR@10']:.3f} | {r['text_ms']:.0f} | {r['img_ms']:.0f} |")
    cos_t = (fp["qp"] * i8["qp"]).sum(1)
    cos_i = (fp["qi"] * i8["qi"]).sum(1)
    print(f"\ncosine(full, int8) SigLIP text vectors: mean {cos_t.mean():.4f}, min {cos_t.min():.4f}")
    print(f"cosine(full, int8) SigLIP photo vectors: mean {cos_i.mean():.4f}, min {cos_i.min():.4f}")
    print(f"peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")


if __name__ == "__main__":
    main()
