"""
Scores every text-query search type against the LLM-written queries from
generate_queries.py.

Each query's correct answer is the product it was written for, the same
one-right-answer setup as evaluate.py. Results are split by who wrote the
query (generator), what it was written from (text-only vs image-only), and
query style (short / specific / need).

Search types scored (all take a text query):
    text -> image    shared model (CLIP / SigLIP / SigLIP 2) over photos
    text -> text     shared model's text tower over titles
    text -> text     MiniLM over titles (separate text model)
    text -> product  MiniLM titles + shared-model photos, fused with RRF

Usage:
    python image_search/benchmark_queries.py                       # every queries/*.jsonl
    python image_search/benchmark_queries.py --queries image_search/queries/queries_claude.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from encoders import MODELS, SHARED_MODELS, TEXT_MODELS, Encoder
from evaluate import metrics, ranks_of_targets, rrf_ranks, topk
from generate_queries import QUERIES_DIR
from settings import GALLERY_PARQUET, RESULTS_DIR, emb_path


def load_queries(paths: list[Path], pid_to_row: dict[str, int]) -> pd.DataFrame:
    df = pd.concat([pd.read_json(p, lines=True, dtype={"product_id": str}) for p in paths],
                   ignore_index=True)
    df = df[df["product_id"].isin(pid_to_row)].reset_index(drop=True)
    df["target"] = df["product_id"].map(pid_to_row)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", nargs="+", type=Path,
                    default=sorted(QUERIES_DIR.glob("queries_*.jsonl")))
    ap.add_argument("--models", nargs="+", default=SHARED_MODELS + TEXT_MODELS,
                    choices=SHARED_MODELS + TEXT_MODELS)
    args = ap.parse_args()
    if not args.queries:
        raise SystemExit("No query files - run `python image_search/generate_queries.py` first.")

    gallery = pd.read_parquet(GALLERY_PARQUET)
    pid_to_row = {pid: i for i, pid in enumerate(gallery["product_id"])}
    q = load_queries(args.queries, pid_to_row)
    targets = q["target"].to_numpy()
    texts = q["query"].tolist()
    print(f"{len(q):,} queries over {q.product_id.nunique():,} products from "
          + ", ".join(p.name for p in args.queries))

    ranks: dict[tuple[str, str], np.ndarray] = {}   # (search type, model label) -> rank per query
    top: dict[str, np.ndarray] = {}
    for key in args.models:
        enc = Encoder(key)
        q_emb = enc.encode_text(texts)
        titles = np.load(emb_path(key, "title"))
        ranks[("text -> text", enc.label)] = ranks_of_targets(q_emb, titles, targets)
        top[f"{key}:t2t"] = topk(q_emb, titles, 50)
        if key in SHARED_MODELS:
            images = np.load(emb_path(key, "image"))
            ranks[("text -> image", enc.label)] = ranks_of_targets(q_emb, images, targets)
            top[f"{key}:t2i"] = topk(q_emb, images, 50)
        print(f"  {enc.label} done")
        del enc
        torch.cuda.empty_cache()

    for t in [k for k in args.models if k in TEXT_MODELS]:
        for s in [k for k in args.models if k in SHARED_MODELS]:
            label = f"{MODELS[t][2]} + {MODELS[s][2]}"
            ranks[("text -> product (RRF)", label)] = rrf_ranks(top[f"{t}:t2t"], top[f"{s}:t2i"], targets)

    # Score each (generator, source) slice, and each style within it.
    groups = [((g, s), (q.generator == g) & (q.source == s))
              for g, s in q[["generator", "source"]].drop_duplicates().itertuples(index=False)]
    results = []
    for (stype, label), r in ranks.items():
        for (gen, src), mask in groups:
            results.append({"type": stype, "model": label, "generator": gen, "source": src,
                            "style": "all", "n": int(mask.sum()), **metrics(r[mask.to_numpy()])})
            for style in q.loc[mask, "style"].unique():
                m = (mask & (q["style"] == style)).to_numpy()
                if m.any():
                    results.append({"type": stype, "model": label, "generator": gen, "source": src,
                                    "style": style, "n": int(m.sum()), **metrics(r[m])})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "generated_queries.json").write_text(json.dumps(results, indent=2))
    md = to_markdown(pd.DataFrame(results), q)
    (RESULTS_DIR / "generated_queries.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"wrote {RESULTS_DIR / 'generated_queries.md'}")


def to_markdown(df: pd.DataFrame, q: pd.DataFrame) -> str:
    out = [f"{len(q):,} LLM-written queries for {q.product_id.nunique():,} products. "
           "One correct answer per query (the product it was written for).\n"]
    allq = df[df["style"] == "all"].copy()
    allq["col"] = allq["generator"] + ", from " + allq["source"]
    for metric in ("MRR@10", "R@10"):
        wide = allq.pivot_table(index=["type", "model"], columns="col", values=metric, sort=False)
        out.append(f"\n### {metric}\n")
        out.append("| Search type | Model | " + " | ".join(wide.columns) + " |")
        out.append("| --- | --- |" + " --- |" * len(wide.columns))
        for (stype, model), row in wide.iterrows():
            out.append(f"| {stype} | {model} | " + " | ".join(f"{v:.3f}" for v in row) + " |")

    styled = df[df["style"] != "all"].copy()
    styled["col"] = styled["generator"] + ", from " + styled["source"] + ", " + styled["style"]
    wide = styled.pivot_table(index=["type", "model"], columns="col", values="MRR@10", sort=False)
    out.append("\n### MRR@10 by query style\n")
    out.append("| Search type | Model | " + " | ".join(wide.columns) + " |")
    out.append("| --- | --- |" + " --- |" * len(wide.columns))
    for (stype, model), row in wide.iterrows():
        out.append(f"| {stype} | {model} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
