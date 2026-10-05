"""
Search UI server over the local product catalog.

Retrieval runs the search_pipeline pipeline:
    BM25 top-50  ┐
                 ├─ RRF fuse ─ cross-encoder rerank
    dense top-50 ┘

Usage:
    python search_app/build_index.py   # once
    python search_app/app.py           # then open http://127.0.0.1:5000
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import math
import threading
import time

import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory

from settings import (  # also puts the pipeline repo on sys.path
    APP_DIR, BM25_DIR, BM25_PIDS, CORPUS_PARQUET, DENSE_EMB, DENSE_PIDS, IMAGES_DIR,
)

from src.bm25.bm25_search import load_index as bm25_load_index, search as bm25_search, tokenize
from src.dense.dense_search import (
    DEFAULT_MODEL as DENSE_MODEL,
    device,
    load_index as dense_load_index,
    load_model as dense_load_model,
    search_batch as dense_search_batch,
)
from src.hybrid.hybrid_search import (
    DEFAULT_RERANKER,
    RECALL_K,
    load_cross_encoder,
    rerank,
    rrf_fuse,
)

MODES = ("rerank", "rrf", "bm25", "dense")


class SearchEngine:
    def __init__(self) -> None:
        if not (CORPUS_PARQUET.exists() and BM25_PIDS.exists() and DENSE_EMB.exists()):
            raise SystemExit("Indexes missing - run `python search_app/build_index.py` first.")

        t0 = time.time()
        print("loading corpus ...", flush=True)
        self.corpus = pd.read_parquet(CORPUS_PARQUET).set_index("product_id", drop=False)
        self.pid_to_title = dict(zip(self.corpus["product_id"], self.corpus["title"]))

        print("loading bm25 index ...", flush=True)
        self.bm25_index, self.bm25_pids = bm25_load_index(str(BM25_DIR), str(BM25_PIDS))

        print(f"loading dense index ({DENSE_MODEL}) ...", flush=True)
        self.dense_faiss, self.dense_pids = dense_load_index(str(DENSE_EMB), str(DENSE_PIDS))
        self.dense_model = dense_load_model(DENSE_MODEL)

        print(f"loading reranker ({DEFAULT_RERANKER}) ...", flush=True)
        self.cross_encoder = load_cross_encoder(DEFAULT_RERANKER)

        # Models are not guaranteed thread-safe under Flask's threaded server.
        self.lock = threading.Lock()
        print(f"ready: {len(self.corpus):,} products on {device()} "
              f"in {time.time()-t0:.0f}s", flush=True)

    def search(self, query: str) -> dict:
        """Same stages as hybrid_search.run_pipeline, with per-stage timings."""
        timings = {}
        with self.lock:
            t = time.perf_counter()
            dense_top = dense_search_batch(
                self.dense_model, self.dense_faiss, self.dense_pids, [query], top_k=RECALL_K
            )[0]
            timings["dense"] = time.perf_counter() - t

            t = time.perf_counter()
            # bm25s returns arbitrary zero-score docs when no query token is in
            # the vocabulary; drop them so they don't get RRF credit.
            bm25_top = [
                pid for pid, score in bm25_search(self.bm25_index, self.bm25_pids, query, top_k=RECALL_K)
                if score > 0
            ]
            timings["bm25"] = time.perf_counter() - t

            t = time.perf_counter()
            rrf_top = rrf_fuse(dense_top, bm25_top, top_k=RECALL_K)
            timings["rrf"] = time.perf_counter() - t

            t = time.perf_counter()
            # Rerank the whole pool (not just TOP_K) so the UI can page through it.
            reranked = rerank(self.cross_encoder, query, rrf_top, self.pid_to_title, top_k=len(rrf_top))
            timings["rerank"] = time.perf_counter() - t

        return {"dense": dense_top, "bm25": bm25_top, "rrf": rrf_top, "rerank": reranked}, timings

    def product(self, pid: str, ranks: dict) -> dict:
        row = self.corpus.loc[pid]
        price = row["price"]
        if price is None or (isinstance(price, float) and math.isnan(price)) or price <= 0 or price > 100_000:
            price = None
        return {
            "id": pid,
            "title": row["title"],
            "merchant": row["merchant"],
            "price": price,
            "url": row["url"],
            "image": f"/images/{row['local_image']}" if row["local_image"] else row["image_url"],
            "image_fallback": row["image_url"],
            "description": row["description"][:1200],
            "ai_description": row["ai_description"],
            "ranks": ranks,
        }


app = Flask(__name__, static_folder=str(APP_DIR / "static"), static_url_path="/static")
engine: SearchEngine | None = None


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/images/<path:name>")
def image(name: str):
    return send_from_directory(IMAGES_DIR, name, max_age=86400)


@app.get("/api/status")
def status():
    return jsonify({
        "products": len(engine.corpus),
        "dense_model": DENSE_MODEL,
        "reranker": DEFAULT_RERANKER,
        "recall_k": RECALL_K,
        "device": device(),
    })


@app.get("/api/search")
def api_search():
    query = request.args.get("q", "").strip()[:200]
    mode = request.args.get("mode", "rerank")
    if mode not in MODES:
        abort(400, f"mode must be one of {MODES}")
    if not query:
        return jsonify({"query": "", "mode": mode, "results": [], "timings": {}})
    if not tokenize(query):
        return jsonify({"query": query, "mode": mode, "results": [], "timings": {},
                        "message": "Query has no searchable words."})

    t0 = time.perf_counter()
    stages, timings = engine.search(query)
    rank_maps = {s: {pid: i + 1 for i, pid in enumerate(lst)} for s, lst in stages.items()}
    # The catalog lists some products several times under different ids;
    # show each (title, merchant) once, at its best rank.
    results, seen = [], set()
    for pid in stages[mode]:
        p = engine.product(pid, {s: rank_maps[s].get(pid) for s in MODES})
        key = (p["title"].lower(), p["merchant"])
        if key not in seen:
            seen.add(key)
            results.append(p)
    timings["total"] = time.perf_counter() - t0
    return jsonify({
        "query": query,
        "mode": mode,
        "results": results,
        "timings": {k: round(v * 1000, 1) for k, v in timings.items()},
        "counts": {s: len(lst) for s, lst in stages.items()},
    })


if __name__ == "__main__":
    engine = SearchEngine()
    port = int(os.environ.get("PORT", 5000))
    print(f"open http://127.0.0.1:{port}", flush=True)
    app.run(host="127.0.0.1", port=port, threaded=True)
