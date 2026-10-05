from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.bm25.bm25_search import (
    format_metrics,
    hits_and_mrr,
    load_us_corpus,
    load_us_test_set,
    search_batch as bm25_search_batch,
)
from src.dense.dense_search import (
    DEFAULT_MODEL as DEFAULT_DENSE_MODEL,
    search_batch as dense_search_batch,
)

#strong general-domain reranker with low parameter count
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Recall stage pulls a 50-candidate pool per retriever; the cross-encoder
# then trims that down to the final 10 results returned to the user
RECALL_K = 50
TOP_K = 10

# RRF's smoothing constant; 60 is the value from the original Cormack et al.
# paper and is robust to wide rank distributions
RRF_K = 60


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_cross_encoder(
    model_id: str = DEFAULT_RERANKER,
    max_length: int = 256,
) -> CrossEncoder:
    return CrossEncoder(model_id, device=device(), max_length=max_length)


def rrf_fuse(*lists: list[str], k: int = RRF_K, top_k: int = RECALL_K) -> list[str]:
    """Reciprocal Rank Fusion: score(item) = Σ_L 1 / (k + rank_L(item))."""
    scores: dict[str, float] = {}
    for L in lists:
        for r, item in enumerate(L):
            if item is None:
                continue
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + r + 1)
    return sorted(scores, key=lambda x: -scores[x])[:top_k]


def rerank(
    cross_encoder: CrossEncoder,
    query: str,
    candidate_pids: list[str],
    pid_to_title: dict[str, str],
    top_k: int = TOP_K,
    batch_size: int = 64,
) -> list[str]:
    # Builds (query, title) pairs and scores them jointly, attention runs over both sides together, 
    # which is what lets the model demote topically related but wrong item
    pairs = [(query, pid_to_title.get(pid, "")) for pid in candidate_pids]
    scores = cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    order = np.argsort(-np.asarray(scores))[:top_k]
    return [candidate_pids[i] for i in order]


def run_pipeline(
    query: str,
    bm25_index, bm25_pids,
    dense_model: SentenceTransformer, dense_faiss, dense_pids,
    cross_encoder: CrossEncoder,
    pid_to_title: dict[str, str],
    recall_k: int = RECALL_K,
    top_k: int = TOP_K,
) -> dict:
    dense_top = dense_search_batch(dense_model, dense_faiss, dense_pids, [query], top_k=recall_k)[0]
    bm25_top  = bm25_search_batch(bm25_index, bm25_pids, [query], top_k=recall_k)[0]
    rrf_top   = rrf_fuse(dense_top, bm25_top, top_k=recall_k)
    reranked  = rerank(cross_encoder, query, rrf_top, pid_to_title, top_k=top_k)
    return {
        "dense_top": dense_top,
        "bm25_top":  bm25_top,
        "rrf_top":   rrf_top,
        "reranked":  reranked,
    }


def evaluate_test_set(
    bm25_index, bm25_pids,
    dense_model: SentenceTransformer, dense_faiss, dense_pids,
    cross_encoder: CrossEncoder,
    pid_to_title: dict[str, str],
    recall_k: int = RECALL_K,
    top_k: int = TOP_K,
) -> dict:
    ts = load_us_test_set()
    qs = ts.queries["query"].tolist()

    print(f"[stage 1a] dense top-{recall_k} ...", flush=True)
    t0 = time.time()
    dense_lists = dense_search_batch(dense_model, dense_faiss, dense_pids, qs, top_k=recall_k)
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"[stage 1b] BM25 top-{recall_k} ...", flush=True)
    t0 = time.time()
    bm25_lists = bm25_search_batch(bm25_index, bm25_pids, qs, top_k=recall_k)
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"[stage 2] RRF fuse -> top-{recall_k} ...", flush=True)
    rrf_lists = [rrf_fuse(d, b, top_k=recall_k) for d, b in zip(dense_lists, bm25_lists)]

    print(f"[stage 3] cross-encoder rerank top-{recall_k} -> top-{top_k} ...", flush=True)
    t0 = time.time()
    reranked = []
    for i, (q, cands) in enumerate(zip(qs, rrf_lists)):
        reranked.append(rerank(cross_encoder, q, cands, pid_to_title, top_k=top_k))
        if (i + 1) % 500 == 0:
            print(f"    {i+1:,}/{len(qs):,}", flush=True)
    print(f"  done in {time.time()-t0:.0f}s")

    def measure(lists, name):
        lists_top10 = [L[:top_k] for L in lists]
        return {
            "name": name,
            "strict_E":   hits_and_mrr(lists_top10, ts.queries, ts.e_by_q),
            "lenient_ES": hits_and_mrr(lists_top10, ts.queries, ts.es_by_q),
        }

    return {
        "dense":    measure(dense_lists, "dense alone"),
        "bm25":     measure(bm25_lists,  "BM25 alone"),
        "rrf":      measure(rrf_lists,   "RRF(dense, BM25)"),
        "reranked": measure(reranked,    f"RRF + rerank top-{recall_k}"),
    }


def build_pid_to_title() -> dict[str, str]:
    df = load_us_corpus()
    return dict(zip(df["product_id"], df["product_title"]))


__all__ = [
    "DEFAULT_DENSE_MODEL",
    "DEFAULT_RERANKER",
    "RECALL_K",
    "TOP_K",
    "RRF_K",
    "device",
    "load_cross_encoder",
    "rrf_fuse",
    "rerank",
    "run_pipeline",
    "evaluate_test_set",
    "build_pid_to_title",
    "format_metrics",
]
