from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import re
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import bm25s
import numpy as np
import pandas as pd

from src.data.load_data import load_products, load_task_data
from src.config.paths import BM25_INDEX_DIR, BM25_PIDS_PATH


INDEX_DIR = str(BM25_INDEX_DIR)
PIDS_PATH = str(BM25_PIDS_PATH)


# Minimal stop list, product titles are already short, aggressive stop-word
# removal would strip useful signal (e.g. "in" inside "8 in drill bit")
STOPWORDS = {"the", "a", "an"}


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
    # Lowercase + alphanumeric-only split; keeps "1/2in" → ["1", "2in"] which
    # is acceptable on these short product titles but is a known sharp edge.
        return []
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in toks if t not in STOPWORDS]


def build_index(
    corpus_texts: list[str],
    k1: float = 1.2,
    b: float = 0.75,
    show_progress: bool = True,
) -> bm25s.BM25:
    # k1/b are the bm25s defaults
    tokenized = [tokenize(t) for t in corpus_texts]
    retriever = bm25s.BM25(k1=k1, b=b)
    retriever.index(tokenized, show_progress=show_progress)
    return retriever


def search(
    index: bm25s.BM25,
    pids: np.ndarray,
    query: str,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    results = index.retrieve([q_tokens], k=top_k, show_progress=False)
    doc_idxs = results.documents[0]
    scores = results.scores[0]
    return [(str(pids[i]), float(s)) for i, s in zip(doc_idxs, scores)]


def search_batch(
    index: bm25s.BM25,
    pids: np.ndarray,
    queries: list[str],
    top_k: int = 10,
    show_progress: bool = False,
) -> list[list[str]]:
    q_tokens = [tokenize(q) for q in queries]
    results = index.retrieve(q_tokens, k=top_k, show_progress=show_progress)
    return [[str(pids[i]) for i in row] for row in results.documents]


# ── Corpus + test-set loaders ──────────
def load_us_corpus() -> pd.DataFrame:
    """Return US products with at least a product_id and product_title."""
    products = load_products()
    us = products[products["product_locale"] == "us"].copy()
    us = us.dropna(subset=["product_id", "product_title"])
    us = us.drop_duplicates(subset=["product_id"])
    return us.reset_index(drop=True)


@dataclass
class TestSet:
    queries: pd.DataFrame           # columns: query_id, query
    e_by_q: dict[str, set[str]]     # strict — only E judgements
    es_by_q: dict[str, set[str]]    # lenient — E or S judgements


def load_us_test_set() -> TestSet:
    """US small-version test split, packaged like the original `common.load_test_set`."""
    # Builds both strict (E) and lenient (E & S) ground-truth maps so the
    # same retrieval run can be scored under both criteria without re-querying.
    _, test_df = load_task_data(small_version=True, locale="us")

    queries = (
        test_df[["query_id", "query"]]
        .dropna()
        .drop_duplicates(subset=["query_id"])
        .reset_index(drop=True)
    )

    e_rows = test_df[test_df["esci_label"] == "E"]
    es_rows = test_df[test_df["esci_label"].isin(["E", "S"])]
    e_by_q = e_rows.groupby("query")["product_id"].apply(set).to_dict()
    es_by_q = es_rows.groupby("query")["product_id"].apply(set).to_dict()

    return TestSet(queries=queries, e_by_q=e_by_q, es_by_q=es_by_q)


# Metric helpers
def hits_and_mrr(
    retrieved: list[list[str]],
    queries: pd.DataFrame,
    gt_by_q: dict[str, set[str]],
    ks: tuple[int, ...] = (1, 5, 10),
    mrr_k: int = 10,
) -> dict:
    # Counts Hits@K and MRR@mrr_k
    # Queries with no ground-truth in gt_by_q are skipped so the denominator reflects the
    # judged subset rather than the full retrieval batch.
    n = 0
    hits = {k: 0 for k in ks}
    mrr_sum = 0.0
    q_list = queries["query"].tolist()

    for q, row in zip(q_list, retrieved):
        gt = gt_by_q.get(q)
        if not gt:
            continue
        n += 1
        for k in ks:
            if any(pid in gt for pid in row[:k]):
                hits[k] += 1
        rr = 0.0
        for rank, pid in enumerate(row[:mrr_k], start=1):
            if pid in gt:
                rr = 1.0 / rank
                break
        mrr_sum += rr

    if n == 0:
        out = {f"Hits@{k}": 0.0 for k in ks}
        out[f"MRR@{mrr_k}"] = 0.0
        out["n_queries"] = 0
        return out

    out = {f"Hits@{k}": hits[k] / n for k in ks}
    out[f"MRR@{mrr_k}"] = mrr_sum / n
    out["n_queries"] = n
    return out


def format_metrics(label: str, m: dict, strict_label: str = "E") -> str:
    return (
        f"{label:<25}  "
        f"Hits@1={m['Hits@1']:.4f}  "
        f"Hits@5={m['Hits@5']:.4f}  "
        f"Hits@10={m['Hits@10']:.4f}  "
        f"MRR@10={m['MRR@10']:.4f}   "
        f"(GT={strict_label}, n={m['n_queries']:,})"
    )


def evaluate_test_set(
    index: bm25s.BM25,
    pids: np.ndarray,
    top_k: int = 10,
) -> dict:
    ts = load_us_test_set()
    retrieved = search_batch(index, pids, ts.queries["query"].tolist(), top_k=top_k)
    m_strict = hits_and_mrr(retrieved, ts.queries, ts.e_by_q)
    m_lenient = hits_and_mrr(retrieved, ts.queries, ts.es_by_q)
    return {
        "strict_E": m_strict,
        "lenient_ES": m_lenient,
        "n_queries": m_strict["n_queries"],
    }


# Save / load 
def save_index(
    index: bm25s.BM25,
    pids: np.ndarray,
    index_dir: str = INDEX_DIR,
    pids_path: str = PIDS_PATH,
) -> None:
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    Path(pids_path).parent.mkdir(parents=True, exist_ok=True)
    index.save(index_dir)
    np.save(pids_path, pids)


def load_index(
    index_dir: str = INDEX_DIR,
    pids_path: str = PIDS_PATH,
) -> tuple[bm25s.BM25, np.ndarray]:
    return bm25s.BM25.load(index_dir), np.load(pids_path, allow_pickle=True)


def index_exists(
    index_dir: str = INDEX_DIR,
    pids_path: str = PIDS_PATH,
) -> bool:
    return (
        Path(index_dir).is_dir()
        and (Path(index_dir) / "params.index.json").exists()
        and Path(pids_path).exists()
    )
