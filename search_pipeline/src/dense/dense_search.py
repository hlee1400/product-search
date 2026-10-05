from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import re
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.bm25.bm25_search import (
    format_metrics,
    hits_and_mrr,
    load_us_corpus,
    load_us_test_set,
)
from src.config.paths import dense_emb_path, dense_pids_path, slug  # noqa: F401


#DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Cache paths are keyed by DEFAULT_MODEL via paths.py. Swap DEFAULT_MODEL
# above and each model gets its own cache file
EMB_PATH = str(dense_emb_path(DEFAULT_MODEL))
PIDS_PATH = str(dense_pids_path(DEFAULT_MODEL))


#GPU or CPU or Apple Silicon
def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str = DEFAULT_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_id, device=device())


def embed_corpus(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 128,
    show_progress_bar: bool = True,
) -> np.ndarray:
    # normalize_embeddings=True so that inner-product on the FAISS index is
    # equivalent to cosine similarity. float32 to keep faiss-cpu happy.
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    ).astype("float32")
    return emb


def build_faiss_index(emb: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return q.astype("float32")


def search(
    model: SentenceTransformer,
    index: faiss.Index,
    pids: np.ndarray,
    query: str,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    q_emb = embed_query(model, query)
    scores, idxs = index.search(q_emb, top_k)
    return [(str(pids[idxs[0][i]]), float(scores[0][i])) for i in range(top_k)]


def search_batch(
    model: SentenceTransformer,
    index: faiss.Index,
    pids: np.ndarray,
    queries: list[str],
    top_k: int = 10,
    show_progress_bar: bool = False,
) -> list[list[str]]:
    q_emb = model.encode(
        queries,
        batch_size=256,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    ).astype("float32")
    scores, idxs = index.search(q_emb, top_k)
    return [[str(pids[idxs[i][k]]) for k in range(top_k)] for i in range(len(queries))]


def evaluate_test_set(
    model: SentenceTransformer,
    index: faiss.Index,
    pids: np.ndarray,
    top_k: int = 10,
) -> dict:
    ts = load_us_test_set()
    retrieved = search_batch(model, index, pids, ts.queries["query"].tolist(), top_k=top_k)
    m_strict = hits_and_mrr(retrieved, ts.queries, ts.e_by_q)
    m_lenient = hits_and_mrr(retrieved, ts.queries, ts.es_by_q)
    return {
        "strict_E": m_strict,
        "lenient_ES": m_lenient,
        "n_queries": m_strict["n_queries"],
    }


def save_index(
    emb: np.ndarray,
    pids: np.ndarray,
    emb_path: str = EMB_PATH,
    pids_path: str = PIDS_PATH,
) -> None:
    Path(emb_path).parent.mkdir(parents=True, exist_ok=True)
    Path(pids_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    np.save(pids_path, pids)


def load_index(
    emb_path: str = EMB_PATH,
    pids_path: str = PIDS_PATH,
) -> tuple[faiss.Index, np.ndarray]:
    emb = np.load(emb_path)
    pids = np.load(pids_path, allow_pickle=True)
    return build_faiss_index(emb), pids


def index_exists(
    emb_path: str = EMB_PATH,
    pids_path: str = PIDS_PATH,
) -> bool:
    return Path(emb_path).exists() and Path(pids_path).exists()


__all__ = [
    "DEFAULT_MODEL",
    "EMB_PATH",
    "PIDS_PATH",
    "device",
    "load_model",
    "embed_corpus",
    "build_faiss_index",
    "embed_query",
    "search",
    "search_batch",
    "evaluate_test_set",
    "save_index",
    "load_index",
    "index_exists",
    "load_us_corpus",
    "format_metrics",
]
