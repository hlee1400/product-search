import time

from src.config.paths import (
    create_output_dirs,
    PRODUCTS_PROCESSED_CSV,
    QUERIES_PROCESSED_CSV,
    RELEVANCE_JUDGMENTS_CSV,
)

from src.data.load_data import (
    load_task_data,
    load_examples,
    load_products,
    load_sources,
)

from src.data.inspect_data import inspect_raw_dataset

from src.bm25.bm25_search import (
    INDEX_DIR,
    PIDS_PATH,
    build_index,
    evaluate_test_set,
    format_metrics,
    index_exists,
    load_index,
    load_us_corpus,
    save_index,
    search,
)

from src.dense.dense_search import (
    DEFAULT_MODEL as DENSE_MODEL,
    EMB_PATH as DENSE_EMB_PATH,
    PIDS_PATH as DENSE_PIDS_PATH,
    build_faiss_index as dense_build_faiss_index,
    device as dense_device,
    embed_corpus as dense_embed_corpus,
    evaluate_test_set as dense_evaluate_test_set,
    index_exists as dense_index_exists,
    load_index as dense_load_index,
    load_model as dense_load_model,
    save_index as dense_save_index,
    search as dense_search,
)

from src.hybrid.hybrid_search import (
    DEFAULT_RERANKER,
    RECALL_K,
    TOP_K,
    build_pid_to_title as hybrid_build_pid_to_title,
    evaluate_test_set as hybrid_evaluate_test_set,
    load_cross_encoder as hybrid_load_cross_encoder,
    run_pipeline as hybrid_run_pipeline,
)


# Variety of queries to probe failure modes 
SAMPLE_QUERIES = [
    "rose gold iphone case",            # color + brand + category
    "storage bin 52qt with lid",        # numeric spec + accessory
    "wheelbarrow",                      # single-token
    "champion sweatshirt women",        # brand-prominent
    "car charger",                      # synonym test (vs. "auto USB power adapter")
    "aeropress",                        # rare brand / single token
]


def run_bm25_pipeline() -> None:
    # ── 1. Build (or reuse) the index ──────────────────────────────
    if index_exists(INDEX_DIR, PIDS_PATH):
        print(f"loading cached bm25 index from {INDEX_DIR} ...")
        idx, pids = load_index(INDEX_DIR, PIDS_PATH)
    else:
        print("building bm25 index from parquet ...")
        df = load_us_corpus()
        print(f"  corpus: {len(df):,} us products")
        t0 = time.time()
        idx = build_index(df["product_title"].tolist(), k1=1.2, b=0.75, show_progress=False)
        print(f"  built index in {time.time()-t0:.1f}s")
        pids = df["product_id"].to_numpy()
        save_index(idx, pids, INDEX_DIR, PIDS_PATH)
        print(f"saved -> {INDEX_DIR}, {PIDS_PATH}")
    print(f"  index has {len(pids):,} products\n")

    # ── 2. Sample searches ─────────────────────────────────────────
    for query in SAMPLE_QUERIES:
        print(f"top-10 for: {query!r}")
        for rank, (pid, score) in enumerate(search(idx, pids, query, top_k=10), 1):
            print(f"  {rank:>2}. score={score:.3f}  {pid}")
        print()

    # ── 3. Eval on the US small test split ─────────────────────────
    print("evaluating on US small-version test split ...")
    t0 = time.time()
    out = evaluate_test_set(idx, pids, top_k=10)
    print(f"eval took {time.time()-t0:.1f}s")
    print(f"  ran on {out['n_queries']:,} queries\n")
    print(format_metrics("BM25 strict (E)",        out["strict_E"]))
    print(format_metrics("BM25 lenient (E or S)",  out["lenient_ES"], strict_label="E or S"))


def run_dense_pipeline() -> None:
    # ── 1. Load the embedding model ────────────────────────────────
    print(f"device: {dense_device()}")
    print(f"model:  {DENSE_MODEL}")
    model = dense_load_model(DENSE_MODEL)

    # ── 2. Build (or reuse) the FAISS index ────────────────────────
    if dense_index_exists(DENSE_EMB_PATH, DENSE_PIDS_PATH):
        print(f"loading cached dense embeddings from {DENSE_EMB_PATH} ...")
        index, pids = dense_load_index(DENSE_EMB_PATH, DENSE_PIDS_PATH)
    else:
        print("embedding corpus from parquet ...")
        df = load_us_corpus()
        print(f"  corpus: {len(df):,} us products")
        t0 = time.time()
        emb = dense_embed_corpus(
            model, df["product_title"].tolist(),
            batch_size=128, show_progress_bar=False,
        )
        print(f"  embedded in {time.time()-t0:.0f}s, shape={emb.shape}")
        pids = df["product_id"].to_numpy()
        dense_save_index(emb, pids, DENSE_EMB_PATH, DENSE_PIDS_PATH)
        print(f"saved -> {DENSE_EMB_PATH}, {DENSE_PIDS_PATH}")
        index = dense_build_faiss_index(emb)
    print(f"  index has {len(pids):,} products\n")

    # ── 3. Sample searches ─────────────────────────────────────────
    for query in SAMPLE_QUERIES:
        print(f"top-10 for: {query!r}")
        for rank, (pid, score) in enumerate(dense_search(model, index, pids, query, top_k=10), 1):
            print(f"  {rank:>2}. score={score:.3f}  {pid}")
        print()

    # ── 4. Eval on the US small test split ─────────────────────────
    print("evaluating on US small-version test split ...")
    t0 = time.time()
    out = dense_evaluate_test_set(model, index, pids, top_k=10)
    print(f"eval took {time.time()-t0:.1f}s")
    print(f"  ran on {out['n_queries']:,} queries\n")
    print(format_metrics("dense strict (E)",        out["strict_E"]))
    print(format_metrics("dense lenient (E or S)",  out["lenient_ES"], strict_label="E or S"))


def run_hybrid_pipeline() -> None:
    # ── 1. Load (cached) BM25 + dense indexes ──────────────────────
    if not index_exists(INDEX_DIR, PIDS_PATH):
        raise RuntimeError(
            "BM25 index missing; run_bm25_pipeline() must run first to build it."
        )
    if not dense_index_exists(DENSE_EMB_PATH, DENSE_PIDS_PATH):
        raise RuntimeError(
            "Dense index missing; run_dense_pipeline() must run first to build it."
        )

    print(f"loading cached bm25 index from {INDEX_DIR} ...")
    bm25_index, bm25_pids = load_index(INDEX_DIR, PIDS_PATH)
    print(f"loading cached dense embeddings from {DENSE_EMB_PATH} ...")
    dense_faiss, dense_pids = dense_load_index(DENSE_EMB_PATH, DENSE_PIDS_PATH)

    # ── 2. Load dense model + cross-encoder ────────────────────────
    print(f"dense model:    {DENSE_MODEL}")
    print(f"reranker model: {DEFAULT_RERANKER}")
    dense_model = dense_load_model(DENSE_MODEL)
    cross_encoder = hybrid_load_cross_encoder(DEFAULT_RERANKER)

    # ── 3. product_id -> title (for rerank input) ──────────────────
    print("building pid -> title map from parquet ...")
    pid_to_title = hybrid_build_pid_to_title()
    print(f"  {len(pid_to_title):,} titles\n")

    # ── 4. Sample queries end-to-end ───────────────────────────────
    for query in SAMPLE_QUERIES:
        print(f"hybrid top-{TOP_K} for: {query!r}")
        out = hybrid_run_pipeline(
            query,
            bm25_index, bm25_pids,
            dense_model, dense_faiss, dense_pids,
            cross_encoder,
            pid_to_title,
            recall_k=RECALL_K, top_k=TOP_K,
        )
        for rank, pid in enumerate(out["reranked"], 1):
            print(f"  {rank:>2}. {pid}  {pid_to_title.get(pid, '')[:90]}")
        print()

    # ── 5. Eval on the US small test split ─────────────────────────
    print("evaluating hybrid stages on US small-version test split ...")
    t0 = time.time()
    metrics = hybrid_evaluate_test_set(
        bm25_index, bm25_pids,
        dense_model, dense_faiss, dense_pids,
        cross_encoder,
        pid_to_title,
        recall_k=RECALL_K, top_k=TOP_K,
    )
    print(f"eval took {time.time()-t0:.0f}s\n")
    for key in ("dense", "bm25", "rrf", "reranked"):
        m = metrics[key]
        print(f"--- {m['name']} ---")
        print(format_metrics("  strict (E)",       m["strict_E"]))
        print(format_metrics("  lenient (E or S)", m["lenient_ES"], strict_label="E or S"))
        print()


def main() -> None:
    create_output_dirs()

    examples = load_examples()
    products = load_products()
    sources = load_sources()

    inspect_raw_dataset(
        examples=examples,
        products=products,
        sources=sources,
    )

    run_bm25_pipeline()
    run_dense_pipeline()
    run_hybrid_pipeline()


if __name__ == "__main__":
    main()
