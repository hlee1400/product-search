## Repo layout

```
search_pipeline/
├── README.md                       ← this file
├── esci-data/                      ← ESCI dataset (parquet + sources CSV)
│   ├── shopping_queries_dataset/
│   │   ├── shopping_queries_dataset_examples.parquet
│   │   ├── shopping_queries_dataset_products.parquet
│   │   └── shopping_queries_dataset_sources.csv
│   ├── classification_identification/   ← unused here
│   └── ranking/                         ← unused here
├── src/
│   ├── main.py                     ← orchestrator
│   ├── run_eval.py                 ← clean-log runner entrypoint
│   ├── config/
│   │   └── paths.py                ← all paths + model-keyed cache naming
│   ├── data/
│   │   ├── load_data.py            ← parquet loaders + train/test split
│   │   └── inspect_data.py         ← EDA prints used at startup
│   ├── bm25/
│   │   └── bm25_search.py          ← lexical retriever + metrics + test-set loader
│   ├── dense/
│   │   └── dense_search.py         ← sentence-transformer + FAISS dense retriever
│   └── hybrid/
│       └── hybrid_search.py        ← RRF fuse + cross-encoder rerank
└── data/
    └── artifacts/
        ├── bm25_us/                ← saved BM25 index (vocab + CSR shards)
        ├── bm25_pids_us.npy        ← row → product_id mapping for BM25
        ├── dense_emb_us_*.npy      ← model-keyed corpus embedding cache
        ├── dense_pids_us_*.npy     ← row → product_id for the dense cache
        └── logs/                   ← timestamped per-run logs from run_eval.py
```

