# Product Search

A multi-stage hybrid search engine over a 423k-product e-commerce catalog, with a web UI
that shows how each retrieval stage contributes to the final ranking.

![Search results page](search_app/docs/results.png)

## How it works

```
                ┌─ BM25 (keyword) top-50 ──┐
query ──────────┤                          ├─ Reciprocal Rank Fusion ─ cross-encoder rerank ─ results
                └─ dense (FAISS) top-50 ───┘
```

| Stage | Model / method | What it's good at |
| --- | --- | --- |
| BM25 | `bm25s` over product titles | Exact brands, model numbers, specs |
| Dense | `all-MiniLM-L6-v2` + FAISS `IndexFlatIP` | Synonyms, paraphrases, category-level intent |
| RRF | `Σ 1 / (60 + rank)` | Merges the two candidate pools, whose failure modes barely overlap |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Jointly scores (query, title) to fix the final order |

On the Amazon ESCI benchmark (8,955 judged queries, 1.2M products), the full pipeline
improves Hits@1 from 0.296 (BM25 alone) to **0.374** and MRR@10 from 0.385 to **0.474**.
See [the pipeline write-up](search_pipeline/README.md) for the experiments,
error analysis, and design decisions.

Warm queries take about 110 ms end to end on a laptop GPU (RTX 5050).

## The search UI

- **Stage switcher:** rank by the full pipeline, RRF only, BM25 only, or dense only, so
  you can compare the stages on the same query.
- **Retrieval badges:** `KW` / `SEM` on each result show which retriever found it.
- **Pipeline trace:** the detail panel shows the product's rank at every stage.
- Per-stage latency, store filters, dark mode, and a responsive layout.

## Repo layout

```
search_pipeline/               retrieval pipeline + offline evaluation on ESCI
  src/bm25/                    BM25 retriever
  src/dense/                   sentence-transformer + FAISS retriever
  src/hybrid/                  RRF fusion + cross-encoder rerank
search_app/                    web UI over a product catalog, built on the pipeline
  build_index.py               builds BM25 + dense indexes from the catalog CSVs
  app.py                       Flask API server
  static/index.html            search page
```

## Running it

```powershell
pip install bm25s pandas pyarrow numpy faiss-cpu sentence-transformers torch flask
```

The product catalog is not included in this repo. Put it in `data/`:

```
data/
├── product_catalog_10_17_24.csv        # required: one row per product
├── *_enriched_descriptions.csv         # optional: AI summaries for the detail panel
└── product_images/product_images/      # optional: local images, else remote URLs are used
```

Then:

```powershell
python search_app/build_index.py   # one-time: ~2 min on GPU
python search_app/app.py           # open http://127.0.0.1:5000
```

To reproduce the ESCI evaluation, see
[search_pipeline/README.md](search_pipeline/README.md#reproducing-the-results).

## License

MIT
