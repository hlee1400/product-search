# Product Search UI

A web search page over the local catalog in `../data`, powered by the retrieval
pipeline in `../search_pipeline` (imported as-is, not copied):

```
query ─┬─ BM25 top-50  (bm25_search.search)        ┐
       └─ dense top-50 (dense_search.search_batch) ┴─ rrf_fuse ─ cross-encoder rerank ─ results
```

## Run

```powershell
python search_app/build_index.py   # one-time: ~1 min BM25 + a few min embeddings on GPU
python search_app/app.py           # http://127.0.0.1:5000
```

`build_index.py --rebuild` regenerates everything. Indexes go to `search_app/artifacts/`.

## What gets indexed

- **Corpus:** `product_catalog_10_17_24.csv` (423k products), joined with the
  `*_enriched_descriptions.csv` files for the AI summary shown in the detail panel.
- **Retrieval text:** product title only, following the pipeline's title-only design.
  `Amazon.com:` prefixes and `- Etsy` suffixes are stripped first.
- **Images:** served from `data/product_images/product_images` when present, otherwise
  the product's remote image URL.

## UI

- Mode switch: **Hybrid + Rerank** (full pipeline), **RRF**, **BM25**, **Dense**. All four
  stage lists are computed on every query, so you can compare them directly.
- `KW` / `SEM` badges on each card show whether BM25 and/or dense retrieval found it.
- Click a card for the pipeline trace (its rank at each stage), description, and store link.
- Merchant filter chips, `/` to focus search, and URL state (`?q=…&mode=…`).

## Swapping models

The embedding model is `DEFAULT_MODEL` in `src/dense/dense_search.py` (currently
`all-MiniLM-L6-v2`; the README results favour `BAAI/bge-small-en-v1.5`). Change it and
re-run `build_index.py`; the dense cache is keyed by model name, so both can coexist.
