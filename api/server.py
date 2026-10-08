"""
Product search API: text, photo and "more like this" search over the bundled
index, with color / category filters.

  GET  /health                         liveness (no key needed)
  GET  /v1/facets                      colors, categories, index info
  POST /v1/search/text                 {"query": "red summer dress", ...}
  POST /v1/search/image                multipart photo upload
  GET  /v1/items/{product_id}          one item
  GET  /v1/items/{product_id}/similar  items with similar photos

Every /v1 route needs the API key, as `Authorization: Bearer <key>` or
`X-API-Key: <key>`. The key comes from the API_KEY environment variable (a Fly
secret in production); the server won't start without one unless
ALLOW_NO_AUTH=1 is set for local development.

Run locally:
    python api/export_index.py
    set ALLOW_NO_AUTH=1  (or API_KEY=...)
    uvicorn api.server:app --port 8080
"""
from __future__ import annotations

import io
import json
import os
import secrets
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "image_search"))   # same encoders + fusion as the benchmarks
sys.path.insert(0, str(HERE))

from encoders import Encoder  # noqa: E402
from fusion import blend  # noqa: E402

BUNDLE = Path(os.environ.get("BUNDLE_DIR", HERE / "bundle"))
PHOTO_BASE_URL = os.environ.get("PHOTO_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
if not API_KEY and os.environ.get("ALLOW_NO_AUTH") != "1":
    raise SystemExit("Set API_KEY (or ALLOW_NO_AUTH=1 for local development).")

def cpu_limit() -> int:
    """CPUs this process may actually use. Under a container CPU quota, os.cpu_count()
    still reports every host core; torch then starts one thread per core, and on a
    2-CPU quota of a 16-core host that made each query ~170x slower (18 s vs 0.1 s)."""
    if os.environ.get("TORCH_THREADS"):
        return int(os.environ["TORCH_THREADS"])
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1


MAX_LIMIT = 100
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


class Index:
    """The read-only index, held in memory: ~0.1 GB of vectors + metadata at 24k items."""

    def __init__(self) -> None:
        t0 = time.time()
        self.manifest = json.loads((BUNDLE / "manifest.json").read_text())
        self.items = pd.read_parquet(BUNDLE / "items.parquet")
        self.row = {pid: i for i, pid in enumerate(self.items["product_id"])}
        self.photo = torch.from_numpy(np.load(BUNDLE / "photo_vectors.npy").astype(np.float32))
        self.title = torch.from_numpy(np.load(BUNDLE / "title_vectors.npy").astype(np.float32))
        self.photo_model = Encoder("siglip2-b16")
        self.title_model = Encoder("minilm")
        self.color = self.items["color"].to_numpy()
        self.category = self.items["category"].to_numpy()
        self.category_l1 = self.items["category"].str.split(" > ").str[0].fillna("").to_numpy()
        # One request at a time through the models: CPU inference already uses every core.
        self.lock = threading.Lock()
        print(f"index loaded: {len(self.items):,} items in {time.time() - t0:.0f}s", flush=True)

        # Warm-up: on Fly the first pass through each model took ~20 s (weights are
        # memory-mapped and paged in from disk, and CPU kernels initialize lazily).
        # Paying that here, before the server accepts traffic, keeps it off the first user.
        t0 = time.time()
        self.photo_model.encode_text(["warm up"])
        self.photo_model.encode_images([Image.new("RGB", (224, 224))])
        self.title_model.encode_text(["warm up"])
        print(f"warm-up done in {time.time() - t0:.0f}s", flush=True)

    def text_scores(self, query: str, title: bool = True, photo: bool = True):
        """Similarity of a text query to every item's title (MiniLM) and photo (SigLIP 2)."""
        with self.lock:
            s_photo = torch.from_numpy(self.photo_model.encode_text([query]))[0] @ self.photo.T if photo else None
            s_title = torch.from_numpy(self.title_model.encode_text([query]))[0] @ self.title.T if title else None
        return s_title, s_photo

    def image_scores(self, img: Image.Image) -> torch.Tensor:
        with self.lock:
            q = torch.from_numpy(self.photo_model.encode_images([img]))[0]
        return q @ self.photo.T

    def mask(self, color: str | None, category: str | None) -> torch.Tensor | None:
        if not color and not category:
            return None
        keep = np.ones(len(self.items), dtype=bool)
        if color:
            keep &= self.color == color
        if category:   # a level-1 name or a full level-2 path
            keep &= (self.category if " > " in category else self.category_l1) == category
        return torch.from_numpy(keep)

    @staticmethod
    def top(scores: torch.Tensor, k: int, mask: torch.Tensor | None, exclude: int | None = None):
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        if exclude is not None:
            scores = scores.clone()
            scores[exclude] = float("-inf")
        top = scores.topk(min(k, len(scores)))
        keep = torch.isfinite(top.values)
        return list(zip(top.indices[keep].tolist(), top.values[keep].tolist()))

    def item(self, i: int, score: float | None = None) -> dict:
        r = self.items.iloc[i]
        out = {
            "id": r["product_id"], "title": r["title"], "merchant": r["merchant"],
            "price": None if pd.isna(r["price"]) else float(r["price"]), "url": r["url"],
            "image": f"{PHOTO_BASE_URL}/{r['photo_key']}" if PHOTO_BASE_URL else r["retailer_image_url"],
            "color": r["color"] or None, "color_name": r["color_name"] or None,
            "category": r["category"] or None,
        }
        if score is not None:
            out["score"] = round(float(score), 4)
        return out


index: Index | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global index
    torch.set_grad_enabled(False)
    torch.set_num_threads(cpu_limit())
    print(f"torch threads: {torch.get_num_threads()}", flush=True)
    index = Index()   # models + vectors load before the first request is accepted
    yield


app = FastAPI(title="Product Search API", version="1.0", lifespan=lifespan,
              description="Text and photo search over a product catalog. See api/README.md.")


# Declared as security schemes (not plain headers) so /docs shows an "Authorize" button.
bearer = HTTPBearer(auto_error=False, description="Authorization: Bearer <API key>")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="X-API-Key: <API key>")


def require_key(creds: HTTPAuthorizationCredentials | None = Depends(bearer),
                x_api_key: str | None = Depends(api_key_header)) -> None:
    if not API_KEY:
        return
    given = x_api_key or (creds.credentials if creds else "")
    if not secrets.compare_digest(given, API_KEY):
        raise HTTPException(401, "missing or invalid API key")


def check_filters(color: str | None, category: str | None) -> None:
    if color and color not in index.manifest["palette"]:
        raise HTTPException(422, f"color must be one of {index.manifest['palette']}")
    if category and not ((index.category == category).any() or (index.category_l1 == category).any()):
        raise HTTPException(422, "unknown category; see /v1/facets")


class Results(BaseModel):
    results: list[dict]
    took_ms: float
    mode: str


class TextQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=300, examples=["red summer dress"])
    mode: Literal["blend", "photo", "title"] = Field(
        "blend", description="blend: titles + photos (default); photo: SigLIP 2 over photos only; "
                             "title: MiniLM over titles only")
    limit: int = Field(20, ge=1, le=MAX_LIMIT)
    color: str | None = None
    category: str | None = Field(None, description="level-1 name or level-2 path")


LANDING = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Product Search API</title>
<style>
  :root { --bg: #f6f6f4; --fg: #1b1b19; --muted: #6d6b65; --card: #fff; --border: #e4e2dd; --accent: #2f5bea; }
  @media (prefers-color-scheme: dark) { :root { --bg: #121211; --fg: #ecebe7; --muted: #9b9991; --card: #1c1c1a; --border: #33322f; --accent: #7c9cff; } }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.55 system-ui, -apple-system, Segoe UI, sans-serif; }
  main { max-width: 720px; margin: 0 auto; padding: 48px 16px; }
  h1 { margin: 0 0 6px; letter-spacing: -.02em; } p { color: var(--muted); }
  a { color: var(--accent); }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px 20px; margin: 18px 0; }
  code, pre { font: 13px ui-monospace, Consolas, monospace; } pre { overflow-x: auto; white-space: pre; margin: 0; }
  table { border-collapse: collapse; width: 100%; font-size: 14px; } td { padding: 6px 8px; border-top: 1px solid var(--border); vertical-align: top; }
  .btn { display: inline-block; background: var(--accent); color: #fff; text-decoration: none; padding: 8px 16px; border-radius: 999px; font-weight: 600; }
</style></head><body><main>
<h1>Product Search API</h1>
<p>Text and photo search over {items} products, with color and category filters.
Search uses SigLIP 2 over product photos blended with MiniLM over titles.</p>
<p><a class="btn" href="/docs">Try it in the interactive docs</a> &nbsp; <a href="/">search page</a> &nbsp; <a href="/health">health check</a></p>
<div class="card"><table>
<tr><td><code>POST /v1/search/text</code></td><td>search by a text query</td></tr>
<tr><td><code>POST /v1/search/image</code></td><td>search by an uploaded photo</td></tr>
<tr><td><code>GET /v1/items/{{id}}/similar</code></td><td>products with similar photos</td></tr>
<tr><td><code>GET /v1/items/{{id}}</code></td><td>one product</td></tr>
<tr><td><code>GET /v1/facets</code></td><td>available colors and categories</td></tr>
</table></div>
<p>Every <code>/v1</code> endpoint needs an API key. In the interactive docs, click <b>Authorize</b>
and paste it; from code, send it as a header:</p>
<div class="card"><pre>curl {base}/v1/search/text \\
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \\
  -d '{{"query": "red summer dress", "limit": 5}}'</pre></div>
</main></body></html>"""


@app.get("/about", response_class=HTMLResponse, include_in_schema=False)
def landing(request: Request) -> str:
    base = str(request.base_url).rstrip("/").replace("http://", "https://")
    # Plain substitution: the page's CSS braces would trip str.format.
    page = LANDING.replace("{{", "{").replace("}}", "}")
    return page.replace("{items}", f"{len(index.items):,}" if index else "").replace("{base}", base)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "items": len(index.items) if index else 0}


@app.get("/v1/facets", dependencies=[Depends(require_key)])
def facets() -> dict:
    s = pd.Series
    return {
        "index": index.manifest,
        "colors": {k: int(v) for k, v in s(index.color).value_counts().items() if k},
        "categories": {k: int(v) for k, v in s(index.category_l1).value_counts().items() if k},
        "subcategories": {k: int(v) for k, v in s(index.category).value_counts().items() if k},
    }


@app.post("/v1/search/text", response_model=Results, dependencies=[Depends(require_key)])
def search_text(q: TextQuery) -> Results:
    check_filters(q.color, q.category)
    t0 = time.perf_counter()
    s_title, s_photo = index.text_scores(q.query, title=q.mode != "photo", photo=q.mode != "title")
    scores = blend(s_title, s_photo) if q.mode == "blend" else (s_photo if q.mode == "photo" else s_title)
    hits = index.top(scores, q.limit, index.mask(q.color, q.category))
    return Results(results=[index.item(i, s) for i, s in hits], mode=q.mode,
                   took_ms=round((time.perf_counter() - t0) * 1000, 1))


@app.post("/v1/search/image", response_model=Results, dependencies=[Depends(require_key)])
def search_image(image: UploadFile = File(...), limit: int = Form(20, ge=1, le=MAX_LIMIT),
                 color: str | None = Form(None), category: str | None = Form(None)) -> Results:
    check_filters(color, category)
    img = read_upload(image)
    t0 = time.perf_counter()
    hits = index.top(index.image_scores(img), limit, index.mask(color, category))
    return Results(results=[index.item(i, s) for i, s in hits], mode="image",
                   took_ms=round((time.perf_counter() - t0) * 1000, 1))


def read_upload(upload: UploadFile) -> Image.Image:
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image larger than 8 MB")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(422, "could not read that image")


def row_of(product_id: str) -> int:
    if product_id not in index.row:
        raise HTTPException(404, "unknown product id")
    return index.row[product_id]


@app.get("/v1/items/{product_id}", dependencies=[Depends(require_key)])
def get_item(product_id: str) -> dict:
    return index.item(row_of(product_id))


@app.get("/v1/items/{product_id}/similar", response_model=Results, dependencies=[Depends(require_key)])
def similar(product_id: str, limit: int = Query(20, ge=1, le=MAX_LIMIT),
            color: str | None = None, category: str | None = None) -> Results:
    check_filters(color, category)
    i = row_of(product_id)
    t0 = time.perf_counter()
    hits = index.top(index.photo[i] @ index.photo.T, limit, index.mask(color, category), exclude=i)
    return Results(results=[index.item(j, s) for j, s in hits], mode="similar",
                   took_ms=round((time.perf_counter() - t0) * 1000, 1))


# The password-protected search page and the endpoints it calls (api/web.py).
from web import create_router  # noqa: E402

app.include_router(create_router(lambda: index, read_upload))
