"""
Image search UI over the product gallery.

Query with text, an uploaded photo, or "more like this" on a result, and pick
which of the six search types answers it:

  same model (CLIP / SigLIP / SigLIP 2)      separate models
    text  -> image                             text  -> text   (MiniLM)
    image -> text                              image -> image  (DINOv2)
    text  -> text
    image -> image
  plus text -> product: MiniLM titles + shared-model photos, score-blended
  (70% photo, see fusion.py and IMPROVEMENTS.md).

Results can be filtered by color and Google-taxonomy category, classified per
product by extract_attributes.py.

Usage:
    python image_search/build_index.py   # once
    python image_search/app.py           # then open http://127.0.0.1:5001
"""
from __future__ import annotations

import os
os.environ.setdefault("TQDM_DISABLE", "1")

import io
import math
import threading
import time

import numpy as np
import pandas as pd
import torch
from flask import Flask, abort, jsonify, request, send_from_directory
from PIL import Image

from encoders import IMAGE_MODELS, MODELS, SHARED_MODELS, TEXT_MODELS, Encoder, device
from attributes import PALETTE, load_attributes
from fusion import PHOTO_WEIGHT, blend
from settings import GALLERY_PARQUET, IMAGE_SEARCH_DIR, IMAGES_DIR, emb_path

TOP_K = 48
# Best shared model in evaluate.py (results/results.md).
DEFAULT_SHARED = "siglip2-b16"
RECALL_K = 100

# search type -> (query modality, target modality, model family, label)
SEARCH_TYPES = {
    "t2i":       ("text",  "image", "shared", "text → image"),
    "t2t":       ("text",  "title", "shared", "text → text"),
    "i2i":       ("image", "image", "shared", "image → image"),
    "i2t":       ("image", "title", "shared", "image → text"),
    "t2t-sep":   ("text",  "title", "text",   "text → text"),
    "i2i-sep":   ("image", "image", "image",  "image → image"),
    "t2p-fused": ("text",  "both",  "fused",  "text → product (fused)"),
}


class ImageSearchEngine:
    def __init__(self) -> None:
        if not GALLERY_PARQUET.exists():
            raise SystemExit("Indexes missing - run `python image_search/build_index.py` first.")
        t0 = time.time()
        self.gallery = pd.read_parquet(GALLERY_PARQUET)
        self.pid_to_row = {pid: i for i, pid in enumerate(self.gallery["product_id"])}
        self.attrs = load_attributes(self.gallery)

        self.encoders: dict[str, Encoder] = {}
        self.emb: dict[tuple[str, str], torch.Tensor] = {}
        for key in MODELS:
            have = {m: emb_path(key, m) for m in ("image", "title") if emb_path(key, m).exists()}
            if not have:
                continue
            print(f"loading {MODELS[key][2]} ...", flush=True)
            self.encoders[key] = Encoder(key)
            for modality, path in have.items():
                self.emb[(key, modality)] = torch.from_numpy(np.load(path)).to(device())
        if not self.encoders:
            raise SystemExit("No embeddings found - run `python image_search/build_index.py` first.")

        self.shared = [k for k in SHARED_MODELS if k in self.encoders]
        self.default_shared = DEFAULT_SHARED if DEFAULT_SHARED in self.shared else next(iter(self.shared), None)
        self.text_model = next((k for k in TEXT_MODELS if k in self.encoders), None)
        self.image_model = next((k for k in IMAGE_MODELS if k in self.encoders), None)
        self.lock = threading.Lock()
        print(f"ready: {len(self.gallery):,} products on {device()} in {time.time()-t0:.0f}s", flush=True)

    def _model_for(self, search_type: str, shared_key: str) -> str:
        family = SEARCH_TYPES[search_type][2]
        key = {"shared": shared_key, "fused": shared_key,
               "text": self.text_model, "image": self.image_model}[family]
        if key is None:
            abort(400, f"no {family} model is indexed")
        return key

    def _embed_query(self, key: str, text: str | None, image: Image.Image | None,
                     pid: str | None, kind: str) -> torch.Tensor:
        if kind == "text":
            return torch.from_numpy(self.encoders[key].encode_text([text])).to(device())
        if pid is not None and (key, "image") in self.emb:
            # "More like this": reuse the product's cached image embedding.
            return self.emb[(key, "image")][self.pid_to_row[pid]][None]
        return torch.from_numpy(self.encoders[key].encode_images([image])).to(device())

    def filter_mask(self, color: str | None, category: str | None) -> torch.Tensor | None:
        """Products matching the filters, or None when there are no filters."""
        if not color and not category:
            return None
        keep = np.ones(len(self.gallery), dtype=bool)
        if color:
            keep &= (self.attrs["color"] == color).to_numpy()
        if category:   # a level-1 name or a full level-2 path
            col = "category" if " > " in category else "category_l1"
            keep &= (self.attrs[col] == category).to_numpy()
        return torch.from_numpy(keep).to(device())

    def facets(self) -> dict:
        return {
            "colors": {c: int(n) for c, n in self.attrs["color"].value_counts().items() if c},
            "categories": {c: int(n) for c, n in self.attrs["category_l1"].value_counts().items() if c},
        }

    @staticmethod
    def _top(s: torch.Tensor, k: int, mask: torch.Tensor | None):
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))
        top = s.topk(min(k, len(s)))
        keep = torch.isfinite(top.values)   # fewer matches than k
        return top.indices[keep].tolist(), top.values[keep].tolist()

    def search(self, search_type: str, shared_key: str, text=None, image=None, pid=None,
               mask: torch.Tensor | None = None) -> dict:
        q_kind, target, family, _ = SEARCH_TYPES[search_type]
        if (q_kind == "text") != (text is not None):
            abort(400, f"{search_type} needs a {q_kind} query")
        key = self._model_for(search_type, shared_key)

        with self.lock:
            if family != "fused":
                q = self._embed_query(key, text, image, pid, q_kind)
                idx, scores = self._top((q @ self.emb[(key, target)].T)[0], TOP_K + 1, mask)
                return {"model": MODELS[key][2], "hits": list(zip(idx, scores))}

            # Separate text model over titles + shared model over photos, score-blended.
            if self.text_model is None:
                abort(400, "no text-only model is indexed")
            s_title = (self._embed_query(self.text_model, text, None, None, "text")
                       @ self.emb[(self.text_model, "title")].T)[0]
            s_photo = (self._embed_query(key, text, None, None, "text") @ self.emb[(key, "image")].T)[0]
            order, scores = self._top(blend(s_title, s_photo), TOP_K + 1, mask)
            # TXT / IMG tags: which side alone would have ranked the product in its top RECALL_K.
            strong = {"text": set(self._top(s_title, RECALL_K, mask)[0]),
                      "image": set(self._top(s_photo, RECALL_K, mask)[0])}
        return {
            "model": f"{MODELS[self.text_model][2]} + {MODELS[key][2]}",
            "hits": list(zip(order, scores)),
            "sources": {i: [s for s, ids in strong.items() if i in ids] for i in order},
        }

    def product(self, i: int) -> dict:
        row = self.gallery.iloc[i]
        price = row["price"]
        if price is None or (isinstance(price, float) and math.isnan(price)) or price <= 0 or price > 100_000:
            price = None
        return {
            "id": row["product_id"],
            "title": row["title"],
            "merchant": row["merchant"],
            "price": price,
            "url": row["url"],
            "image": f"/images/{row['local_image']}",
            "ai_description": row["ai_description"],
            **self.attrs.iloc[i][["color", "color_name", "category"]].to_dict(),
        }


app = Flask(__name__, static_folder=str(IMAGE_SEARCH_DIR / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.json.sort_keys = False  # keep SEARCH_TYPES in display order
engine: ImageSearchEngine | None = None


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/images/<path:name>")
def image(name: str):
    return send_from_directory(IMAGES_DIR, name, max_age=86400)


@app.get("/api/status")
def status():
    return jsonify({
        "products": len(engine.gallery),
        "device": device(),
        "shared_models": [{"key": k, "label": MODELS[k][2]} for k in engine.shared],
        "default_model": engine.default_shared,
        "text_model": engine.text_model and MODELS[engine.text_model][2],
        "image_model": engine.image_model and MODELS[engine.image_model][2],
        "photo_weight": PHOTO_WEIGHT,
        "palette": list(PALETTE),
        "facets": engine.facets(),
        "search_types": {k: {"query": v[0], "family": v[2], "label": v[3]} for k, v in SEARCH_TYPES.items()},
    })


@app.post("/api/search")
def api_search():
    search_type = request.form.get("type", "t2i")
    if search_type not in SEARCH_TYPES:
        abort(400, f"type must be one of {list(SEARCH_TYPES)}")
    shared_key = request.form.get("model") or engine.default_shared
    if SEARCH_TYPES[search_type][2] in ("shared", "fused") and shared_key not in engine.shared:
        abort(400, f"model must be one of {engine.shared}")

    text = (request.form.get("q") or "").strip()[:300] or None
    pid = request.form.get("pid") or None
    img = None
    if pid is not None:
        if pid not in engine.pid_to_row:
            abort(404, "unknown product")
        img = Image.open(IMAGES_DIR / engine.gallery["local_image"].iloc[engine.pid_to_row[pid]]).convert("RGB")
    elif "image" in request.files:
        try:
            img = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
        except Exception:
            abort(400, "could not read that image")
    if SEARCH_TYPES[search_type][0] == "image":
        text = None
        if img is None:
            abort(400, "upload an image or pick a product for an image search")

    color = request.form.get("color") or None
    category = request.form.get("category") or None
    if color and color not in PALETTE:
        abort(400, f"color must be one of {list(PALETTE)}")

    t0 = time.perf_counter()
    res = engine.search(search_type, shared_key, text=text, image=img, pid=pid,
                        mask=engine.filter_mask(color, category))
    elapsed = time.perf_counter() - t0

    sources = res.get("sources", {})
    results = []
    for i, score in res["hits"]:
        p = engine.product(i)
        if p["id"] == pid:  # don't return the query product itself
            continue
        p["score"] = round(float(score), 4)
        if sources:
            p["sources"] = sources[i]
        results.append(p)
    return jsonify({
        "type": search_type,
        "model": res["model"],
        "results": results[:TOP_K],
        "ms": round(elapsed * 1000, 1),
        "query_product": engine.product(engine.pid_to_row[pid]) if pid else None,
    })


if __name__ == "__main__":
    engine = ImageSearchEngine()
    port = int(os.environ.get("PORT", 5001))
    print(f"open http://127.0.0.1:{port}", flush=True)
    app.run(host="127.0.0.1", port=port, threaded=True)
