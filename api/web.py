"""
The hosted search page: the same page as the local app (image_search/static/
index.html), behind a site password.

  GET  /login, POST /login, GET /logout   site-password login
  GET  /                                  the search page
  GET  /api/status, POST /api/search      what the page calls (same shapes as
                                          image_search/app.py, so one page serves both)

The page never sees the API key: it calls these endpoints with a signed session
cookie, and they run the search in-process. SITE_PASSWORD is a Fly secret. The
cookie is signed with a key derived from the password and the API key, so
changing the password logs everyone out.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from encoders import MODELS
from fusion import PHOTO_WEIGHT, blend

PAGE = Path(__file__).resolve().parent.parent / "image_search" / "static" / "index.html"
SITE_PASSWORD = os.environ.get("SITE_PASSWORD", "")
COOKIE = "ps_session"
SESSION_DAYS = 30
TOP_K = 48
RECALL_K = 100   # TXT / IMG tags: was the result in that side's own top 100?

# The search types the deployed models support (SigLIP 2 + MiniLM), in display order.
# key -> (query kind, family, label); families match the local page's groups.
SEARCH_TYPES = {
    "t2p-fused": ("text", "fused", "text → product (blend)"),
    "t2i": ("text", "shared", "text → image"),
    "t2t-sep": ("text", "text", "text → text"),
    "i2i": ("image", "shared", "image → image"),
}

_KEY = hashlib.sha256(f"{SITE_PASSWORD}|{os.environ.get('API_KEY', '')}".encode()).digest()


def _sign(expires: int) -> str:
    return hmac.new(_KEY, str(expires).encode(), hashlib.sha256).hexdigest()


def _authed(request: Request) -> bool:
    if not SITE_PASSWORD:
        return os.environ.get("ALLOW_NO_AUTH") == "1"
    value = request.cookies.get(COOKIE, "")
    expires, _, sig = value.partition(".")
    return expires.isdigit() and int(expires) > time.time() and hmac.compare_digest(sig, _sign(int(expires)))


LOGIN = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image Search</title>
<style>
  :root { --bg: #f6f6f4; --fg: #1b1b19; --muted: #6d6b65; --card: #fff; --border: #e4e2dd; --accent: #2f5bea; --img: #2b7a5b; --err: #c4352b; }
  @media (prefers-color-scheme: dark) { :root { --bg: #121211; --fg: #ecebe7; --muted: #9b9991; --card: #1c1c1a; --border: #33322f; --accent: #7c9cff; --img: #4cc38a; --err: #ff8a80; } }
  * { box-sizing: border-box; }
  body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: var(--bg); color: var(--fg);
         font: 15px/1.5 Inter, system-ui, -apple-system, Segoe UI, sans-serif; padding: 16px; }
  form { width: min(360px, 100%); background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 28px; }
  h1 { margin: 0 0 4px; font-size: 22px; letter-spacing: -.02em; } h1 span { color: var(--img); }
  p { margin: 0 0 18px; color: var(--muted); font-size: 14px; }
  input { width: 100%; padding: 10px 12px; font: inherit; border: 1px solid var(--border); border-radius: 10px;
          background: var(--bg); color: var(--fg); outline: none; }
  input:focus { border-color: var(--accent); }
  button { margin-top: 12px; width: 100%; padding: 10px; border: 0; border-radius: 999px; background: var(--accent);
           color: #fff; font: inherit; font-weight: 600; cursor: pointer; }
  .err { color: var(--err); font-size: 13px; margin: 10px 0 0; }
  .foot { margin-top: 16px; font-size: 12px; } .foot a { color: var(--muted); }
</style></head><body>
<form method="post" action="/login">
  <h1>Image<span>Search</span></h1>
  <p>Enter the site password to search the catalog.</p>
  <input type="password" name="password" placeholder="Password" autofocus required aria-label="Password">
  <button type="submit">Continue</button>
  {error}
  <div class="foot"><a href="/about">About the API</a></div>
</form></body></html>"""


def create_router(get_index, read_upload) -> APIRouter:
    router = APIRouter(include_in_schema=False)

    def require(request: Request) -> None:
        if not _authed(request):
            raise HTTPException(401, "log in at /login")

    @router.get("/login", response_class=HTMLResponse)
    def login_page(error: int = 0) -> str:
        if not SITE_PASSWORD:
            return LOGIN.replace("{error}", '<p class="err">The site password isn\'t configured yet.</p>')
        return LOGIN.replace("{error}", '<p class="err">Wrong password.</p>' if error else "")

    @router.post("/login")
    def login(password: str = Form("")):
        if not SITE_PASSWORD or not hmac.compare_digest(password.encode(), SITE_PASSWORD.encode()):
            time.sleep(1)   # slows down guessing
            return RedirectResponse("/login?error=1", status_code=303)
        expires = int(time.time()) + SESSION_DAYS * 86400
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie(COOKIE, f"{expires}.{_sign(expires)}", max_age=SESSION_DAYS * 86400,
                        httponly=True, secure=True, samesite="lax")
        return resp

    @router.get("/logout")
    def logout():
        resp = RedirectResponse("/login", status_code=303)
        resp.delete_cookie(COOKIE)
        return resp

    @router.get("/")
    def page(request: Request):
        if not _authed(request):
            return RedirectResponse("/login", status_code=303)
        return FileResponse(PAGE, headers={"Cache-Control": "no-store"})

    @router.get("/api/status")
    def status(request: Request) -> dict:
        require(request)
        index = get_index()
        s = index.items
        return {
            "products": len(s),
            "device": "cpu",
            "shared_models": [{"key": "siglip2-b16", "label": MODELS["siglip2-b16"][2]}],
            "default_model": "siglip2-b16",
            "text_model": MODELS["minilm"][2],
            "image_model": None,
            "photo_weight": PHOTO_WEIGHT,
            "palette": index.manifest["palette"],
            "facets": {
                "colors": {c: int(n) for c, n in s["color"].value_counts().items() if c},
                "categories": {c: int(n) for c, n in s["category"].str.split(" > ").str[0]
                               .value_counts().items() if c},
            },
            "search_types": {k: {"query": v[0], "family": v[1], "label": v[2]} for k, v in SEARCH_TYPES.items()},
            "default_type": "t2p-fused",
            "logout": "/logout",
        }

    @router.post("/api/search")
    def search(request: Request, type: str = Form("t2p-fused"), q: str = Form(""), pid: str = Form(""),
               color: str = Form(""), category: str = Form(""), image: UploadFile | None = File(None)) -> dict:
        require(request)
        index = get_index()
        if type not in SEARCH_TYPES:
            raise HTTPException(400, f"type must be one of {list(SEARCH_TYPES)}")
        if color and color not in index.manifest["palette"]:
            raise HTTPException(400, "unknown color")
        mask = index.mask(color or None, category or None)
        q = q.strip()[:300]
        kind, family, _ = SEARCH_TYPES[type]
        t0 = time.perf_counter()

        sources, exclude, query_product = {}, None, None
        if kind == "text":
            if not q:
                raise HTTPException(400, "type a query")
            s_title, s_photo = index.text_scores(q, title=type != "t2i", photo=type != "t2t-sep")
            if type == "t2p-fused":
                scores = blend(s_title, s_photo)
                strong = {"text": {i for i, _ in index.top(s_title, RECALL_K, mask)},
                          "image": {i for i, _ in index.top(s_photo, RECALL_K, mask)}}
            else:
                scores = s_photo if type == "t2i" else s_title
            model = {"t2p-fused": f"{MODELS['minilm'][2]} + {MODELS['siglip2-b16'][2]}",
                     "t2i": MODELS["siglip2-b16"][2], "t2t-sep": MODELS["minilm"][2]}[type]
        else:
            if pid:   # "More like this": the product's stored photo vector
                if pid not in index.row:
                    raise HTTPException(404, "unknown product")
                exclude = index.row[pid]
                scores = index.photo[exclude] @ index.photo.T
                query_product = index.item(exclude)
            elif image is not None:
                scores = index.image_scores(read_upload(image))
            else:
                raise HTTPException(400, "upload an image or pick a product")
            model = MODELS["siglip2-b16"][2]

        results = []
        for i, score in index.top(scores, TOP_K, mask, exclude=exclude):
            item = index.item(i, score)
            if type == "t2p-fused":
                item["sources"] = [side for side, ids in strong.items() if i in ids]
            results.append(item)
        return {"type": type, "model": model, "results": results, "query_product": query_product,
                "ms": round((time.perf_counter() - t0) * 1000, 1)}

    return router
