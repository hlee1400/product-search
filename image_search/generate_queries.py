"""
Generates a search benchmark: for a sample of products, an LLM writes the
queries a shopper might type to find each one. It does this twice per product:

  source=text   the model sees only the title, merchant and store description
  source=image  the model sees only the product photo

Each request asks for three queries in different styles (short, specific,
need). A third source, brand (--sources brand), writes brand + product type
and brand + model queries from the listing text. The AI descriptions are
left out of the text prompts because they were probably written from the
photo.

Backends:
  claude  Claude via the Message Batches API (50% price, usually < 1 hour).
          Needs ANTHROPIC_API_KEY (or an `ant auth login` profile).
  ollama  a local vision model through Ollama (free, slower).
  api     any OpenAI-compatible endpoint; defaults to the LunaRoute gateway
          (https://gw.lunaroute.com/v1, key in LUNAROUTE_API_KEY) with
          deepseek-4.1-flash. The model runs remotely, so the laptop does no
          inference. Runs 8 requests at once.
The ollama and api backends are resumable: re-running skips products this
model has already written.

Output: image_search/queries/queries_<backend>.jsonl, one row per query.
Score it with `python image_search/benchmark_queries.py`.

Usage:
    python image_search/generate_queries.py --backend ollama
    python image_search/generate_queries.py --backend claude
    python image_search/generate_queries.py --backend claude --batch-id msgbatch_...   # resume polling
    python image_search/generate_queries.py --backend api      # LunaRoute gateway, deepseek-4.1-flash
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from settings import GALLERY_PARQUET, IMAGE_SEARCH_DIR, IMAGES_DIR, SOURCE_CORPUS

QUERIES_DIR = IMAGE_SEARCH_DIR / "queries"
STYLES = ("short", "specific", "need")
BRAND_STYLES = ("brand", "exact")
MAX_IMAGE_SIDE = 768

INSTRUCTIONS = """You are helping build a benchmark for an online shopping search engine.

Write the search queries a real shopper might type into a store's search box when \
they want to find this product. Write exactly three, one per style:
- short: 2 to 4 words, the way most people search (e.g. "white chalk paint")
- specific: adds the attributes that set this product apart, such as color, material, \
pattern, shape, size or style (e.g. "matte white chalk paint for furniture")
- need: describes what the shopper wants it for, in natural language \
(e.g. "paint to give an old dresser a vintage look")

Rules:
- Don't copy the product title. Write how a shopper would phrase it.
- No brand names, store names, model numbers or prices.
- Only describe what you can actually tell from the information you're given."""

TEXT_PROMPT = INSTRUCTIONS + """

You're given the product's listing text only (no photo).

Title: {title}
Store: {merchant}
Store description: {description}"""

IMAGE_PROMPT = INSTRUCTIONS + """

You're given the product photo only (no listing text)."""

# A shopper who already knows what they want: the one kind of query the
# text/image prompts above rule out, and where title search should matter most.
BRAND_PROMPT = """You are helping build a benchmark for an online shopping search engine.

Write the search queries a shopper who already knows exactly which product they want might type into a store's search box. Write exactly two:
- brand: the brand name plus the kind of product, 2 to 5 words (e.g. "alex and ani charm bangle")
- exact: the brand plus the specific product line, model name or model number, if the listing has one (e.g. "golden goose super-star sneakers"); otherwise the brand plus the product's most distinctive feature

Rules:
- The brand is whoever makes the product, not the store selling it (unless the store sells its own brand).
- Don't copy the whole title. Write how a shopper would phrase it, lowercase is fine.
- If the listing doesn't show a brand, return empty strings for both.

Title: {title}
Store: {merchant}
Store description: {description}"""


def schema_for(styles: tuple[str, ...]) -> dict:
    return {
        "type": "object",
        "properties": {s: {"type": "string"} for s in styles},
        "required": list(styles),
        "additionalProperties": False,
    }


SCHEMA = schema_for(STYLES)
BRAND_SCHEMA = schema_for(BRAND_STYLES)
STYLES_BY_SOURCE = {"text": STYLES, "image": STYLES, "brand": BRAND_STYLES}


def sample_products(n: int, seed: int) -> pd.DataFrame:
    gallery = pd.read_parquet(GALLERY_PARQUET)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(gallery), size=min(n, len(gallery)), replace=False))
    sample = gallery.iloc[idx]
    # The gallery doesn't keep the store description; take it from the catalog.
    desc = pd.read_parquet(SOURCE_CORPUS, columns=["product_id", "description"])
    return sample.merge(desc.drop_duplicates("product_id"), on="product_id", how="left")


def description_for(row) -> str:
    # Store descriptions are often boilerplate or a copy of the title; keep them short.
    d = " ".join(str(row["description"]).split())
    return d[:600] if d and d.lower() != "na" else "(none)"


def jpeg_b64(name: str) -> str:
    """Product photo as a base64 JPEG, downscaled to cut tokens and request size."""
    with Image.open(IMAGES_DIR / name) as im:
        im = im.convert("RGB")
        im.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()


def rows_for(pid: str, source: str, data: dict, generator: str) -> list[dict]:
    return [
        {"product_id": pid, "source": source, "style": s, "query": data[s].strip(), "generator": generator}
        for s in STYLES_BY_SOURCE[source] if data.get(s, "").strip()
    ]


# ---------------------------------------------------------------- claude

def run_claude(products: pd.DataFrame, out: Path, model: str, batch_id: str | None,
               sources: list[str]) -> None:
    import anthropic
    client = anthropic.Anthropic()

    if batch_id is None:
        requests = []
        for _, row in products.iterrows():
            text_content = TEXT_PROMPT.format(
                title=row["title"], merchant=row["merchant"], description=description_for(row))
            image_content = [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": jpeg_b64(row["local_image"])}},
                {"type": "text", "text": IMAGE_PROMPT},
            ]
            brand_content = BRAND_PROMPT.format(
                title=row["title"], merchant=row["merchant"], description=description_for(row))
            for source, content in (("text", text_content), ("image", image_content),
                                    ("brand", brand_content)):
                if source not in sources:
                    continue
                requests.append({
                    "custom_id": f"{row['product_id']}-{source}",
                    "params": {
                        "model": model,
                        "max_tokens": 4000,
                        # A short, well-specified writing task: low effort is plenty.
                        "output_config": {"effort": "low",
                                          "format": {"type": "json_schema", "schema":
                                                     BRAND_SCHEMA if source == "brand" else SCHEMA}},
                        "messages": [{"role": "user", "content": content}],
                    },
                })
        batch = client.messages.batches.create(requests=requests)
        batch_id = batch.id
        print(f"submitted batch {batch_id} ({len(requests)} requests)")
        print(f"  if this stops, resume with: --backend claude --batch-id {batch_id}")

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        print(f"  {batch.processing_status}: {c.succeeded} ok, {c.errored} errored, "
              f"{c.processing} processing", flush=True)
        if batch.processing_status == "ended":
            break
        time.sleep(60)

    rows, failed = [], []
    for result in client.messages.batches.results(batch_id):
        pid, source = result.custom_id.rsplit("-", 1)
        msg = result.result.message if result.result.type == "succeeded" else None
        if msg is None or msg.stop_reason != "end_turn":
            failed.append(f"{result.custom_id}: {result.result.type}"
                          + (f" / {msg.stop_reason}" if msg else ""))
            continue
        text = next(b.text for b in msg.content if b.type == "text")
        rows += rows_for(pid, source, json.loads(text), model)

    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} queries -> {out}")
    if failed:
        print(f"{len(failed)} requests failed:\n  " + "\n  ".join(failed[:20]))


# ---------------------------------------------------------------- ollama / API

def ollama_caller(model: str, host: str):
    """Returns call(prompt, image_b64 or None) -> parsed JSON, via Ollama's chat API."""
    import requests

    def call(prompt: str, image: str | None, schema: dict | None = SCHEMA) -> dict | str:
        msg = {"role": "user", "content": prompt, **({"images": [image]} if image else {})}
        body = {"model": model, "messages": [msg], "stream": False, "think": False,
                "options": {"temperature": 0.7}}
        if schema:
            body["format"] = schema
        r = requests.post(f"{host}/api/chat", timeout=300, json=body)
        r.raise_for_status()
        text = r.json()["message"]["content"]
        return json.loads(text) if schema else text
    return call


def api_caller(model: str, base_url: str, key_env: str, reasoning_effort: str | None = None):
    """Same interface over any OpenAI-compatible /chat/completions endpoint: a hosted
    provider (OpenRouter, Together, ...) or a local LunaRoute proxy in front of one."""
    import os
    import requests

    key = os.environ.get(key_env, "")
    if not key:
        raise SystemExit(f"Set the {key_env} environment variable to your API key.")
    headers = {"Authorization": f"Bearer {key}"}

    def call(prompt: str, image: str | None, schema: dict | None = SCHEMA) -> dict | str:
        content = [{"type": "text", "text": prompt}]
        if image:
            content.insert(0, {"type": "image_url",
                               "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
        body = {"model": model, "messages": [{"role": "user", "content": content}],
                "temperature": 0.7, "max_tokens": 400}
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "queries", "strict": True, "schema": schema}}
        r = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers,
                          timeout=300, json=body)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        if not schema:
            return text
        # Not every provider enforces the schema; fall back to the first {...} in the reply.
        return json.loads(text[text.index("{"): text.rindex("}") + 1])
    return call


def sees_images(call) -> bool:
    """Some setups silently drop images; check with a plain red square."""
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (220, 20, 20)).save(buf, format="JPEG")
    answer = call("What color is this image? Answer with one word.",
                  base64.standard_b64encode(buf.getvalue()).decode(), schema=None)
    print(f"vision check: {answer.strip()[:80]!r}")
    return "red" in answer.lower()


def run_requests(products: pd.DataFrame, out: Path, generator: str, call, sources: list[str],
                 workers: int) -> None:
    """Writes queries for every (product, source) not already in `out` for this generator."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if "image" in sources and not sees_images(call):
        raise SystemExit(f"{generator} can't see images here (it answered without looking at the "
                         "test image). Use --sources text, another vision model, or another backend.")

    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            r = json.loads(line)
            if r["generator"] == generator:
                done.add((r["product_id"], r["source"]))

    todo = [(row, src) for _, row in products.iterrows() for src in sources
            if (row["product_id"], src) not in done]
    print(f"{len(done)} requests already done, {len(todo)} to go with {generator} ({workers} at a time)")

    def one(row, source):
        if source == "brand":
            prompt = BRAND_PROMPT.format(title=row["title"], merchant=row["merchant"],
                                         description=description_for(row))
            return call(prompt, None, schema=BRAND_SCHEMA)
        if source == "text":
            prompt = TEXT_PROMPT.format(title=row["title"], merchant=row["merchant"],
                                        description=description_for(row))
            return call(prompt, None)
        return call(IMAGE_PROMPT, jpeg_b64(row["local_image"]))

    t0, skipped = time.time(), 0
    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, row, src): (row["product_id"], src) for row, src in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            pid, source = futures[fut]
            try:
                rows = rows_for(pid, source, fut.result(), generator)
            except Exception as e:  # one bad response shouldn't stop a long run
                skipped += 1
                print(f"  skip {pid}-{source}: {e}")
                continue
            for q in rows:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                print(f"  {i}/{len(todo)}  ({rate:.1f}s each, ~{rate*(len(todo)-i)/60:.0f} min left)", flush=True)
    print(f"done -> {out}" + (f" ({skipped} skipped; re-run to retry them)" if skipped else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("claude", "ollama", "api"), required=True)
    ap.add_argument("--n", type=int, default=500, help="number of products")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--claude-model", default="claude-opus-5-5")
    ap.add_argument("--ollama-model", default="gemma4:e4b")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--api-model", default="deepseek-4.1-flash",
                    help="model name at the API endpoint (needs vision for --sources image)")
    ap.add_argument("--api-base-url", default="https://gw.lunaroute.com/v1",
                    help="OpenAI-compatible base URL (default: the LunaRoute gateway)")
    ap.add_argument("--api-reasoning-effort", default="none",
                    help="reasoning_effort sent to the API; 'none' skips reasoning tokens (pass '' to omit)")
    ap.add_argument("--api-key-env", default="LUNAROUTE_API_KEY",
                    help="environment variable holding the API key, if the endpoint needs one")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel requests (default: 1 for ollama, 8 for api)")
    ap.add_argument("--sources", nargs="+", choices=("text", "image", "brand"), default=["text", "image"],
                    help="brand: queries naming the brand / model, written from the listing text")
    ap.add_argument("--batch-id", help="resume polling an already-submitted Claude batch")
    args = ap.parse_args()

    QUERIES_DIR.mkdir(exist_ok=True)
    out = QUERIES_DIR / f"queries_{args.backend}.jsonl"
    products = sample_products(args.n, args.seed)
    print(f"{len(products)} products (seed {args.seed})")

    if args.backend == "claude":
        run_claude(products, out, args.claude_model, args.batch_id, args.sources)
    elif args.backend == "ollama":
        run_requests(products, out, args.ollama_model, ollama_caller(args.ollama_model, args.ollama_host),
                     args.sources, args.workers or 1)
    else:
        run_requests(products, out, args.api_model,
                     api_caller(args.api_model, args.api_base_url, args.api_key_env,
                                args.api_reasoning_effort or None),
                     args.sources, args.workers or 8)


if __name__ == "__main__":
    main()
