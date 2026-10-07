"""
Generates a search benchmark: for a sample of products, an LLM writes the
queries a shopper might type to find each one. It does this twice per product:

  source=text   the model sees only the title, merchant and store description
  source=image  the model sees only the product photo

Each request asks for three queries in different styles (short, specific,
need). The AI descriptions are left out of the text prompt because they were
probably written from the photo.

Two backends:
  claude  Claude via the Message Batches API (50% price, usually < 1 hour).
          Needs ANTHROPIC_API_KEY (or an `ant auth login` profile).
  ollama  a local vision model through Ollama (free, slower).
  api     any OpenAI-compatible endpoint: a hosted provider running the model
          (e.g. Qwen2.5-VL on OpenRouter), optionally through a local LunaRoute
          proxy that forwards and records the requests. Runs 8 requests at once.
The ollama and api backends are resumable: re-running skips products this
model has already written.

Output: image_search/queries/queries_<backend>.jsonl, one row per query.
Score it with `python image_search/benchmark_queries.py`.

Usage:
    python image_search/generate_queries.py --backend ollama
    python image_search/generate_queries.py --backend claude
    python image_search/generate_queries.py --backend claude --batch-id msgbatch_...   # resume polling
    python image_search/generate_queries.py --backend api --api-model qwen/qwen2.5-vl-7b-instruct \
        --api-base-url <LunaRoute or provider URL>/v1
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

SCHEMA = {
    "type": "object",
    "properties": {s: {"type": "string"} for s in STYLES},
    "required": list(STYLES),
    "additionalProperties": False,
}


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
        for s in STYLES if data.get(s, "").strip()
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
            for source, content in (("text", text_content), ("image", image_content)):
                if source not in sources:
                    continue
                requests.append({
                    "custom_id": f"{row['product_id']}-{source}",
                    "params": {
                        "model": model,
                        "max_tokens": 4000,
                        # A short, well-specified writing task: low effort is plenty.
                        "output_config": {"effort": "low",
                                          "format": {"type": "json_schema", "schema": SCHEMA}},
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


def api_caller(model: str, base_url: str, key_env: str):
    """Same interface over any OpenAI-compatible /chat/completions endpoint: a hosted
    provider (OpenRouter, Together, ...) or a local LunaRoute proxy in front of one."""
    import os
    import requests

    key = os.environ.get(key_env, "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    def call(prompt: str, image: str | None, schema: dict | None = SCHEMA) -> dict | str:
        content = [{"type": "text", "text": prompt}]
        if image:
            content.insert(0, {"type": "image_url",
                               "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
        body = {"model": model, "messages": [{"role": "user", "content": content}],
                "temperature": 0.7, "max_tokens": 400}
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
    ap.add_argument("--api-model", help="model name at the API endpoint, e.g. qwen/qwen2.5-vl-7b-instruct")
    ap.add_argument("--api-base-url",
                    help="OpenAI-compatible base URL: your LunaRoute proxy's address, or the provider's")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY",
                    help="environment variable holding the API key, if the endpoint needs one")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel requests (default: 1 for ollama, 8 for api)")
    ap.add_argument("--sources", nargs="+", choices=("text", "image"), default=["text", "image"])
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
        if not (args.api_model and args.api_base_url):
            raise SystemExit("--backend api needs --api-model and --api-base-url")
        run_requests(products, out, args.api_model,
                     api_caller(args.api_model, args.api_base_url, args.api_key_env),
                     args.sources, args.workers or 8)


if __name__ == "__main__":
    main()
