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
  ollama  a local vision model through Ollama (free, slower). Resumable:
          re-running skips products already written.

Output: image_search/queries/queries_<backend>.jsonl, one row per query.
Score it with `python image_search/benchmark_queries.py`.

Usage:
    python image_search/generate_queries.py --backend ollama
    python image_search/generate_queries.py --backend claude
    python image_search/generate_queries.py --backend claude --batch-id msgbatch_...   # resume polling
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


# ---------------------------------------------------------------- ollama

def ollama_sees_images(model: str, host: str) -> bool:
    """Some Ollama builds silently drop images; check with a plain red square."""
    import requests
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (220, 20, 20)).save(buf, format="JPEG")
    r = requests.post(f"{host}/api/chat", timeout=300, json={
        "model": model, "stream": False, "think": False,
        "messages": [{"role": "user", "content": "What color is this image? Answer with one word.",
                      "images": [base64.standard_b64encode(buf.getvalue()).decode()]}],
    })
    r.raise_for_status()
    answer = r.json()["message"]["content"]
    print(f"vision check: {answer.strip()[:80]!r}")
    return "red" in answer.lower()


def run_ollama(products: pd.DataFrame, out: Path, model: str, host: str, sources: list[str]) -> None:
    import requests

    if "image" in sources and not ollama_sees_images(model, host):
        raise SystemExit(f"{model} can't see images in this Ollama install (it answered without "
                         "looking at the test image). Use --sources text, another vision model, "
                         "or --backend claude.")

    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            r = json.loads(line)
            done.add((r["product_id"], r["source"]))

    todo = [(row, src) for _, row in products.iterrows() for src in sources
            if (row["product_id"], src) not in done]
    print(f"{len(done)} requests already done, {len(todo)} to go with {model}")

    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for i, (row, source) in enumerate(todo, 1):
            if source == "text":
                msg = {"role": "user", "content": TEXT_PROMPT.format(
                    title=row["title"], merchant=row["merchant"], description=description_for(row))}
            else:
                msg = {"role": "user", "content": IMAGE_PROMPT, "images": [jpeg_b64(row["local_image"])]}
            try:
                r = requests.post(f"{host}/api/chat", timeout=300, json={
                    "model": model, "messages": [msg], "format": SCHEMA, "stream": False,
                    "think": False, "options": {"temperature": 0.7},
                })
                r.raise_for_status()
                data = json.loads(r.json()["message"]["content"])
            except Exception as e:  # one bad response shouldn't stop an hour-long run
                print(f"  skip {row['product_id']}-{source}: {e}")
                continue
            for q in rows_for(row["product_id"], source, data, model):
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                print(f"  {i}/{len(todo)}  ({rate:.1f}s each, ~{rate*(len(todo)-i)/60:.0f} min left)", flush=True)
    print(f"done -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("claude", "ollama"), required=True)
    ap.add_argument("--n", type=int, default=500, help="number of products")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--claude-model", default="claude-opus-5-5")
    ap.add_argument("--ollama-model", default="gemma4:e4b")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--sources", nargs="+", choices=("text", "image"), default=["text", "image"])
    ap.add_argument("--batch-id", help="resume polling an already-submitted Claude batch")
    args = ap.parse_args()

    QUERIES_DIR.mkdir(exist_ok=True)
    out = QUERIES_DIR / f"queries_{args.backend}.jsonl"
    products = sample_products(args.n, args.seed)
    print(f"{len(products)} products (seed {args.seed})")

    if args.backend == "claude":
        run_claude(products, out, args.claude_model, args.batch_id, args.sources)
    else:
        run_ollama(products, out, args.ollama_model, args.ollama_host, args.sources)


if __name__ == "__main__":
    main()
