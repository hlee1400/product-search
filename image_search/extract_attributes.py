"""
Classifies each product's color and Google-taxonomy category with a vision
model through the LunaRoute gateway (photo + title in, JSON out).

Color, two methods:
  closed  1-of-N: the response schema only allows PALETTE values
  open    the model names the color in its own words; attributes.normalize_color()
          maps it to the palette afterwards
Category: 1-of-N over the 192 level-2 Google product taxonomy paths (closed method only).

The pilot (300 products) chose the closed method for the full run: open answers
like "black and white" normalize to multicolor where a shopper filtering by color
wants the dominant color, and open answers need a synonym map that never covers
every word. The closed call also returns a free-text color_name ("dusty rose"),
which keeps the open method's nuance for display and search.

  --pilot   N sampled products, both methods, to compare them
  (default) every gallery product, closed method

Output: attributes/<run>_<model>.jsonl, one row per (product, method). Resumable.

Usage:
    python image_search/extract_attributes.py --pilot 300
    python image_search/extract_attributes.py
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

from attributes import ATTR_DIR, PALETTE, normalize_color, taxonomy_level2
from generate_queries import jpeg_b64
from settings import GALLERY_PARQUET

BASE_URL = "https://gw.lunaroute.com/v1"

COLOR_RULES = """Color means the color of the product itself, not the background, the model \
wearing it, or the packaging. If the product shows several colors and none clearly \
dominates, the color is multicolor."""

# The pilot showed the closed method put cream / camel / taupe in white, brown or gray,
# so the palette boundaries are spelled out.
PALETTE_GUIDE = """Allowed colors and what they cover:
black; white (also ivory, off-white); gray (also charcoal, silver-gray); beige (also cream, tan, camel, khaki, taupe, nude, sand); brown (darker browns, chocolate, wood); red (also burgundy, maroon); pink (also blush, rose, fuchsia, coral); orange (also peach, rust, copper); yellow (also mustard); green (also olive, sage, teal-green); blue (also navy, teal, turquoise, denim); purple (also lavender, lilac, plum, mauve); gold (also brass, bronze); silver (also chrome, steel); multicolor (several colors, none dominant: prints, patterns, rainbow); clear (transparent).
If one color clearly dominates and another is a small accent (a sole, trim, a logo), choose the dominant color."""

CLOSED_PROMPT = f"""Classify this product from its photo and title.

- color: the product's main color, chosen from the allowed values. {COLOR_RULES}
- color_name: that color in 1 to 3 words, the way a shopper would describe it (e.g. "navy", "dusty rose", "camel").
- category: the Google product taxonomy category that best fits the product.

{PALETTE_GUIDE}

Title: {{title}}
Store: {{merchant}}"""

OPEN_PROMPT = f"""Look at this product's photo and title.

- color: the product's main color in 1 to 3 words, the way a shopper would describe it \
(e.g. "navy", "dusty rose", "black and white"). {COLOR_RULES}

Title: {{title}}
Store: {{merchant}}"""


def closed_schema() -> dict:
    return {"type": "object", "additionalProperties": False,
            "required": ["color", "color_name", "category"],
            "properties": {"color": {"type": "string", "enum": list(PALETTE)},
                           "color_name": {"type": "string"},
                           "category": {"type": "string", "enum": taxonomy_level2()}}}


OPEN_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["color"],
               "properties": {"color": {"type": "string"}}}


def classify(row, method: str, model: str, key: str) -> dict:
    prompt = (CLOSED_PROMPT if method == "closed" else OPEN_PROMPT).format(
        title=row["title"], merchant=row["merchant"])
    schema = closed_schema() if method == "closed" else OPEN_SCHEMA
    body = {
        "model": model, "max_tokens": 200, "reasoning_effort": "none", "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64(row['local_image'])}"}},
            {"type": "text", "text": prompt}]}],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "attributes", "strict": True, "schema": schema}},
    }
    for attempt in range(4):
        r = requests.post(f"{BASE_URL}/chat/completions", timeout=120, json=body,
                          headers={"Authorization": f"Bearer {key}"})
        if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
            time.sleep(2 ** attempt * 2)
            continue
        r.raise_for_status()
        break
    res = r.json()
    text = res["choices"][0]["message"]["content"]
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    out = {"product_id": row["product_id"], "method": method, "model": model,
           "tokens_in": res["usage"]["prompt_tokens"], "tokens_out": res["usage"]["completion_tokens"]}
    if method == "closed":
        # Validate rather than trust the schema: not every backend enforces enums.
        out["color"] = data["color"] if data["color"] in PALETTE else None
        out["category"] = data["category"] if data["category"] in taxonomy_level2() else None
        out["color_name"] = data.get("color_name", "").strip().lower()
    else:
        out["color_raw"] = data["color"]
        out["color"] = normalize_color(data["color"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, help="classify this many sampled products with both methods")
    ap.add_argument("--model", default="deepseek-4.1-flash")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    key = os.environ.get("LUNAROUTE_API_KEY")
    if not key:
        raise SystemExit("Set the LUNAROUTE_API_KEY environment variable.")

    gallery = pd.read_parquet(GALLERY_PARQUET)
    if args.pilot:
        rng = np.random.default_rng(args.seed)
        gallery = gallery.iloc[np.sort(rng.choice(len(gallery), size=args.pilot, replace=False))]
        methods, run = ["closed", "open"], "pilot"
    else:
        methods, run = ["closed"], "full"
    out = ATTR_DIR / f"{run}_{args.model}.jsonl"

    done = set()
    if out.exists():
        done = {(r["product_id"], r["method"]) for r in map(json.loads, out.open(encoding="utf-8"))}
    todo = [(row, m) for _, row in gallery.iterrows() for m in methods if (row["product_id"], m) not in done]
    print(f"{len(done)} done, {len(todo)} to go ({len(gallery):,} products x {methods}) with {args.model}")

    t0, failed, tok_in, tok_out = time.time(), 0, 0, 0
    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(classify, row, m, args.model, key): (row["product_id"], m) for row, m in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                r = fut.result()
            except Exception as e:  # keep going; a re-run retries what's missing
                failed += 1
                print(f"  fail {futures[fut]}: {str(e)[:120]}")
                continue
            tok_in += r["tokens_in"]
            tok_out += r["tokens_out"]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 200 == 0 or i == len(todo):
                f.flush()
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  {el:.0f}s, ~{el / i * (len(todo) - i) / 60:.0f} min left, "
                      f"avg {tok_in / max(1, i - failed):.0f} tokens in / {tok_out / max(1, i - failed):.0f} out",
                      flush=True)
    print(f"done -> {out}  ({failed} failed; re-run to retry)  "
          f"tokens this run: {tok_in:,} in, {tok_out:,} out")


if __name__ == "__main__":
    main()
