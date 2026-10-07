"""
Product attribute vocabularies: a color palette and Google's product taxonomy.

Color is classified two ways (see extract_attributes.py):
  closed  the model picks one PALETTE value directly (1-of-N)
  open    the model names the color freely ("dusty rose"), and normalize_color()
          maps that to the palette

Category is 1-of-N over the 192 second-level paths of the Google product
taxonomy (e.g. "Apparel & Accessories > Shoes").
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

ATTR_DIR = Path(__file__).resolve().parent / "attributes"
TAXONOMY_FILE = ATTR_DIR / "google_product_taxonomy.txt"   # taxonomy-with-ids.en-US.txt, 2021-09-21

# Broad enough to filter on, narrow enough that two people would agree.
PALETTE = (
    "black", "white", "gray", "beige", "brown", "red", "pink", "orange", "yellow",
    "green", "blue", "purple", "gold", "silver", "multicolor", "clear",
)

# Free-form color words -> palette. Checked last word first ("dusty rose" -> rose).
SYNONYMS = {
    # black / white / gray
    "black": "black", "jet": "black", "onyx": "black", "charcoal": "gray", "ebony": "black",
    "white": "white", "ivory": "white", "off-white": "white", "offwhite": "white", "snow": "white",
    "gray": "gray", "grey": "gray", "slate": "gray", "graphite": "gray", "heather": "gray",
    "ash": "gray", "pewter": "gray", "smoke": "gray", "stone": "beige",
    # neutrals
    "beige": "beige", "cream": "beige", "tan": "beige", "khaki": "beige", "camel": "beige",
    "nude": "beige", "sand": "beige", "oatmeal": "beige", "ecru": "beige", "taupe": "beige",
    "natural": "beige", "champagne": "beige", "bone": "beige", "linen": "beige", "neutral": "beige",
    "brown": "brown", "chocolate": "brown", "espresso": "brown", "walnut": "brown", "mocha": "brown",
    "cognac": "brown", "rust": "orange", "chestnut": "brown", "mahogany": "brown", "wood": "brown",
    "oak": "brown", "caramel": "brown", "coffee": "brown", "bronze": "gold", "copper": "orange",
    # reds / pinks
    "red": "red", "burgundy": "red", "maroon": "red", "wine": "red", "crimson": "red",
    "scarlet": "red", "cherry": "red", "brick": "red", "oxblood": "red", "ruby": "red",
    "pink": "pink", "rose": "pink", "blush": "pink", "fuchsia": "pink", "magenta": "pink",
    "coral": "pink", "salmon": "pink", "mauve": "purple", "hot pink": "pink",
    # warm
    "orange": "orange", "tangerine": "orange", "peach": "orange", "apricot": "orange",
    "terracotta": "orange", "yellow": "yellow", "mustard": "yellow", "lemon": "yellow",
    # greens / blues / purples
    "green": "green", "olive": "green", "sage": "green", "emerald": "green", "mint": "green",
    "forest": "green", "lime": "green", "teal": "blue", "turquoise": "blue", "aqua": "blue",
    "blue": "blue", "navy": "blue", "denim": "blue", "cobalt": "blue", "indigo": "blue",
    "sky": "blue", "royal": "blue", "cyan": "blue", "periwinkle": "blue",
    "purple": "purple", "lavender": "purple", "lilac": "purple", "violet": "purple",
    "plum": "purple", "eggplant": "purple",
    # metallic / other
    "gold": "gold", "golden": "gold", "brass": "gold", "silver": "silver", "chrome": "silver",
    "metallic": "silver", "platinum": "silver", "steel": "silver", "nickel": "silver",
    "multicolor": "multicolor", "multicolored": "multicolor", "multi": "multicolor",
    "rainbow": "multicolor", "colorful": "multicolor", "assorted": "multicolor", "print": "multicolor",
    "clear": "clear", "transparent": "clear",
}


def normalize_color(raw: str) -> str | None:
    """Map a free-form color ("Dusty Rose", "navy blue") to a PALETTE value, or None."""
    text = raw.lower().strip()
    if text in SYNONYMS:
        return SYNONYMS[text]
    words = re.findall(r"[a-z]+(?:-[a-z]+)?", text)
    if any(sep in text for sep in (" and ", "/", ",", "&")) and len(words) > 1:
        # Two named colors ("black and white") -> multicolor, unless both map the same.
        mapped = {SYNONYMS[w] for w in words if w in SYNONYMS}
        if len(mapped) > 1:
            return "multicolor"
    for w in reversed(words):   # head noun last: "dusty rose", "light blue"
        if w in SYNONYMS:
            return SYNONYMS[w]
    return None


@lru_cache
def taxonomy_level2() -> list[str]:
    """The 192 second-level Google product taxonomy paths."""
    paths = []
    for line in TAXONOMY_FILE.open(encoding="utf-8"):
        if line[:1].isdigit():
            path = line.rstrip("\n").split(" - ", 1)[1]
            if path.count(" > ") == 1:
                paths.append(path)
    return paths


def load_attributes(gallery):
    """Attributes aligned to the gallery's rows (empty strings where unclassified).

    Reads every attributes/full_*.jsonl (closed method); later files win.
    """
    import pandas as pd

    files = sorted(ATTR_DIR.glob("full_*.jsonl"))
    cols = ["color", "color_name", "category"]
    if not files:
        return pd.DataFrame("", index=range(len(gallery)), columns=cols + ["category_l1"])
    rows = pd.concat([pd.read_json(f, lines=True, dtype={"product_id": str}) for f in files])
    rows = rows[rows["method"] == "closed"].drop_duplicates("product_id", keep="last")
    out = gallery[["product_id"]].merge(rows[["product_id"] + cols], on="product_id", how="left")
    out = out[cols].fillna("").reset_index(drop=True)
    out["category_l1"] = out["category"].str.split(" > ").str[0].fillna("")
    return out


# Words that are colors in a product description but usually something else in a
# search query ("coffee table", "wine glass", "art print", "espresso machine").
QUERY_AMBIGUOUS = {
    "print", "coffee", "wood", "wine", "natural", "neutral", "stone", "linen", "steel",
    "chocolate", "espresso", "champagne", "oak", "snow", "sky", "cherry", "ruby", "bone",
    "multi", "heather", "brick", "apricot", "lemon", "lime", "smoke", "ash", "sand", "royal",
    "forest", "jet", "metallic", "walnut", "coral", "salmon", "peach", "mint", "caramel",
}


def query_colors(text: str) -> set[str]:
    """Palette colors a search query asks for ("red summer dress" -> {"red"})."""
    t = text.lower()
    if "rose gold" in t:   # between pink and gold; don't guess
        return set()
    words = re.findall(r"[a-z]+(?:-[a-z]+)?", t)
    return {SYNONYMS[w] for w in words if w in SYNONYMS and w not in QUERY_AMBIGUOUS}
