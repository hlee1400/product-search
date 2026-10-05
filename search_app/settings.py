"""Paths shared by build_index.py and app.py."""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

# The search pipeline lives in its own folder; import it as the `src` package.
PIPELINE_ROOT = ROOT / "search_pipeline"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from src.config.paths import slug  # noqa: E402
from src.dense.dense_search import DEFAULT_MODEL  # noqa: E402

DATA_DIR = ROOT / "data"
CATALOG_CSV = DATA_DIR / "product_catalog_10_17_24.csv"
IMAGES_DIR = DATA_DIR / "product_images" / "product_images"

ARTIFACTS_DIR = APP_DIR / "artifacts"
CORPUS_PARQUET = ARTIFACTS_DIR / "corpus.parquet"
BM25_DIR = ARTIFACTS_DIR / "bm25"
BM25_PIDS = ARTIFACTS_DIR / "bm25_pids.npy"
# Dense cache keyed by embedding model, same convention as the pipeline.
DENSE_EMB = ARTIFACTS_DIR / f"dense_emb_{slug(DEFAULT_MODEL)}.npy"
DENSE_PIDS = ARTIFACTS_DIR / f"dense_pids_{slug(DEFAULT_MODEL)}.npy"
