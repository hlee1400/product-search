"""Paths shared by build_index.py, evaluate.py and app.py."""
from pathlib import Path

IMAGE_SEARCH_DIR = Path(__file__).resolve().parent
ROOT = IMAGE_SEARCH_DIR.parent

DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "product_images" / "product_images"

# Reuse the cleaned catalog that search_app/build_index.py already writes.
SOURCE_CORPUS = ROOT / "search_app" / "artifacts" / "corpus.parquet"

ARTIFACTS_DIR = IMAGE_SEARCH_DIR / "artifacts"
GALLERY_PARQUET = ARTIFACTS_DIR / "gallery.parquet"
RESULTS_DIR = IMAGE_SEARCH_DIR / "results"


def emb_path(model_key: str, modality: str) -> Path:
    """Cached embeddings for one model over one side of the gallery."""
    return ARTIFACTS_DIR / f"{model_key}_{modality}.npy"
