import re
from pathlib import Path

#root of the project
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def slug(name: str) -> str:
    """Filename-safe lowercase slug. Used to key caches by model id."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

#raw dataset folder
RAW_DATA_DIR = PROJECT_ROOT / "esci-data"

#main dataset folders
CLASSIFICATION_IDENTIFICATION_DIR = RAW_DATA_DIR / "classification_identification"
RANKING_DIR = RAW_DATA_DIR / "ranking"
SHOPPING_QUERIES_DIR = RAW_DATA_DIR / "shopping_queries_dataset"

#specific dataset files
SHOPPING_QUERIES_SOURCES_CSV = (
    SHOPPING_QUERIES_DIR / "shopping_queries_dataset_sources.csv"
)

SHOPPING_QUERIES_EXAMPLES_PARQUET = (
    SHOPPING_QUERIES_DIR / "shopping_queries_dataset_examples.parquet"
)

SHOPPING_QUERIES_PRODUCTS_PARQUET = (
    SHOPPING_QUERIES_DIR / "shopping_queries_dataset_products.parquet"
)

#classification identification folders
QUERY_PRODUCT_DIR = CLASSIFICATION_IDENTIFICATION_DIR / "query_product"

#output folders
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

#generated artifact directories
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
INDEX_DIR = ARTIFACTS_DIR / "vector_index"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

#generated artifacts
PRODUCTS_PROCESSED_CSV = PROCESSED_DATA_DIR / "products_processed.csv"
QUERIES_PROCESSED_CSV = PROCESSED_DATA_DIR / "queries_processed.csv"

PRODUCT_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "product_embeddings.npy"
QUERY_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "query_embeddings.npy"

VECTOR_INDEX_PATH = INDEX_DIR / "product_vector.index"
EVALUATION_METRICS_PATH = METRICS_DIR / "retrieval_metrics.json"
SEARCH_RESULTS_PATH = METRICS_DIR / "search_results.csv"

RELEVANCE_JUDGMENTS_CSV = PROCESSED_DATA_DIR / "relevance_judgments.csv"

# ── Retrieval cache paths ────────────────────────────────────────────
# BM25 cache (single model, fixed paths)
BM25_INDEX_DIR = ARTIFACTS_DIR / "bm25_us"
BM25_PIDS_PATH = ARTIFACTS_DIR / "bm25_pids_us.npy"

# Dense cache (varies by embedding model — keyed by slug(model_id))
def dense_emb_path(model_id: str) -> Path:
    return ARTIFACTS_DIR / f"dense_emb_us_{slug(model_id)}.npy"

def dense_pids_path(model_id: str) -> Path:
    return ARTIFACTS_DIR / f"dense_pids_us_{slug(model_id)}.npy"

# Run logs (timestamped, written by run_eval.py)
LOGS_DIR = ARTIFACTS_DIR / "logs"


def create_output_dirs() -> None:
    """
    create the output directories
    """

    for path in [
        PROCESSED_DATA_DIR,
        ARTIFACTS_DIR,
        EMBEDDINGS_DIR,
        INDEX_DIR,
        METRICS_DIR,
        LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)