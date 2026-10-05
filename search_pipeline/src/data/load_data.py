import pandas as pd

from src.config.paths import (
    SHOPPING_QUERIES_EXAMPLES_PARQUET,
    SHOPPING_QUERIES_PRODUCTS_PARQUET, 
    SHOPPING_QUERIES_SOURCES_CSV,
)

def load_examples() -> pd.DataFrame:
    return pd.read_parquet(SHOPPING_QUERIES_EXAMPLES_PARQUET)

def load_products() -> pd.DataFrame:
    return pd.read_parquet(SHOPPING_QUERIES_PRODUCTS_PARQUET)

def load_sources() -> pd.DataFrame:
    return pd.read_csv(SHOPPING_QUERIES_SOURCES_CSV)


# SELECT *
# FROM examples
# LEFT JOIN products
# ON examples.product_locale = products.product_locale
# AND examples.product_id = products.product_id;
def load_merged_examples_products() -> pd.DataFrame:
    examples = load_examples()
    products = load_products()

    merged = pd.merge(
        examples,
        products,
        how="left",
        left_on=["product_locale", "product_id"],
        right_on=["product_locale", "product_id"],
    )

    return merged

def load_task_data(
    small_version:bool = True,
    locale: str = "us",
) ->tuple[pd.DataFrame, pd.DataFrame]:
    df = load_merged_examples_products()
    version_col = "small_version" if small_version else "large_version"

    # SELECT *
    # FROM df
    # WHERE version_col = 1
    # AND product_locale = 'us';
    df = df[(df[version_col] == 1)
            & (df["product_locale"] == locale)
    ].copy()

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    return train_df, test_df