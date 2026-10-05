import pandas as pd

def inspect_raw_dataset(
        examples: pd.DataFrame,
        products: pd.DataFrame,
        sources: pd.DataFrame,
) -> None:
    print("\n========== RAW DATASET VERIFICATION ==========")

    #confirm the 3 files loaded and match the documented sizes
    print("\n[1] DataSet shapes")
    print("Examples shape: ", examples.shape)
    print("Products shape: ", products.shape)
    print("Sources shape: ", sources.shape)

    #locale mix
    print("\n[2] Locale distribution in examples")
    print(examples["product_locale"].value_counts())

    print("\n[3] Locale distribution in products")
    print(products["product_locale"].value_counts())

    #train/test split should match roughly 80/20 split
    print("\n[4] Split distribution")
    print(examples["split"].value_counts())

    #class balance among ESCI
    print("\n[5] ESCI label distribution overall")
    #E = Exact match, S = Substitute, C = Complement, I = Irrelevant
    label_counts = examples["esci_label"].value_counts()
    label_pct = (label_counts / len(examples) * 100).round(1)
    print(pd.DataFrame({"n": label_counts, "pct": label_pct}))


    #a model's effective trainging is bounded by unique-query count, not row count
    print("\n[6] Unique query counts")
    print("All unique queries:", examples["query_id"].nunique())
    print(
        "Small version unique queries:",
        examples[examples["small_version"] == 1]["query_id"].nunique(),
    )
    print(
        "Large version unique queries:",
        examples[examples["large_version"] == 1]["query_id"].nunique(),
    )

    #see at a glance how many judged products/query and how many queries we get
    # for every (locale, split) cell — pick the working slice from this table.
    # US train should have the most rows and the highest avg_depth, that's why we use it
    print("\n[7] Judgements by version, locale, and split")
    summary = (
        examples
        .groupby(["product_locale", "split"])
        .agg(
            judgements=("example_id", "count"),
            unique_queries=("query_id", "nunique"),
            avg_depth=("product_id", lambda x: len(x) / examples.loc[x.index, "query_id"].nunique()),
        )
        .reset_index()
    )
    print(summary)

    # --- Slice we actually evaluate on: US small-version ---------------------
    us_small = examples[
        (examples["product_locale"] == "us") & (examples["small_version"] == 1)
    ]
    us_small_test = us_small[us_small["split"] == "test"]
    us_small_train = us_small[us_small["split"] == "train"]


    print("\n[8] ESCI label distribution on US small-version test split")
    print(us_small_test["esci_label"].value_counts())

    # eyeballing one query and judging whether E/S/C/I labels match human judgement 
    print("\n[9] Walk through a single query")
    if not us_small_train.empty:
        sample_query_id = us_small_train["query_id"].iloc[0]
        sample = (
            us_small_train[us_small_train["query_id"] == sample_query_id]
            .merge(
                products[["product_id", "product_locale", "product_title"]],
                on=["product_id", "product_locale"],
                how="left",
            )
        )
        label_order = {"E": 0, "S": 1, "C": 2, "I": 3}
        sample = sample.assign(
            _ord=sample["esci_label"].map(label_order),
            title=sample["product_title"].fillna("").str.slice(0, 95),
        ).sort_values("_ord")
        print(f"Sample query_id={sample_query_id}, query={sample['query'].iloc[0]!r}")
        print(sample[["esci_label", "product_id", "title"]].to_string(index=False))


 #query length dictates retriever design, short queries kill TF-IDF/BM25
    # signal and benefit more from query expansion or dense embeddings
    print("\n[10] Query length distribution on US small-version test split")
    unique_queries = us_small_test["query"].dropna().drop_duplicates()
    n_tokens = unique_queries.str.split().str.len()
    print(n_tokens.describe())
    print("\nToken count -> number of queries (top 15):")
    print(n_tokens.value_counts().sort_index().head(15))


 # cross-check completeness across all locales
    print("\n[11] Missing value rates in product text fields (all locales)")
    product_text_cols = [
        "product_title",
        "product_description",
        "product_bullet_point",
        "product_brand",
        "product_color",
    ]
    print(products[product_text_cols].isna().mean().sort_values(ascending=False))


 # which product fields to rely on as primary retrieval signal
    print("\n[12] Product field completeness (US locale)")
    products_us = products[products["product_locale"] == "us"]
    completeness = pd.DataFrame({
        "filled": products_us[[
            "product_title",
            "product_description",
            "product_bullet_point",
            "product_brand",
            "product_color",
        ]].notna().sum(),
        "total": len(products_us),
    })
    completeness["pct_filled"] = (completeness["filled"] / completeness["total"] * 100).round(1)
    print(completeness)

  # #brand is a high-precision filter signal IF the field is reliable — check the
    # # head of the distribution to see if brand strings are clean enough to filter on.
    # print("\n[13] Top 15 brands (US locale)")
    # top_brands = (
    #     products_us["product_brand"]
    #     .dropna()
    #     .value_counts()
    #     .head(15)
    # )
    # print(top_brands)

    # #color is the textbook "looks structured, actually free text" field. High
    # # cardinality means it's unusable as a clean facet without heavy normalisation.
    # print("\n[14] Color field is dirty (US locale)")
    # top_colors = (
    #     products_us["product_color"]
    #     .dropna()
    #     .value_counts()
    #     .head(15)
    # )
    # print(top_colors)
    # print(
    #     "Unique color strings:",
    #     int(products_us["product_color"].dropna().nunique()),
    # )

    #if a query_id ever appears in both train and test, our eval is leaking.
    # this must be zero.
    print("\n[15] Train/test split is stratified by query (sanity check)")
    splits_per_query = (
        us_small.groupby("query_id")["split"].nunique()
    )
    leaked = int((splits_per_query > 1).sum())
    print("Queries appearing in both splits:", leaked)

    # sizes the supervised-positive set if we want to fine-tune a retriever
    # warns us if the unique-product count is too small to learn rich representations.
    print("\n[16] Available positive (query, product) pairs for supervised training")
    positives = us_small_train[us_small_train["esci_label"] == "E"]
    print("positive_pairs:    ", len(positives))
    print("unique_queries:    ", positives["query_id"].nunique())
    print("unique_products:   ", positives["product_id"].nunique())


    #random observation
    print("\n Korean query: ")
    query_id = 130539


    query_rows = us_small_train[us_small_train["query_id"] == query_id]

    query_with_titles = query_rows.merge(
        products[["product_id", "product_locale", "product_title"]],
        on=["product_id", "product_locale"],
        how="left",
    )

    print(query_with_titles[["query", "product_id", "product_title", "esci_label"]])    

    print("\n========== END DATASET VERIFICATION ==========\n")