"""Entry point: python -m shelfsense.build_features

Reads the raw Kaggle CSVs, subsamples to a tractable customer set, applies the
time-based train/test split, and writes processed parquet files that every
other module (training, evaluation, serving) reads from.
"""
from shelfsense import config, data


def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print("Loading raw CSVs...")
    articles, customers, transactions = data.load_raw()

    print(f"Subsampling to {config.N_CUSTOMERS_SAMPLE} customers...")
    transactions = data.subsample_customers(transactions)
    kept_customers = transactions["customer_id"].unique()
    customers = customers[customers["customer_id"].isin(kept_customers)].reset_index(drop=True)
    kept_articles = transactions["article_id"].unique()
    articles = articles[articles["article_id"].isin(kept_articles)].reset_index(drop=True)

    print(f"Splitting at last {config.HOLDOUT_DAYS} days...")
    train, test = data.time_split(transactions)

    print(
        f"train={len(train):,} rows / test={len(test):,} rows / "
        f"{len(kept_customers):,} customers / {len(kept_articles):,} articles"
    )

    articles.to_parquet(config.DATA_PROCESSED / "articles.parquet", index=False)
    customers.to_parquet(config.DATA_PROCESSED / "customers.parquet", index=False)
    train.to_parquet(config.DATA_PROCESSED / "transactions_train.parquet", index=False)
    test.to_parquet(config.DATA_PROCESSED / "transactions_test.parquet", index=False)
    print(f"Wrote processed parquet files to {config.DATA_PROCESSED}")


if __name__ == "__main__":
    main()
