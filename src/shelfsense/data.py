"""Loading, subsampling, and train/test splitting for the H&M transactions dataset."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from shelfsense import config


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    articles = pd.read_csv(config.DATA_RAW / "articles.csv", dtype={"article_id": str})
    customers = pd.read_csv(config.DATA_RAW / "customers.csv", dtype={"customer_id": str})
    transactions = pd.read_csv(
        config.DATA_RAW / "transactions_train.csv",
        dtype={"customer_id": str, "article_id": str},
        parse_dates=["t_dat"],
    )
    return articles, customers, transactions


def subsample_customers(
    transactions: pd.DataFrame,
    n_customers: int = config.N_CUSTOMERS_SAMPLE,
    seed: int = config.SUBSAMPLE_SEED,
) -> pd.DataFrame:
    all_customers = transactions["customer_id"].unique()
    rng = np.random.default_rng(seed)
    keep = rng.choice(all_customers, size=min(n_customers, len(all_customers)), replace=False)
    return transactions[transactions["customer_id"].isin(keep)].reset_index(drop=True)


def time_split(
    transactions: pd.DataFrame, holdout_days: int = config.HOLDOUT_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by a global date cutoff, not per-customer — this mirrors how the model
    will actually be used (predict next week's baskets for everyone at once) and
    prevents leaking future popularity trends into training."""
    cutoff = transactions["t_dat"].max() - pd.Timedelta(days=holdout_days)
    train = transactions[transactions["t_dat"] <= cutoff].reset_index(drop=True)
    test = transactions[transactions["t_dat"] > cutoff].reset_index(drop=True)
    return train, test


@dataclass
class IndexMaps:
    customer_ids: np.ndarray
    article_ids: np.ndarray
    customer_to_idx: dict
    article_to_idx: dict

    @classmethod
    def build(cls, train: pd.DataFrame) -> "IndexMaps":
        customer_ids = np.sort(train["customer_id"].unique())
        article_ids = np.sort(train["article_id"].unique())
        return cls(
            customer_ids=customer_ids,
            article_ids=article_ids,
            customer_to_idx={c: i for i, c in enumerate(customer_ids)},
            article_to_idx={a: i for i, a in enumerate(article_ids)},
        )


def build_interaction_matrix(train: pd.DataFrame, idx: IndexMaps) -> sparse.csr_matrix:
    """Customers x articles matrix of purchase counts, used as implicit-feedback
    confidence weights (repeat purchases signal stronger preference)."""
    rows = train["customer_id"].map(idx.customer_to_idx).to_numpy()
    cols = train["article_id"].map(idx.article_to_idx).to_numpy()
    data = np.ones(len(train), dtype=np.float32)
    mat = sparse.coo_matrix(
        (data, (rows, cols)), shape=(len(idx.customer_ids), len(idx.article_ids))
    ).tocsr()
    mat.sum_duplicates()
    return mat


def customer_test_baskets(test: pd.DataFrame) -> dict:
    """customer_id -> set of article_ids actually purchased in the holdout window."""
    return test.groupby("customer_id")["article_id"].apply(set).to_dict()
