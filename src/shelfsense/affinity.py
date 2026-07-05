"""Category-affinity popularity: recommend recent best-sellers within a
customer's own favorite product category, instead of an item-level
collaborative-filtering match.

This exists because item-based ALS was benchmarked against it (see
train.py / docs/ab_test_results.md) and lost by roughly 2x on Recall@12 —
fashion purchases here are driven far more by category-level taste and
what's currently trending than by fine-grained item co-purchase patterns,
so personalizing at the category level captures the taste signal that
item-level CF was too sparse to find.
"""
import pandas as pd


class CategoryAffinityModel:
    def __init__(self, recency_days: int = 30):
        self.recency_days = recency_days
        self.group_popularity: dict[str, list[str]] = {}
        self.customer_fav_group: dict[str, str] = {}

    def fit(self, train: pd.DataFrame, articles: pd.DataFrame) -> "CategoryAffinityModel":
        groups = articles[["article_id", "product_group_name"]]

        cutoff = train["t_dat"].max() - pd.Timedelta(days=self.recency_days)
        recent = train[train["t_dat"] > cutoff].merge(groups, on="article_id", how="left")
        self.group_popularity = (
            recent.groupby("product_group_name")["article_id"]
            .apply(lambda s: s.value_counts().index.tolist())
            .to_dict()
        )

        train_with_group = train.merge(groups, on="article_id", how="left")
        self.customer_fav_group = (
            train_with_group.groupby("customer_id")["product_group_name"]
            .agg(lambda s: s.value_counts().idxmax() if len(s) else None)
            .to_dict()
        )
        return self

    def recommend(self, customer_id: str, k: int, exclude: set[str] | None = None) -> list[str]:
        exclude = exclude or set()
        fav_group = self.customer_fav_group.get(customer_id)
        if not fav_group or fav_group not in self.group_popularity:
            return []
        return [a for a in self.group_popularity[fav_group] if a not in exclude][:k]
