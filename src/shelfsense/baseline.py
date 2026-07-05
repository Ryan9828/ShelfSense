"""Popularity baseline — the control arm for the offline A/B comparison.

Recommending the same best-sellers to everyone is what most naive
"recommendation" implementations actually ship as v0; it's the honest
bar the hybrid model has to clear.
"""
import pandas as pd


class PopularityModel:
    def __init__(self, recency_days: int = 30):
        self.recency_days = recency_days
        self.ranked_articles: list[str] = []

    def fit(self, train: pd.DataFrame) -> "PopularityModel":
        cutoff = train["t_dat"].max() - pd.Timedelta(days=self.recency_days)
        recent = train[train["t_dat"] > cutoff]
        counts = recent["article_id"].value_counts()
        if counts.empty:  # fall back to all-time if the recency window is empty
            counts = train["article_id"].value_counts()
        self.ranked_articles = counts.index.tolist()
        return self

    def recommend(self, k: int, exclude: set[str] | None = None) -> list[str]:
        exclude = exclude or set()
        out = [a for a in self.ranked_articles if a not in exclude]
        return out[:k]
