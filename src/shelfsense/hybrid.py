"""Blends collaborative, content-based, and popularity signals depending on how
much history we have for a customer — this is the actual cold-start strategy,
not just a modeling detail: a pure-CF system recommends nothing sensible to a
customer with zero or one purchases, which in production is a large fraction
of daily active users on any given day.
"""
import pandas as pd

from shelfsense import config
from shelfsense.baseline import PopularityModel
from shelfsense.collaborative import ALSModel
from shelfsense.content import ContentModel


class HybridRecommender:
    def __init__(
        self,
        als: ALSModel,
        content: ContentModel,
        popularity: PopularityModel,
        min_interactions_for_cf: int = config.MIN_INTERACTIONS_FOR_CF,
    ):
        self.als = als
        self.content = content
        self.popularity = popularity
        self.min_interactions_for_cf = min_interactions_for_cf

    def _purchase_history(self, customer_id: str, train: pd.DataFrame) -> list[str]:
        return train.loc[train["customer_id"] == customer_id, "article_id"].tolist()

    def recommend(self, customer_id: str, k: int, train: pd.DataFrame) -> list[str]:
        n_interactions = self.als.n_interactions(customer_id)
        history = self._purchase_history(customer_id, train)
        exclude = set(history)

        if n_interactions >= self.min_interactions_for_cf:
            # warm customer: trust collaborative filtering, top up with
            # popularity only if ALS couldn't fill k slots
            recs = [a for a, _ in self.als.recommend(customer_id, k)]
            if len(recs) < k:
                recs += self.popularity.recommend(k - len(recs), exclude=exclude | set(recs))
            return recs[:k]

        if history:
            # cold-ish customer with a little history: content similarity to what
            # they've already bought, topped up with popularity for diversity
            recs = [a for a, _ in self.content.recommend_similar(history, k, exclude=exclude)]
            if len(recs) < k:
                recs += self.popularity.recommend(k - len(recs), exclude=exclude | set(recs))
            return recs[:k]

        # fully cold customer: no signal at all, fall back to popularity
        return self.popularity.recommend(k, exclude=exclude)
