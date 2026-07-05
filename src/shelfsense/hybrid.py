"""Routes each customer to a different recommendation strategy depending on
how much purchase history we have for them — this is the actual cold-start
strategy, not just a modeling detail: a system with a single strategy either
has nothing sensible to say to a customer with 0-1 purchases, or (as the
benchmark in train.py shows) wastes its best signal on customers where it
doesn't actually help.

Warm customers use category-affinity popularity, not item-based collaborative
filtering — see affinity.py for why: ALS was benchmarked against it and lost.
"""
import pandas as pd

from shelfsense import config
from shelfsense.affinity import CategoryAffinityModel
from shelfsense.baseline import PopularityModel
from shelfsense.content import ContentModel


class HybridRecommender:
    def __init__(
        self,
        affinity: CategoryAffinityModel,
        content: ContentModel,
        popularity: PopularityModel,
        min_interactions_for_cf: int = config.MIN_INTERACTIONS_FOR_CF,
    ):
        self.affinity = affinity
        self.content = content
        self.popularity = popularity
        self.min_interactions_for_cf = min_interactions_for_cf

    def _purchase_history(self, customer_id: str, train: pd.DataFrame) -> list[str]:
        return train.loc[train["customer_id"] == customer_id, "article_id"].tolist()

    def recommend(self, customer_id: str, k: int, train: pd.DataFrame) -> list[str]:
        history = self._purchase_history(customer_id, train)
        exclude = set(history)

        if len(history) >= self.min_interactions_for_cf:
            # warm customer: best-sellers in their favorite category, topped
            # up with global popularity only if that category is too small
            recs = self.affinity.recommend(customer_id, k, exclude=exclude)
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

    def recommend_for_selection(self, liked_article_ids: list[str], k: int) -> list[str]:
        """Same tiering logic as recommend(), but for a customer we have no stored
        purchase history for — e.g. a live demo query built from items someone just
        picked. This *is* the cold-start problem the project is about, just triggered
        interactively instead of by looking up an existing customer_id."""
        exclude = set(liked_article_ids)

        if len(liked_article_ids) >= self.min_interactions_for_cf:
            fav_group = self.affinity.favorite_group(liked_article_ids)
            recs = self.affinity.recommend_for_group(fav_group, k, exclude=exclude)
            if len(recs) < k:
                recs += self.popularity.recommend(k - len(recs), exclude=exclude | set(recs))
            return recs[:k]

        if liked_article_ids:
            recs = [a for a, _ in self.content.recommend_similar(liked_article_ids, k, exclude=exclude)]
            if len(recs) < k:
                recs += self.popularity.recommend(k - len(recs), exclude=exclude | set(recs))
            return recs[:k]

        return self.popularity.recommend(k, exclude=exclude)
