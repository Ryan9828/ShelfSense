import pandas as pd

from shelfsense.hybrid import HybridRecommender


class FakeALS:
    def __init__(self, interactions: dict, recs: dict):
        self.interactions = interactions
        self.recs = recs

    def n_interactions(self, customer_id):
        return self.interactions.get(customer_id, 0)

    def recommend(self, customer_id, k):
        return self.recs.get(customer_id, [])[:k]


class FakeContent:
    def recommend_similar(self, seed_article_ids, k, exclude=None):
        exclude = exclude or set()
        candidates = [(f"similar-to-{a}", 1.0) for a in seed_article_ids]
        return [(cid, score) for cid, score in candidates if cid not in exclude][:k]


class FakePopularity:
    def recommend(self, k, exclude=None):
        exclude = exclude or set()
        return [a for a in ["pop1", "pop2", "pop3"] if a not in exclude][:k]


def _train_df():
    return pd.DataFrame(
        {
            "customer_id": ["warm", "warm", "warm", "cold_with_history"],
            "article_id": ["a1", "a2", "a3", "b1"],
        }
    )


def test_warm_customer_uses_als():
    als = FakeALS(interactions={"warm": 5}, recs={"warm": [("cf1", 0.9), ("cf2", 0.8)]})
    hybrid = HybridRecommender(als, FakeContent(), FakePopularity(), min_interactions_for_cf=3)
    recs = hybrid.recommend("warm", k=2, train=_train_df())
    assert recs == ["cf1", "cf2"]


def test_warm_customer_tops_up_with_popularity_if_als_short():
    als = FakeALS(interactions={"warm": 5}, recs={"warm": [("cf1", 0.9)]})
    hybrid = HybridRecommender(als, FakeContent(), FakePopularity(), min_interactions_for_cf=3)
    recs = hybrid.recommend("warm", k=3, train=_train_df())
    assert recs[0] == "cf1"
    assert set(recs[1:]) <= {"pop1", "pop2", "pop3"}


def test_cold_customer_with_history_uses_content():
    als = FakeALS(interactions={"cold_with_history": 1}, recs={})
    hybrid = HybridRecommender(als, FakeContent(), FakePopularity(), min_interactions_for_cf=3)
    recs = hybrid.recommend("cold_with_history", k=1, train=_train_df())
    assert recs == ["similar-to-b1"]


def test_fully_cold_customer_falls_back_to_popularity():
    als = FakeALS(interactions={}, recs={})
    hybrid = HybridRecommender(als, FakeContent(), FakePopularity(), min_interactions_for_cf=3)
    recs = hybrid.recommend("never-seen", k=2, train=_train_df())
    assert recs == ["pop1", "pop2"]
