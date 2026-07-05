import pandas as pd

from shelfsense.hybrid import HybridRecommender


class FakeAffinity:
    def __init__(self, recs: dict):
        self.recs = recs

    def recommend(self, customer_id, k, exclude=None):
        exclude = exclude or set()
        candidates = self.recs.get(customer_id, [])
        return [a for a in candidates if a not in exclude][:k]


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


def test_warm_customer_uses_affinity():
    affinity = FakeAffinity(recs={"warm": ["cf1", "cf2"]})
    hybrid = HybridRecommender(affinity, FakeContent(), FakePopularity(), min_interactions_for_cf=3)
    recs = hybrid.recommend("warm", k=2, train=_train_df())
    assert recs == ["cf1", "cf2"]


def test_warm_customer_tops_up_with_popularity_if_affinity_short():
    affinity = FakeAffinity(recs={"warm": ["cf1"]})
    hybrid = HybridRecommender(affinity, FakeContent(), FakePopularity(), min_interactions_for_cf=3)
    recs = hybrid.recommend("warm", k=3, train=_train_df())
    assert recs[0] == "cf1"
    assert set(recs[1:]) <= {"pop1", "pop2", "pop3"}


def test_cold_customer_with_history_uses_content():
    affinity = FakeAffinity(recs={})
    hybrid = HybridRecommender(affinity, FakeContent(), FakePopularity(), min_interactions_for_cf=3)
    recs = hybrid.recommend("cold_with_history", k=1, train=_train_df())
    assert recs == ["similar-to-b1"]


def test_fully_cold_customer_falls_back_to_popularity():
    affinity = FakeAffinity(recs={})
    hybrid = HybridRecommender(affinity, FakeContent(), FakePopularity(), min_interactions_for_cf=3)
    recs = hybrid.recommend("never-seen", k=2, train=_train_df())
    assert recs == ["pop1", "pop2"]
