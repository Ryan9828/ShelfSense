import pandas as pd

from shelfsense.hybrid import HybridRecommender


class FakeAffinity:
    def __init__(self, recs: dict | None = None, groups: dict | None = None, group_recs: dict | None = None):
        self.recs = recs or {}
        self.groups = groups or {}  # article_id -> group
        self.group_recs = group_recs or {}  # group -> candidates

    def recommend(self, customer_id, k, exclude=None):
        exclude = exclude or set()
        candidates = self.recs.get(customer_id, [])
        return [a for a in candidates if a not in exclude][:k]

    def favorite_group(self, article_ids):
        groups = [self.groups[a] for a in article_ids if a in self.groups]
        return max(set(groups), key=groups.count) if groups else None

    def recommend_for_group(self, group, k, exclude=None):
        exclude = exclude or set()
        candidates = self.group_recs.get(group, [])
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


def test_selection_with_enough_items_uses_affinity_group():
    affinity = FakeAffinity(
        groups={"x1": "Denim", "x2": "Denim", "x3": "Denim"},
        group_recs={"Denim": ["cf1", "cf2"]},
    )
    hybrid = HybridRecommender(affinity, FakeContent(), FakePopularity(), min_interactions_for_cf=3)
    recs = hybrid.recommend_for_selection(["x1", "x2", "x3"], k=2)
    assert recs == ["cf1", "cf2"]


def test_selection_with_little_history_uses_content():
    affinity = FakeAffinity(recs={})
    hybrid = HybridRecommender(affinity, FakeContent(), FakePopularity(), min_interactions_for_cf=3)
    recs = hybrid.recommend_for_selection(["b1"], k=1)
    assert recs == ["similar-to-b1"]


def test_selection_empty_falls_back_to_popularity():
    affinity = FakeAffinity(recs={})
    hybrid = HybridRecommender(affinity, FakeContent(), FakePopularity(), min_interactions_for_cf=3)
    assert hybrid.recommend_for_selection([], k=2) == ["pop1", "pop2"]
