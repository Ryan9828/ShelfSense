import pandas as pd

from shelfsense.affinity import CategoryAffinityModel


def _articles():
    return pd.DataFrame(
        {
            "article_id": ["a1", "a2", "a3", "b1"],
            "product_group_name": ["Garment Lower", "Garment Lower", "Garment Lower", "Accessories"],
        }
    )


def _train():
    return pd.DataFrame(
        {
            "t_dat": pd.to_datetime(["2020-09-01"] * 5),
            "customer_id": ["c1", "c1", "c2", "c2", "c2"],
            # c1 buys only Garment Lower -> favorite group is Garment Lower
            # c2 buys mostly Accessories via repeated a1/a3/b1 purchases below
            "article_id": ["a1", "a2", "a1", "a3", "b1"],
        }
    )


def test_recommends_within_customers_favorite_group():
    model = CategoryAffinityModel(recency_days=30).fit(_train(), _articles())
    recs = model.recommend("c1", k=5)
    assert set(recs) <= {"a1", "a2", "a3"}  # only Garment Lower items
    assert "b1" not in recs


def test_excludes_given_articles():
    model = CategoryAffinityModel(recency_days=30).fit(_train(), _articles())
    recs = model.recommend("c1", k=5, exclude={"a1", "a2"})
    assert recs == ["a3"]


def test_unknown_customer_returns_empty():
    model = CategoryAffinityModel(recency_days=30).fit(_train(), _articles())
    assert model.recommend("never-seen", k=5) == []
