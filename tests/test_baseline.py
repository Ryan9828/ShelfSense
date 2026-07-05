import pandas as pd

from shelfsense.baseline import PopularityModel


def _transactions():
    return pd.DataFrame(
        {
            "t_dat": pd.to_datetime(
                ["2020-09-01", "2020-09-01", "2020-09-10", "2020-09-10", "2020-09-10"]
            ),
            "customer_id": ["c1", "c2", "c1", "c2", "c3"],
            "article_id": ["a1", "a1", "a2", "a2", "a2"],
        }
    )


def test_recommend_ranks_by_recent_popularity():
    model = PopularityModel(recency_days=30).fit(_transactions())
    assert model.recommend(k=2) == ["a2", "a1"]


def test_recommend_respects_k():
    model = PopularityModel(recency_days=30).fit(_transactions())
    assert model.recommend(k=1) == ["a2"]


def test_recommend_excludes_given_articles():
    model = PopularityModel(recency_days=30).fit(_transactions())
    assert model.recommend(k=2, exclude={"a2"}) == ["a1"]
