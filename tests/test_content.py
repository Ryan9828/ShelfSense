import pandas as pd

from shelfsense.content import ContentModel


def _articles():
    return pd.DataFrame(
        {
            "article_id": ["a1", "a2", "a3"],
            "prod_name": ["Blue Jeans", "Blue Denim Jeans", "Red Silk Scarf"],
            "product_type_name": ["Trousers", "Trousers", "Accessory"],
            "product_group_name": ["Garment Lower", "Garment Lower", "Accessories"],
            "department_name": ["Menswear", "Menswear", "Womenswear"],
            "index_name": ["Men", "Men", "Women"],
            "garment_group_name": ["Denim", "Denim", "Accessories"],
            "detail_desc": ["classic fit", "slim fit", "lightweight scarf"],
        }
    )


def test_recommends_similar_item_over_dissimilar():
    model = ContentModel().fit(_articles())
    recs = model.recommend_similar(["a1"], k=2)
    rec_ids = [aid for aid, _ in recs]
    assert rec_ids[0] == "a2"  # near-duplicate jeans should outrank the scarf


def test_excludes_seed_and_excluded_items():
    model = ContentModel().fit(_articles())
    recs = model.recommend_similar(["a1"], k=3, exclude={"a2"})
    rec_ids = [aid for aid, _ in recs]
    assert "a1" not in rec_ids
    assert "a2" not in rec_ids


def test_unknown_seed_returns_empty():
    model = ContentModel().fit(_articles())
    assert model.recommend_similar(["does-not-exist"], k=3) == []
