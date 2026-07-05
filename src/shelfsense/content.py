"""Content-based similarity, used for cold-start customers/articles that ALS
has no interaction history for. Built from article text metadata only, so it
works even for an article that sold zero units in the training window.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

TEXT_FIELDS = [
    "prod_name",
    "product_type_name",
    "product_group_name",
    "department_name",
    "index_name",
    "garment_group_name",
    "detail_desc",
]


class ContentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=20_000, stop_words="english")
        self.article_ids: np.ndarray | None = None
        self.article_to_row: dict | None = None
        self.matrix = None  # sparse, rows=articles, cols=tfidf terms

    def fit(self, articles: pd.DataFrame) -> "ContentModel":
        text = articles[TEXT_FIELDS].fillna("").agg(" ".join, axis=1)
        self.matrix = self.vectorizer.fit_transform(text)
        self.article_ids = articles["article_id"].to_numpy()
        self.article_to_row = {a: i for i, a in enumerate(self.article_ids)}
        return self

    def recommend_similar(
        self, seed_article_ids: list[str], k: int, exclude: set[str] | None = None
    ) -> list[tuple[str, float]]:
        """Average the TF-IDF vectors of a customer's known purchases and return
        the k nearest unpurchased articles by cosine similarity."""
        exclude = (exclude or set()) | set(seed_article_ids)  # never recommend a seed item back to itself
        rows = [self.article_to_row[a] for a in seed_article_ids if a in self.article_to_row]
        if not rows:
            return []
        profile = self.matrix[rows].mean(axis=0)
        profile = np.asarray(profile)  # np.matrix -> ndarray, shape (1, n_terms)
        sims = self.matrix.dot(profile.T).ravel()  # cosine numerator; TF-IDF rows are L2-normed
        order = np.argsort(-sims)
        out = []
        for i in order:
            aid = self.article_ids[i]
            if aid in exclude or sims[i] <= 0:
                continue
            out.append((aid, float(sims[i])))
            if len(out) >= k:
                break
        return out
