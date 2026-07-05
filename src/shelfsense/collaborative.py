"""Collaborative filtering via implicit-feedback ALS (Hu, Koren & Volinsky, 2008)."""
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy import sparse

from shelfsense import config
from shelfsense.data import IndexMaps


class ALSModel:
    def __init__(
        self,
        factors: int = config.ALS_FACTORS,
        regularization: float = config.ALS_REGULARIZATION,
        iterations: int = config.ALS_ITERATIONS,
        alpha: float = config.ALS_ALPHA,
    ):
        self.model = AlternatingLeastSquares(
            factors=factors, regularization=regularization, iterations=iterations
        )
        self.alpha = alpha
        self.idx: IndexMaps | None = None
        self.interactions: sparse.csr_matrix | None = None

    def fit(self, interactions: sparse.csr_matrix, idx: IndexMaps) -> "ALSModel":
        # implicit's own confidence weighting: C = 1 + alpha * count
        confidence = interactions.copy()
        confidence.data = 1.0 + self.alpha * confidence.data
        self.model.fit(confidence)
        self.idx = idx
        self.interactions = interactions
        return self

    def recommend(self, customer_id: str, k: int) -> list[tuple[str, float]]:
        if self.idx is None or customer_id not in self.idx.customer_to_idx:
            return []
        cidx = self.idx.customer_to_idx[customer_id]
        user_items = self.interactions[cidx]
        item_ids, scores = self.model.recommend(
            cidx, user_items, N=k, filter_already_liked_items=True
        )
        return [
            (self.idx.article_ids[i], float(s)) for i, s in zip(item_ids, scores) if s > 0
        ]

    def n_interactions(self, customer_id: str) -> int:
        if self.idx is None or customer_id not in self.idx.customer_to_idx:
            return 0
        cidx = self.idx.customer_to_idx[customer_id]
        return int(self.interactions[cidx].nnz)
