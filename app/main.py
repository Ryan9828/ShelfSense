"""FastAPI serving layer. Loads model artifacts trained by `python -m shelfsense.train`
and serves recommendations for the hybrid model, the popularity baseline, and the
benchmarked (but not shipped) item-CF model, so the frontend demo can show the
offline A/B comparison live.
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shelfsense import config  # noqa: E402
from shelfsense.hybrid import HybridRecommender  # noqa: E402

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["popularity"] = joblib.load(config.ARTIFACTS / "popularity.joblib")
    _state["content"] = joblib.load(config.ARTIFACTS / "content.joblib")
    _state["affinity"] = joblib.load(config.ARTIFACTS / "affinity.joblib")
    _state["als"] = joblib.load(config.ARTIFACTS / "als.joblib")  # kept for the item_cf comparison
    _state["train"] = pd.read_parquet(config.DATA_PROCESSED / "transactions_train.parquet")
    _state["articles"] = pd.read_parquet(config.DATA_PROCESSED / "articles.parquet").set_index(
        "article_id"
    )
    _state["hybrid"] = HybridRecommender(_state["affinity"], _state["content"], _state["popularity"])
    yield
    _state.clear()


app = FastAPI(title="ShelfSense", description="Retail product recommender", lifespan=lifespan)


class RecommendationResponse(BaseModel):
    customer_id: str
    model: str
    article_ids: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": bool(_state)}


@app.get("/customers/sample")
def sample_customers(n: int = 20) -> list[str]:
    train = _state["train"]
    return train["customer_id"].drop_duplicates().head(n).tolist()


@app.get("/recommend/{customer_id}", response_model=RecommendationResponse)
def recommend(customer_id: str, k: int = config.TOP_K, model: str = "hybrid") -> dict:
    history = set(_state["train"].loc[_state["train"]["customer_id"] == customer_id, "article_id"])
    if model == "hybrid":
        article_ids = _state["hybrid"].recommend(customer_id, k, _state["train"])
    elif model == "popularity":
        article_ids = _state["popularity"].recommend(k, exclude=history)
    elif model == "item_cf":
        # benchmarked but not shipped — underperforms popularity, see docs/ab_test_results.md
        article_ids = [a for a, _ in _state["als"].recommend(customer_id, k)]
    else:
        raise HTTPException(400, "model must be 'hybrid', 'popularity', or 'item_cf'")
    return {"customer_id": customer_id, "model": model, "article_ids": article_ids}


@app.get("/articles/{article_id}")
def article_detail(article_id: str) -> dict:
    articles = _state["articles"]
    if article_id not in articles.index:
        raise HTTPException(404, "unknown article_id")
    row = articles.loc[article_id]
    return {
        "article_id": article_id,
        "prod_name": row.get("prod_name"),
        "product_type_name": row.get("product_type_name"),
        "department_name": row.get("department_name"),
        "colour_group_name": row.get("colour_group_name"),
    }
