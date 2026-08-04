"""FastAPI serving layer. Loads model artifacts trained by `python -m shelfsense.train`
and serves recommendations for the hybrid model, the popularity baseline, and the
benchmarked (but not shipped) ALS model, so the frontend demo can show the
offline A/B comparison live.
"""
import ctypes
import gc
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shelfsense import config  # noqa: E402
from shelfsense.hybrid import HybridRecommender  # noqa: E402

_state: dict = {}


def _release_freed_memory() -> None:
    """gc.collect() alone doesn't return freed pages to the OS — glibc's malloc
    keeps them in its own free list for reuse. On a memory-capped host (Render's
    512MB free tier) that means a transient peak during startup (e.g. materializing
    a large column before converting it to a smaller dtype) permanently inflates
    RSS for the rest of the process's life, even once the Python objects are gone.
    malloc_trim forces glibc to actually give that memory back."""
    gc.collect()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (OSError, AttributeError):
        pass  # not glibc (e.g. running this on macOS) — nothing to do


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["popularity"] = joblib.load(config.ARTIFACTS / "popularity.joblib")
    _state["content"] = joblib.load(config.ARTIFACTS / "content.joblib")
    _state["affinity"] = joblib.load(config.ARTIFACTS / "affinity.joblib")
    if config.LOAD_ITEM_CF:
        _state["als"] = joblib.load(config.ARTIFACTS / "als.joblib")
    _release_freed_memory()

    # Only customer_id/article_id are ever read from `train` at serving time (purchase
    # history lookups). Reading via pyarrow with read_dictionary + converting straight
    # to pandas `category` avoids ever materializing the full 1.39M-row object-dtype
    # string columns (~250MB) that a plain `pd.read_parquet(...).astype("category")`
    # would transiently hold before conversion — that peak, not the ~26MB steady state,
    # is what was pushing the container over Render's memory limit.
    table = pq.read_table(
        config.DATA_PROCESSED / "transactions_train.parquet",
        columns=["customer_id", "article_id"],
        read_dictionary=["customer_id", "article_id"],
    )
    _state["train"] = table.to_pandas()
    del table
    _release_freed_memory()

    # The storefront view needs three more fields than the models do:
    # product_group_name (garment silhouette), index_group_name (department),
    # and retail_price. All three are cheap here — the first two are 12- and
    # 5-value categoricals, and retail_price is precomputed at build time by
    # data.attach_retail_price rather than aggregated from the 1.39M-row price
    # column at boot, which cost ~50MB of peak RSS for a result that never
    # changes.
    #
    # read_dictionary on the low-cardinality string columns matters as much as
    # the column list does: read as plain object dtype these cost ~18MB of
    # retained Python strings, and dictionary-encoding them makes that ~0.
    _articles_cols = [
        "article_id", "prod_name", "product_type_name", "colour_group_name",
        "department_name", "product_group_name", "index_group_name", "retail_price",
    ]
    _articles_dict_cols = [
        "product_type_name", "colour_group_name", "department_name",
        "product_group_name", "index_group_name",
    ]
    articles_table = pq.read_table(
        config.DATA_PROCESSED / "articles.parquet",
        columns=_articles_cols,
        read_dictionary=_articles_dict_cols,
    )
    _state["articles"] = articles_table.to_pandas().set_index("article_id")
    del articles_table
    _release_freed_memory()

    _state["hybrid"] = HybridRecommender(_state["affinity"], _state["content"], _state["popularity"])
    yield
    _state.clear()


app = FastAPI(title="ShelfSense", description="Retail product recommender", lifespan=lifespan)


class RecommendationResponse(BaseModel):
    customer_id: str
    model: str
    article_ids: list[str]


class CustomRecommendationRequest(BaseModel):
    article_ids: list[str]
    k: int = config.TOP_K


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": bool(_state)}


def _unknown_article(article_id: str) -> dict:
    """Same shape as _article_dict for an id not in the catalog, so callers
    never have to branch on missing keys."""
    return {
        "article_id": article_id,
        "prod_name": article_id,
        "product_type_name": "",
        "product_group_name": "",
        "index_group_name": "",
        "department_name": "",
        "colour_group_name": "",
        "price": None,
    }


def _article_dict(article_id: str) -> dict:
    row = _state["articles"].loc[article_id]
    price = row.get("retail_price")
    return {
        "article_id": article_id,
        "prod_name": row.get("prod_name"),
        "product_type_name": row.get("product_type_name"),
        "product_group_name": row.get("product_group_name"),
        "index_group_name": row.get("index_group_name"),
        "department_name": row.get("department_name"),
        "colour_group_name": row.get("colour_group_name"),
        # None for an article never sold in the training window
        "price": None if price is None or pd.isna(price) else float(price),
    }


@app.get("/customers/sample")
def sample_customers(n: int = 20) -> list[str]:
    train = _state["train"]
    return train["customer_id"].drop_duplicates().head(n).tolist()


@app.get("/customers/profiles")
def customer_profiles(n: int = 20, history: int = 8) -> list[dict]:
    """Sampled customers with enough context for a UI to introduce them by
    something other than a 64-char hash: how much history each has, what they
    buy most, and a few of the actual items. Also the only way to judge whether
    a recommendation is any good — a list of products means nothing without
    knowing what the customer already bought.
    """
    train = _state["train"]
    articles = _state["articles"]
    ids = train["customer_id"].drop_duplicates().head(n).tolist()
    rows = train[train["customer_id"].isin(ids)]
    # observed=True is load-bearing: customer_id is a categorical over all ~60k
    # customers, so a default groupby would walk every unused category too.
    grouped = rows.groupby("customer_id", observed=True)["article_id"].apply(list)

    profiles = []
    for cid in ids:
        aids = [str(a) for a in grouped.get(cid, [])]
        types = articles["product_type_name"].reindex(aids).dropna()
        top = types.value_counts()
        # transactions_train.parquet is in chronological order and t_dat isn't
        # loaded at serving time, so "most recent" is the tail of their rows.
        recent: list[dict] = []
        seen: set[str] = set()
        for aid in reversed(aids):
            if aid in seen or aid not in articles.index:
                continue
            seen.add(aid)
            recent.append(_article_dict(aid))
            if len(recent) >= history:
                break
        profiles.append(
            {
                "customer_id": cid,
                "n_purchases": len(aids),
                "n_distinct_articles": len(set(aids)),
                "top_category": str(top.index[0]) if len(top) else None,
                "top_category_share": round(float(top.iloc[0] / top.sum()), 3) if len(top) else None,
                "recent": recent,
            }
        )
    return profiles


@app.get("/recommend/{customer_id}", response_model=RecommendationResponse)
def recommend(customer_id: str, k: int = config.TOP_K, model: str = "hybrid") -> dict:
    history = set(_state["train"].loc[_state["train"]["customer_id"] == customer_id, "article_id"])
    if model == "hybrid":
        article_ids = _state["hybrid"].recommend(customer_id, k, _state["train"])
    elif model == "popularity":
        article_ids = _state["popularity"].recommend(k, exclude=history)
    elif model == "item_cf":
        # benchmarked but not shipped — underperforms popularity, see docs/ab_test_results.md
        if "als" not in _state:
            raise HTTPException(
                503,
                "item_cf comparison is disabled in this deployment to fit the free-tier "
                "memory budget — see docs/ab_test_results.md for the benchmark numbers.",
            )
        article_ids = [a for a, _ in _state["als"].recommend(customer_id, k)]
    else:
        raise HTTPException(400, "model must be 'hybrid', 'popularity', or 'item_cf'")
    return {"customer_id": customer_id, "model": model, "article_ids": article_ids}


@app.get("/eval-results")
def eval_results() -> dict:
    path = config.ARTIFACTS / "eval_results.json"
    if not path.exists():
        raise HTTPException(404, "No eval_results.json yet — run `python -m shelfsense.train` first.")
    return json.loads(path.read_text())


@app.get("/articles/search")
def search_articles(q: str, limit: int = 20) -> list[dict]:
    """Powers the 'pick items you like' demo flow — a real product search index
    would replace this, but a substring match is plenty for ~75k articles."""
    articles = _state["articles"]
    if len(q) < 2:
        return []
    mask = (
        articles["prod_name"].str.contains(q, case=False, na=False)
        | articles["product_type_name"].str.contains(q, case=False, na=False)
    )
    return [_article_dict(aid) for aid in articles[mask].head(limit).index]


@app.get("/articles/popular")
def popular_articles(limit: int = 20) -> list[dict]:
    """A browsable default list for the 'pick items you like' flow — so there's
    something to click before typing anything into search."""
    articles = _state["articles"]
    return [
        _article_dict(aid) if aid in articles.index else _unknown_article(aid)
        for aid in _state["popularity"].ranked_articles[:limit]
    ]


@app.get("/articles/batch")
def articles_batch(ids: str) -> list[dict]:
    """Details for a comma-separated list of article ids, in the order given.

    Declared above /articles/{article_id} so FastAPI doesn't route "batch" into
    the path param. The storefront draws 12 recommendations for each of three
    models — without this it made 36 sequential round trips to render one page.
    """
    articles = _state["articles"]
    return [
        _article_dict(aid) if aid in articles.index else _unknown_article(aid)
        for aid in (i for i in ids.split(",") if i)
    ]


@app.post("/recommend/custom", response_model=RecommendationResponse)
def recommend_custom(req: CustomRecommendationRequest) -> dict:
    """Recommendations for a customer with no stored history — e.g. a brand
    new visitor who just picked a few items they like. This is the cold-start
    case the project's hybrid routing exists for, triggered live instead of
    by looking up an existing customer_id."""
    article_ids = _state["hybrid"].recommend_for_selection(req.article_ids, req.k)
    return {"customer_id": "custom", "model": "hybrid", "article_ids": article_ids}


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
