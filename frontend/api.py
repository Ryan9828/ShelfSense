"""HTTP client for the ShelfSense API.

Every call the frontend makes lives here so caching, timeouts, and graceful
degradation sit in one file instead of being re-implemented on each page.

Two behaviours worth knowing about:

* **Long timeouts.** The demo API runs on a free tier that sleeps when idle;
  the first request after a quiet spell can take most of a minute to wake it.
  A 10s timeout turns that into a red error box on an app that is actually
  fine, so the wake-up path gets a much longer budget than the warm one.
* **Version skew.** The frontend and the API deploy separately, so the
  frontend can be newer than the API it talks to. Calls to endpoints added
  later fall back to the older, chattier equivalent on 404 rather than
  breaking the page.
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Generous enough to cover a cold free-tier container spinning up.
WAKE_TIMEOUT = 60
TIMEOUT = 20


class ApiError(RuntimeError):
    """Any failure to get a usable answer out of the API."""


def _get(path: str, params: dict | None = None, timeout: int = TIMEOUT) -> requests.Response:
    try:
        return requests.get(f"{API_URL}{path}", params=params or {}, timeout=timeout)
    except requests.RequestException as e:
        raise ApiError(f"Could not reach the API at {API_URL} — {e}") from e


def _json(path: str, params: dict | None = None, timeout: int = TIMEOUT):
    resp = _get(path, params, timeout)
    if not resp.ok:
        raise ApiError(f"{path} returned {resp.status_code}: {resp.text[:200]}")
    return resp.json()


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def customer_profiles(n: int = 24) -> list[dict]:
    """Sample customers with purchase counts, favourite category, and recent items.

    Falls back to /customers/sample (ids only) when talking to an API that
    predates /customers/profiles — the picker then shows counts as unknown
    rather than failing outright.
    """
    resp = _get("/customers/profiles", {"n": n}, timeout=WAKE_TIMEOUT)
    if resp.ok:
        return resp.json()
    if resp.status_code != 404:
        raise ApiError(f"/customers/profiles returned {resp.status_code}")
    ids = _json("/customers/sample", {"n": n}, timeout=WAKE_TIMEOUT)
    return [
        {"customer_id": cid, "n_purchases": None, "top_category": None, "recent": []}
        for cid in ids
    ]


# ---------------------------------------------------------------------------
# Articles
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def articles(ids: tuple[str, ...]) -> list[dict]:
    """Details for many article ids in one call, order preserved."""
    if not ids:
        return []
    resp = _get("/articles/batch", {"ids": ",".join(ids)})
    if resp.ok:
        return resp.json()
    if resp.status_code != 404:
        raise ApiError(f"/articles/batch returned {resp.status_code}")
    out = []
    for aid in ids:  # older API: one request per article
        r = _get(f"/articles/{aid}")
        out.append(r.json() if r.ok else {"article_id": aid, "prod_name": aid})
    return out


@st.cache_data(ttl=300, show_spinner=False)
def search(q: str, limit: int = 15) -> list[dict]:
    return _json("/articles/search", {"q": q, "limit": limit})


@st.cache_data(ttl=300, show_spinner=False)
def popular(limit: int = 15) -> list[dict]:
    return _json("/articles/popular", {"limit": limit}, timeout=WAKE_TIMEOUT)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def recommend(customer_id: str, model: str, k: int = 12) -> list[dict]:
    """Ranked recommendations, already joined to article details.

    Raises ApiError with the API's own message on 503 — item_cf is disabled on
    the memory-capped deployment and says so in the response body, which is
    more useful to show than a generic failure.
    """
    resp = _get(f"/recommend/{customer_id}", {"model": model, "k": k}, timeout=WAKE_TIMEOUT)
    if resp.status_code == 503:
        try:
            raise ApiError(resp.json().get("detail", "Not available in this deployment."))
        except ValueError:
            raise ApiError("Not available in this deployment.") from None
    if not resp.ok:
        raise ApiError(f"/recommend returned {resp.status_code}: {resp.text[:200]}")
    return articles(tuple(resp.json()["article_ids"]))


def recommend_custom(article_ids: list[str], k: int = 12) -> list[dict]:
    """Cold-start recommendations from a hand-picked basket (not cached — the
    basket changes on every interaction)."""
    try:
        resp = requests.post(
            f"{API_URL}/recommend/custom",
            json={"article_ids": article_ids, "k": k},
            timeout=WAKE_TIMEOUT,
        )
    except requests.RequestException as e:
        raise ApiError(f"Could not reach the API at {API_URL} — {e}") from e
    if not resp.ok:
        raise ApiError(f"/recommend/custom returned {resp.status_code}")
    return articles(tuple(resp.json()["article_ids"]))


@st.cache_data(ttl=300, show_spinner=False)
def eval_results() -> dict:
    return _json("/eval-results", timeout=WAKE_TIMEOUT)
