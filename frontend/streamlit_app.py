"""Storefront demo: pick a customer, compare three recommenders side by side.
This is the artifact to link in a resume/portfolio — it makes the offline A/B
result tangible instead of a table of numbers, including the negative result
(item-CF loses to popularity) which is the more interesting finding.

Run: streamlit run frontend/streamlit_app.py
Requires the FastAPI service running (default http://localhost:8000, override
with the API_URL env var — set this to the deployed API's URL in production).
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="ShelfSense", page_icon="🛍️", layout="wide")
st.title("🛍️ ShelfSense — retail product recommender")
st.caption(
    "Three recommenders, same customer, same catalog. Hybrid (category-affinity + "
    "content-based cold-start) ties the popularity baseline; pure item-based "
    "collaborative filtering loses to it — see docs/ab_test_results.md for why."
)


@st.cache_data(ttl=60)
def get_sample_customers() -> list[str]:
    resp = requests.get(f"{API_URL}/customers/sample", params={"n": 30}, timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=60)
def get_recommendations(customer_id: str, model: str, k: int = 12) -> list[dict]:
    resp = requests.get(
        f"{API_URL}/recommend/{customer_id}", params={"model": model, "k": k}, timeout=10
    )
    resp.raise_for_status()
    article_ids = resp.json()["article_ids"]
    details = []
    for aid in article_ids:
        d = requests.get(f"{API_URL}/articles/{aid}", timeout=10)
        details.append(d.json() if d.ok else {"article_id": aid})
    return details


try:
    customers = get_sample_customers()
except requests.RequestException as e:
    st.error(f"Could not reach the API at {API_URL}: {e}")
    st.stop()

customer_id = st.selectbox("Customer", customers)

if customer_id:
    col_hybrid, col_pop, col_cf = st.columns(3)
    for col, model_name, label in [
        (col_hybrid, "hybrid", "Hybrid (shipped)"),
        (col_pop, "popularity", "Popularity baseline (control)"),
        (col_cf, "item_cf", "Item-CF (benchmarked, not shipped)"),
    ]:
        with col:
            st.subheader(label)
            items = get_recommendations(customer_id, model_name)
            if not items:
                st.write("No recommendations.")
            for item in items:
                st.markdown(
                    f"**{item.get('prod_name', item['article_id'])}**  \n"
                    f"{item.get('product_type_name', '')} · {item.get('colour_group_name', '')}"
                )
