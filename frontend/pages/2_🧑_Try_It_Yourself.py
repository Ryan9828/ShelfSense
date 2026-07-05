"""Cold-start, live: search the real catalog, pick a few items you like, and
get a recommendation for a customer with zero purchase history — this is the
actual hard problem the hybrid's routing logic exists for (see hybrid.py /
recommend_for_selection), demonstrated interactively instead of by looking
up an existing anonymized customer_id.
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="ShelfSense — Try It Yourself", page_icon="🧑", layout="wide")
st.title("🧑 Try it yourself")
st.caption(
    "Search the real H&M catalog, pick a few items you like, and get a live "
    "recommendation — as if you were a brand new customer with no purchase history."
)

if "picks" not in st.session_state:
    st.session_state.picks = {}  # article_id -> display dict


def add_pick(item: dict) -> None:
    st.session_state.picks[item["article_id"]] = item


def remove_pick(article_id: str) -> None:
    st.session_state.picks.pop(article_id, None)


@st.cache_data(ttl=300)
def search(q: str) -> list[dict]:
    resp = requests.get(f"{API_URL}/articles/search", params={"q": q, "limit": 15}, timeout=10)
    resp.raise_for_status()
    return resp.json()


col_search, col_picks = st.columns([3, 2])

with col_search:
    st.subheader("Search products")
    query = st.text_input(
        "Search query", value="", placeholder='e.g. "jeans", "dress", "hoodie"',
        label_visibility="collapsed",
    )
    if len(query) >= 2:
        try:
            results = search(query)
        except requests.RequestException as e:
            st.error(f"Could not reach the API at {API_URL}: {e}")
            results = []
        if not results:
            st.write("No matches.")
        for item in results:
            c1, c2 = st.columns([4, 1])
            c1.markdown(
                f"**{item['prod_name']}** — {item['product_type_name']} · {item['colour_group_name']}"
            )
            c2.button("Add", key=f"add-{item['article_id']}", on_click=add_pick, args=(item,))
    elif query:
        st.caption("Keep typing — need at least 2 characters.")

with col_picks:
    st.subheader(f"Your picks ({len(st.session_state.picks)})")
    if not st.session_state.picks:
        st.write("Nothing yet — search on the left and add a few items.")
    for aid, item in list(st.session_state.picks.items()):
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**{item['prod_name']}** ({item['product_type_name']})")
        c2.button("✕", key=f"remove-{aid}", on_click=remove_pick, args=(aid,), help="Remove")

    if st.session_state.picks:
        if st.button("Clear all"):
            st.session_state.picks = {}
            st.rerun()

st.divider()

if st.button("Get recommendations", type="primary"):
    article_ids = list(st.session_state.picks.keys())
    try:
        resp = requests.post(
            f"{API_URL}/recommend/custom", json={"article_ids": article_ids, "k": 12}, timeout=10
        )
        resp.raise_for_status()
        recs = resp.json()["article_ids"]
    except requests.RequestException as e:
        st.error(f"Could not reach the API at {API_URL}: {e}")
        recs = []

    if not article_ids:
        st.info("You picked nothing, so this is just the popularity baseline — the fallback "
                 "for a customer we truly have no signal for.")
    elif len(article_ids) < 3:
        st.info("With only 1-2 picks, the hybrid uses content-based similarity to those "
                 "specific items rather than category-level personalization.")
    else:
        st.info("With 3+ picks, the hybrid infers your favorite category and recommends "
                 "trending items within it.")

    if recs:
        cols = st.columns(4)
        for i, aid in enumerate(recs):
            detail = requests.get(f"{API_URL}/articles/{aid}", timeout=10)
            item = detail.json() if detail.ok else {"article_id": aid}
            with cols[i % 4]:
                st.markdown(
                    f"**{item.get('prod_name', aid)}**  \n"
                    f"{item.get('product_type_name', '')} · {item.get('colour_group_name', '')}"
                )
