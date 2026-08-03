"""Cold-start, live: search the real catalog, pick a few items you like, and
get a recommendation for a customer with zero purchase history — this is the
actual hard problem the hybrid's routing logic exists for (see hybrid.py /
recommend_for_selection), demonstrated interactively instead of by looking
up an existing anonymized customer_id.

The basket size is the whole point, so the page shows which branch the current
basket routes to *before* the button is pressed — adding a third item visibly
switches the strategy from content similarity to category affinity.
"""
import streamlit as st

import api
import ui
from routing import MIN_INTERACTIONS_FOR_CF, routing_branch

ui.hero(
    "Try it yourself",
    "Arrive as a brand-new customer with no purchase history. Search the real "
    "H&M catalog, add a few items you like, and watch the hybrid change "
    "strategy as your basket grows — this is the cold-start path, running live.",
)

if "picks" not in st.session_state:
    st.session_state.picks = {}  # article_id -> display dict


def add_pick(item: dict) -> None:
    st.session_state.picks[item["article_id"]] = item


def remove_pick(article_id: str) -> None:
    st.session_state.picks.pop(article_id, None)


def render_pickable(items: list[dict]) -> None:
    for item in items:
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        c1.markdown(ui.product_card(item, compact=True), unsafe_allow_html=True)
        c2.button("Add", key=f"add-{item['article_id']}", on_click=add_pick, args=(item,))


col_search, col_picks = st.columns([3, 2], gap="large")

with col_search:
    ui.section("1 · Pick what you like", "search")
    query = st.text_input(
        "Search the catalog", value="", placeholder='e.g. "jeans", "dress", "hoodie"'
    )
    try:
        if len(query) >= 2:
            results = api.search(query)
            if not results:
                ui.callout("No matches — try a broader term.", icon="search")
            render_pickable(results)
        elif query:
            st.caption("Keep typing — need at least 2 characters.")
        else:
            st.caption("Trending now — or search above for something specific.")
            with st.spinner("Loading the catalog…"):
                trending = api.popular(limit=10)
            render_pickable(trending)
    except api.ApiError as e:
        st.error(str(e))
        st.info(
            "The demo API sleeps when idle on its free tier. If this is the "
            "first load in a while, wait a few seconds and refresh."
        )

with col_picks:
    picks = st.session_state.picks
    ui.section(f"2 · Your basket ({len(picks)})", "bag")

    branch = routing_branch(len(picks))
    ui.stat_cards(
        [
            {
                "label": "Strategy this basket triggers",
                "value": branch["name"],
                "sub": branch["why"],
                "accent": True,
            }
        ]
    )
    if len(picks) < MIN_INTERACTIONS_FOR_CF:
        needed = MIN_INTERACTIONS_FOR_CF - len(picks)
        ui.col_note(
            f"Add <strong>{needed}</strong> more item{'s' if needed > 1 else ''} to cross the "
            f"{MIN_INTERACTIONS_FOR_CF}-item threshold and switch to category affinity."
        )

    if not picks:
        ui.callout("Nothing yet — search on the left and add a few items.", icon="sparkles")
    for aid, item in list(picks.items()):
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        c1.markdown(ui.product_card(item, compact=True), unsafe_allow_html=True)
        c2.button("✕", key=f"remove-{aid}", on_click=remove_pick, args=(aid,), help="Remove")

    if picks:
        if st.button("Clear all"):
            st.session_state.picks = {}
            st.rerun()

st.divider()

if st.button("Get recommendations", type="primary"):
    article_ids = list(st.session_state.picks.keys())
    branch = routing_branch(len(article_ids))
    try:
        with st.spinner("Scoring…"):
            recs = api.recommend_custom(article_ids, k=12)
    except api.ApiError as e:
        st.error(str(e))
        recs = []

    ui.callout(
        f"<strong>{branch['name']}</strong> — {branch['how']}",
        icon="sparkles",
    )

    if recs:
        cols = st.columns(4, gap="medium")
        for i, item in enumerate(recs):
            with cols[i % 4]:
                st.markdown(ui.product_card(item, rank=i + 1, compact=True), unsafe_allow_html=True)

ui.footer()
