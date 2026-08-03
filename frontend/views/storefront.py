"""Storefront: pick a customer, compare three recommenders side by side.

The page is built around one problem: a ranked list of product names is
meaningless on its own. So each customer is introduced by their actual
shopping behaviour (how much history, what they buy, which branch of the
hybrid that routes them down) before any recommendation is shown, the raw
64-char customer hash is demoted to a provenance chip, and each column is
folded to its top 6 with the tail behind a single toggle — the fold is what
keeps three 12-item lists from becoming one very long scroll.
"""
import streamlit as st

import api
import ui
from routing import routing_branch

MODELS = [
    ("hybrid", "Hybrid", "Shipped", "shipped"),
    ("popularity", "Popularity", "Control", "control"),
    ("item_cf", "Item-CF", "Not shipped", "bench"),
]
FOLD = 6  # cards shown per column before the "show all 12" toggle
K = 12  # always fetch 12 — it's the rank the offline metrics are measured at

ui.hero(
    "Compare the recommenders",
    "Three models, the same customer, the same catalog. The shipped hybrid "
    "(category-affinity + content-based cold-start) ties the popularity "
    "baseline; pure item-based collaborative filtering loses to it. Pick a "
    "customer and see what each one would actually put in front of them.",
)

try:
    profiles = api.customer_profiles(n=24)
except api.ApiError as e:
    st.error(str(e))
    st.info(
        "The demo API sleeps when idle on its free tier. If this is the first "
        "load in a while, wait a few seconds and refresh — it is waking up."
    )
    st.stop()

if not profiles:
    st.warning("The API returned no customers.")
    st.stop()


def customer_label(p: dict) -> str:
    """A person, not a hash. Falls back gracefully if the API is an older
    build that only returns ids (see api.customer_profiles)."""
    n = p.get("n_purchases")
    if n is None:
        return f"Customer {p['customer_id'][:8]}…"
    cat = p.get("top_category") or "mixed"
    return f"{n} purchases · mostly {cat}"


# --------------------------------------------------------------- pick a customer
pick_col, toggle_col = st.columns([3, 2], gap="large", vertical_alignment="bottom")
with pick_col:
    idx = st.selectbox(
        "Customer",
        range(len(profiles)),
        format_func=lambda i: f"Customer {i + 1:02d} — {customer_label(profiles[i])}",
        help=(
            "A sample of real anonymized H&M customers from the training data, "
            "labelled by their own purchase history rather than by their id."
        ),
    )
with toggle_col:
    show_all = st.toggle(
        f"Show all {K} per model",
        value=False,
        help=f"Recall@{K} and NDCG@{K} are measured over all {K}; the top {FOLD} is just the fold.",
    )

profile = profiles[idx]
customer_id = profile["customer_id"]
n_purchases = profile.get("n_purchases")
branch = routing_branch(n_purchases)

# --------------------------------------------------------------- who they are
if n_purchases is not None:
    ui.stat_cards(
        [
            {
                "label": "Purchases on record",
                "value": f"{n_purchases:,}",
                "sub": f"{profile.get('n_distinct_articles', 0):,} distinct items",
            },
            {
                "label": "Buys most often",
                "value": profile.get("top_category") or "Mixed",
                "sub": (
                    f"{profile['top_category_share']:.0%} of their basket"
                    if profile.get("top_category_share")
                    else "no dominant category"
                ),
            },
            {
                "label": "Hybrid routes them to",
                "value": branch["name"],
                "sub": branch["why"],
                "accent": True,
            },
        ]
    )
ui.id_chip("customer_id", customer_id)

if profile.get("recent"):
    with st.expander(f"What Customer {idx + 1:02d} has actually bought — most recent first"):
        st.caption(
            "The point of comparison: a good recommendation should look like a "
            "plausible next purchase for this person."
        )
        hist_cols = st.columns(2, gap="medium")
        for i, item in enumerate(profile["recent"]):
            with hist_cols[i % 2]:
                st.markdown(ui.product_card(item, compact=True), unsafe_allow_html=True)

st.divider()

# --------------------------------------------------------------- the three models
# Fetch first, render second: the overlap footnote under each column needs the
# popularity list to compare against, and one spinner beats three.
with st.spinner("Scoring all three models…"):
    results: dict[str, list[dict] | str] = {}
    for model, *_ in MODELS:
        try:
            results[model] = api.recommend(customer_id, model, k=K)
        except api.ApiError as e:
            results[model] = str(e)

pop_ids = (
    {i["article_id"] for i in results["popularity"]}
    if isinstance(results["popularity"], list)
    else set()
)

for col, (model, label, badge_text, badge_kind) in zip(st.columns(3, gap="medium"), MODELS):
    with col:
        ui.model_header(label, badge_text, badge_kind)
        items = results[model]

        if isinstance(items, str):
            ui.callout(items, kind="amber", icon="alert")
            continue
        if not items:
            ui.callout("No recommendations for this customer.", icon="alert")
            continue

        shown = items if show_all else items[:FOLD]
        ui.product_list(shown, compact=True)
        if not show_all and len(items) > FOLD:
            ui.col_note(f"+ {len(items) - FOLD} more — use the toggle above to see all {K}.")

        if model != "popularity" and pop_ids:
            overlap = len({i["article_id"] for i in items} & pop_ids)
            ui.col_note(
                f"<strong>{overlap}/{len(items)}</strong> of these are also in the "
                "popularity baseline — the rest is what personalization actually changed."
            )

ui.callout(
    f"<strong>How to read this:</strong> the hybrid sends this customer down the "
    f"<strong>{branch['name'].lower()}</strong> branch because they have "
    f"{'no purchase history' if not n_purchases else f'{n_purchases} purchases on record'}. "
    "Popularity shows the same list to everyone, so the overlap count under each "
    "column is a direct read on how much personalization is happening at all. "
    "Whether that changes what people buy is the "
    "<a href='Model_Comparison'>Model Comparison</a> page.",
    icon="info",
)

ui.footer()
