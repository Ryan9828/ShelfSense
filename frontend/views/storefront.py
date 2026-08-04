"""Storefront: what each recommender would actually put in front of a shopper,
then the analyst view of the same three lists.

The page is deliberately in two registers. A ranked list of product names is
an artefact of the evaluation harness, not a product — so the top half renders
the selected model's output as an actual shop page (see storefront_ui.py), and
switching the tab re-merchandises the store with a different model's picks.
The bottom half is the comparison the offline A/B test needs: all three lists
side by side, folded to their top 6, with how much each overlaps the baseline.

The previous, analyst-only version of this page is kept at
frontend/views/_storefront_classic.py.bak, and in git history at 3c0bded.
"""
import streamlit as st

import api
import storefront_ui as shop
import ui
from routing import routing_branch

MODELS = [
    ("hybrid", "Hybrid", "Shipped", "shipped"),
    ("popularity", "Popularity", "Control", "control"),
    ("item_cf", "Item-CF", "Not shipped", "bench"),
]
FOLD = 6  # cards per column in the comparison before the "show all" toggle
K = 12  # always fetch 12 — the rank the offline metrics are measured at

shop.store_css()

ui.hero(
    "Compare the recommenders",
    "Three models, the same customer, the same catalog. The shop below is "
    "merchandised live from whichever model you select — switch the tab and "
    "the storefront re-stocks. Underneath is the same output as the offline "
    "A/B test sees it.",
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
    return f"{n} purchases · mostly {p.get('top_category') or 'mixed'}"


idx = st.selectbox(
    "Customer",
    range(len(profiles)),
    format_func=lambda i: f"Customer {i + 1:02d} — {customer_label(profiles[i])}",
    help=(
        "A sample of real anonymized H&M customers from the training data, "
        "labelled by their own purchase history rather than by their id."
    ),
)

profile = profiles[idx]
customer_id = profile["customer_id"]
n_purchases = profile.get("n_purchases")
branch = routing_branch(n_purchases)
shopper = f"Customer {idx + 1:02d}"

# Fetch all three up front: the overlap flag on the shop tiles and the
# footnote in the comparison both need popularity's list to compare against,
# and one spinner beats three.
with st.spinner("Merchandising all three models…"):
    results: dict[str, list[dict] | str] = {}
    for model, *_ in MODELS:
        try:
            results[model] = api.recommend(customer_id, model, k=K)
        except api.ApiError as e:
            results[model] = str(e)

pop_items = results["popularity"]
pop_ids = {i["article_id"] for i in pop_items} if isinstance(pop_items, list) else set()

# ---------------------------------------------------------------- the shop
ui.callout(
    f"<strong>What you're looking at:</strong> a mock shop front for "
    f"<strong>{shopper}</strong>, stocked with the "
    f"<strong>{K} items</strong> the selected model picked for them out of a "
    "75,159-item catalog. Each tab is a different model making that choice — "
    "<strong>switch tabs and the shop re-stocks</strong>. Popularity shows the "
    "same {K} items to every customer in the store, so anything tagged "
    f"<em>picked for you</em> in the other tabs is an item personalization put "
    "there and the baseline would not have. Twelve is the shelf size H&amp;M's "
    "own competition scores at (MAP@12).".replace("{K}", str(K)),
    icon="pointer",
)

tabs = st.tabs([f"{label} · {badge_text}" for _, label, badge_text, _ in MODELS])
for tab, (model, label, _, _) in zip(tabs, MODELS):
    with tab:
        items = results[model]
        if isinstance(items, str):
            ui.callout(items, kind="amber", icon="alert")
            continue
        if not items:
            ui.callout("No recommendations for this customer.", icon="alert")
            continue

        if model == "popularity":
            subline = "The same window display for every shopper — no personalization."
            flags: set[str] = set()
        else:
            added = len([i for i in items if i["article_id"] not in pop_ids])
            subline = (
                f"{added} of {len(items)} are items the popularity baseline would "
                "not have shown — those are tagged."
            )
            flags = {i["article_id"] for i in items if i["article_id"] not in pop_ids}

        shop.store(
            items,
            swatch_of=ui.swatch_hex,
            eyebrow=f"{label} · recommended for {shopper}",
            headline="New in for you",
            subline=subline,
            flag_ids=flags,
        )

ui.callout(
    "<strong>Why there are no product photos.</strong> The H&amp;M Kaggle "
    "competition does ship product images — about 25GB of them — but this "
    "project never downloads that archive: none of the three models look at "
    "pixels, so it would be 25GB of build weight for nothing "
    "(<code>scripts/download_data.sh</code> pulls the three CSVs only). Rather "
    "than fill the gap with stock photos of clothes that aren't these clothes, "
    "each tile is drawn from the article's own record — the block colour is its "
    "real <code>colour_group_name</code>, and the silhouette is picked from its "
    "<code>product_type_name</code>, so a dress draws as a dress. "
    "<strong>Prices are real too:</strong> the dataset's <code>price</code> is "
    "normalized, and rescaling it recovers the shelf prices behind it — 77.9% "
    "of all 1.39M rows land exactly on a .99. No currency symbol, because the "
    "competition never says which currency these are, and that would be the "
    "only invented thing on the tile.",
    kind="amber",
    icon="info",
)

# ------------------------------------------------------- why they see this
ui.section(f"Why {shopper} sees this", "user")
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
    with st.expander(f"What {shopper} has actually bought — most recent first"):
        st.caption(
            "The point of comparison: a good recommendation should look like a "
            "plausible next purchase for this person."
        )
        hist_cols = st.columns(2, gap="medium")
        for i, item in enumerate(profile["recent"]):
            with hist_cols[i % 2]:
                st.markdown(ui.product_card(item, compact=True), unsafe_allow_html=True)

# ------------------------------------------------------------- side by side
ui.section("All three, side by side", "chart")
show_all = st.toggle(
    f"Show all {K} per model",
    value=False,
    help=f"Recall@{K} and NDCG@{K} are measured over all {K}; the top {FOLD} is just the fold.",
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

        ui.product_list(items if show_all else items[:FOLD], compact=True)
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
    "Popularity shows the same window display to everyone, so the overlap count is a "
    "direct read on how much personalization is happening at all. Whether that changes "
    "what people buy is the <a href='Model_Comparison'>Model Comparison</a> page.",
    icon="info",
)

ui.footer()
