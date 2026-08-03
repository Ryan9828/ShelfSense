"""Landing page: what ShelfSense is, what it's built on, what it found, and
what each page of the demo shows.

Makes no API calls on purpose. The demo API sleeps on its free tier, and the
first thing a visitor sees should never be a spinner — by the time they click
through to a data page, the wake-up has usually had a head start.
"""
import streamlit as st

import ui
from routing import BRANCHES, MIN_INTERACTIONS_FOR_CF

ui.hero(
    "A recommender that shows its work",
    "ShelfSense is a retail product recommender built end-to-end on 1.4 million "
    "real anonymized H&M transactions — data pipeline, three competing models, "
    "an offline A/B test with a bootstrap significance test, a FastAPI serving "
    "layer, and this demo. It ships the model that survived the test, and keeps "
    "the one that didn't so you can check the comparison yourself.",
)

ui.pills(
    [
        "1.39M transactions · 59,718 customers · 75,159 articles",
        "3 models benchmarked",
        "Paired bootstrap, 2,000 resamples",
        "FastAPI + Streamlit",
    ]
)

# ------------------------------------------------------------ the headline
ui.section("The finding", "flask")
st.markdown(
    "Item-based collaborative filtering — the textbook answer to "
    "“build me a recommender” — was built first, and **lost decisively to a "
    "trivial popularity baseline**. That is not a bug. Fashion repurchase "
    "rates are low and the catalog turns over fast, so item-level co-purchase "
    "signal is too sparse over a one-week holdout to beat *what is trending "
    "right now*."
)
ui.stat_cards(
    [
        {
            "label": "Popularity (control)",
            "value": "0.0185",
            "sub": "Recall@12 — the baseline to beat",
        },
        {
            "label": "Hybrid (shipped)",
            "value": "0.0155",
            "sub": "statistical tie, p=0.12 — personalizes at no measured cost",
            "accent": True,
        },
        {
            "label": "Item-CF (not shipped)",
            "value": "0.0091",
            "sub": "significantly worse, p<0.001 — the negative result",
        },
    ]
)
ui.callout(
    "So the hybrid ships instead: <strong>category-affinity for customers with "
    "history, content-based similarity for customers without</strong>. It ties "
    "the baseline on aggregate accuracy while still personalizing what each "
    "customer sees — a real trade the metrics alone don't capture. The losing "
    "model stayed in the repo on purpose and is re-benchmarked on every "
    "training run, so the comparison is reproducible rather than quietly deleted.",
    icon="flask",
)

# ------------------------------------------------------------ what's on each page
ui.section("What each page shows", "route")

col_store, col_compare, col_try = st.columns(3, gap="medium")
with col_store:
    ui.feature_card(
        "logo",
        "Compare the recommenders",
        "Pick a real anonymized customer — introduced by what they actually "
        "buy, not by their id — and see all three models' picks side by side, "
        "with how much each one overlaps the baseline.",
    )
    st.page_link("views/storefront.py", label="Open the storefront →")
with col_compare:
    ui.feature_card(
        "chart",
        "Model Comparison",
        "The offline A/B test itself: Recall@12 and NDCG@12 per model, plus "
        "bootstrap confidence intervals on every difference, so “ties” and "
        "“loses” are statistical claims rather than eyeballed gaps.",
    )
    st.page_link("views/model_comparison.py", label="See the results →")
with col_try:
    ui.feature_card(
        "pointer",
        "Try It Yourself",
        "Arrive as a brand-new customer with zero history: search the real "
        "catalog, add a few items, and watch the cold-start routing change "
        "strategy live as your basket grows.",
    )
    st.page_link("views/try_it_yourself.py", label="Start picking →")

# ------------------------------------------------------------ how it works
ui.section("How the hybrid decides", "target")
st.caption(
    "Every customer is routed by how much purchase history they have. This is "
    "the actual answer to “how do you handle new users”, not an afterthought — "
    f"the threshold is {MIN_INTERACTIONS_FOR_CF} purchases, set in config and "
    "used identically at training and serving time."
)
for i, b in enumerate(BRANCHES, start=1):
    ui.step(i, f"<strong>{b['when']} ({b['why'].split(' — ')[0]})</strong> — {b['how']}")

# ------------------------------------------------------------ what it's built on
ui.section("What it's built on", "database")
c1, c2, c3, c4 = st.columns(4, gap="medium")
with c1:
    st.markdown(
        "**1 · Data**\n\n"
        "H&M's public Kaggle transaction dataset, subsampled to 60,000 "
        "customers (1.39M purchases, 75,159 articles) — enough to keep the "
        "sparsity and cold-start behaviour realistic, small enough to iterate "
        "on a laptop."
    )
with c2:
    st.markdown(
        "**2 · Split**\n\n"
        "A time-based split mirroring the competition protocol: the final 7 "
        "days are held out as “future” purchases. Nothing is scored against "
        "data it was trained on."
    )
with c3:
    st.markdown(
        "**3 · Three models**\n\n"
        "A popularity baseline, item-based ALS collaborative filtering, and "
        "the hybrid. All three are trained in one run and scored on the same "
        "3,066 customers who bought something in the held-out week."
    )
with c4:
    st.markdown(
        "**4 · Significance**\n\n"
        "2,000-resample paired bootstrap over per-customer Recall@12 and "
        "NDCG@12, giving a 95% CI and a p-value on every model-vs-baseline "
        "difference."
    )

# ------------------------------------------------------------ honest limits
ui.section("What this demo is not", "alert")
st.markdown(
    "- **It is not a live A/B test.** Every model is scored counterfactually "
    "against purchases that already happened, so it cannot capture novelty "
    "effects, display position bias, or customers changing behaviour in "
    "response to what they're shown. Those need real traffic.\n"
    "- **The absolute numbers are small by design.** Recall@12 of ~0.018 means "
    "12 recommendations out of a 75,000-item catalog catch about 1.8% of what "
    "a customer buys next week. What matters here is the *comparison* between "
    "models on identical data, not the level.\n"
    "- **There are no product images.** The dataset ships metadata only, so "
    "items are drawn as a colour swatch plus name, type and colour.\n"
    "- **The API sleeps.** The demo backend runs on a free tier and idles down; "
    "the first data page after a quiet spell can take up to a minute to wake. "
    "Item-CF is also disabled on the deployed API — its model artifact alone "
    "would not fit the 512MB memory budget — so that column may show a notice "
    "instead of results."
)

ui.footer()
