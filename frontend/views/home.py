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
    "Which 12 products should a clothing shop put on its front page for you? "
    "ShelfSense answers that from 1.4 million real H&M purchases — and because "
    "three different algorithms answer it differently, it runs a controlled "
    "test to see which one is actually right.",
)

ui.pills(
    [
        "1.39M transactions · 59,718 customers · 75,159 articles",
        "3 models benchmarked",
        "Paired bootstrap, 2,000 resamples",
        "FastAPI + Streamlit",
    ]
)

st.caption(
    "New here? The **Storefront** page is the one to see first — it shows those "
    "12 picks as an actual shop, and lets you swap the algorithm behind them."
)

# ------------------------------------------------------------ the headline
ui.section("The finding", "flask")
st.markdown(
    "Item-based collaborative filtering — the textbook answer to "
    "“build me a recommender” — was built first, and **lost decisively to a "
    "trivial popularity baseline**."
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
with st.expander("Why the textbook model lost, and what ships instead"):
    st.markdown(
        "Fashion repurchase rates are low and the catalog turns over fast, so "
        "item-level co-purchase signal is too sparse over a one-week holdout to "
        "beat *what is trending right now*. That is a real result, not a bug.\n\n"
        "So the hybrid ships instead: **category-affinity for customers with "
        "history, content-based similarity for customers without**. It ties the "
        "baseline on aggregate accuracy while still personalizing what each "
        "customer sees — a trade the metrics alone don't capture.\n\n"
        "The losing model stayed in the repo on purpose and is re-benchmarked on "
        "every training run, so the comparison is reproducible rather than "
        "quietly deleted."
    )

# ------------------------------------------------------------ what's on each page
ui.section("What each page shows", "route")

col_store, col_compare, col_try = st.columns(3, gap="medium")
with col_store:
    ui.feature_card(
        "logo",
        "Storefront",
        "The 12 picks, merchandised as a real shop — swap the algorithm behind "
        "them and watch it re-stock.",
    )
    st.page_link("views/storefront.py", label="Open the storefront →")
with col_compare:
    ui.feature_card(
        "chart",
        "Model Comparison",
        "The test that settled it — with confidence intervals, so “ties” and "
        "“loses” are claims rather than eyeballed gaps.",
    )
    st.page_link("views/model_comparison.py", label="See the results →")
with col_try:
    ui.feature_card(
        "pointer",
        "Try It Yourself",
        "Arrive with no history at all. Add a few items and watch the "
        "cold-start logic switch strategy live.",
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
for col, (num, title, body) in zip(
    (c1, c2, c3, c4),
    [
        ("1", "Data", "H&M's public Kaggle dataset, subsampled to 60,000 customers — "
                      "small enough to iterate on a laptop."),
        ("2", "Split", "The final 7 days held out as “future” purchases. Nothing is "
                       "scored against data it trained on."),
        ("3", "Models", "Popularity, item-based ALS, and the hybrid — trained in one "
                        "run, scored on the same 3,066 customers."),
        ("4", "Significance", "2,000-resample paired bootstrap, giving a 95% CI and a "
                              "p-value on every difference."),
    ],
):
    with col:
        st.markdown(f"**{num} · {title}**\n\n{body}")

# ------------------------------------------------------------ honest limits
# Behind an expander on purpose. These caveats matter to anyone reading the
# numbers closely, but leading a first-time visitor with four paragraphs of
# disclaimer buries the thing they came to look at.
with st.expander("What this demo is not"):
    st.markdown(
        "- **It is not a live A/B test.** Every model is scored counterfactually "
        "against purchases that already happened, so it cannot capture novelty "
        "effects, display position bias, or customers changing behaviour in "
        "response to what they're shown. Those need real traffic.\n"
        "- **The absolute numbers are small by design.** Recall@12 of ~0.018 means "
        "12 recommendations out of a 75,000-item catalog catch about 1.8% of what "
        "a customer buys next week. What matters is the *comparison* between "
        "models on identical data, not the level.\n"
        "- **There are no product photos.** The dataset's image archive is 25GB "
        "and none of the models use it, so tiles show each item's real colour and "
        "product type instead.\n"
        "- **The API sleeps.** The demo backend runs on a free tier and idles down; "
        "the first data page after a quiet spell can take up to a minute to wake. "
        "Item-CF is also disabled there — its artifact alone would not fit the "
        "512MB budget — so that tab may show a notice instead of products."
    )

ui.footer()
