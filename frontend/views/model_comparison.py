"""Offline A/B test results, straight from the same held-out week used to
train the models — no live traffic needed, since every model is scored
against real purchases that already happened. See docs/ab_test_results.md
for the full write-up this page visualizes.
"""
import altair as alt
import pandas as pd
import streamlit as st

import api
import ui
from ui import COLORS

# Fixed categorical mapping, consistent across both charts on this page —
# color always follows the model, never its rank. CVD-validated as a set
# against the light surface (deutan/protan/tritan separation all pass).
# Identity is carried by the x-axis labels, so no legend is drawn.
MODEL_COLORS = {
    "Popularity": COLORS["teal"],
    "Hybrid": COLORS["accent"],
    "Item-CF": COLORS["amber"],
}
MODEL_ORDER = list(MODEL_COLORS.keys())

ui.hero(
    "Offline A/B test results",
    "All three models scored against the same real held-out week of purchases. "
    "Popularity = control baseline. Hybrid = what's shipped. Item-CF = benchmarked "
    "but not shipped (see verdict below). This is a counterfactual comparison, not "
    "a live experiment — see the note at the bottom.",
)


try:
    with st.spinner("Loading evaluation results…"):
        results = api.eval_results()
except api.ApiError as e:
    st.error(str(e))
    st.info(
        "The demo API sleeps when idle on its free tier. If this is the first "
        "load in a while, wait a few seconds and refresh — it is waking up."
    )
    st.stop()

k = results["k"]
metrics_df = pd.DataFrame(
    [
        {"Model": "Popularity", "Recall@k": results[f"popularity_recall@{k}"], "NDCG@k": results[f"popularity_ndcg@{k}"]},
        {"Model": "Hybrid", "Recall@k": results[f"hybrid_recall@{k}"], "NDCG@k": results[f"hybrid_ndcg@{k}"]},
        {"Model": "Item-CF", "Recall@k": results[f"item_cf_recall@{k}"], "NDCG@k": results[f"item_cf_ndcg@{k}"]},
    ]
)

ui.stat_cards(
    [
        {
            "label": "Customers evaluated",
            "value": f"{results['n_eval_customers']:,}",
            "sub": "everyone who bought something in the held-out week",
        },
        {
            "label": "Recommendations scored",
            "value": f"@{k}",
            "sub": f"top {k} per customer, matching H&M's own MAP@{k} metric",
        },
        {
            "label": "Bootstrap resamples",
            "value": "2,000",
            "sub": "paired, per-customer — gives every CI and p-value below",
        },
    ]
)


def metric_chart(metric: str) -> alt.LayerChart:
    title = metric.replace("k", str(k))
    bars = (
        alt.Chart(metrics_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=52)
        .encode(
            x=alt.X("Model:N", sort=MODEL_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{metric}:Q", title=title),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(domain=MODEL_ORDER, range=[MODEL_COLORS[m] for m in MODEL_ORDER]),
                legend=None,
            ),
            tooltip=["Model", alt.Tooltip(f"{metric}:Q", format=".4f")],
        )
    )
    # Direct labels (only 3 bars — label them all), in text ink, not series color.
    labels = bars.mark_text(dy=-10, fontWeight=700, fontSize=12.5).encode(
        text=alt.Text(f"{metric}:Q", format=".4f"),
        color=alt.value(COLORS["secondary"]),
    )
    return (
        (bars + labels)
        .properties(title=f"{title} by model", height=320)
        .configure_view(strokeWidth=0)
        .configure_title(
            font="Rubik", fontSize=15, fontWeight=600, color=COLORS["foreground"], anchor="start"
        )
        .configure_axis(
            labelFont="Nunito Sans",
            titleFont="Nunito Sans",
            labelColor=COLORS["muted_text"],
            titleColor=COLORS["muted_text"],
            labelFontSize=12,
            gridColor="#EFEFF1",
            domainColor=COLORS["border"],
            tickColor="transparent",
        )
    )


col_recall, col_ndcg = st.columns(2, gap="medium")
for col, metric in [(col_recall, "Recall@k"), (col_ndcg, "NDCG@k")]:
    with col:
        st.altair_chart(metric_chart(metric), use_container_width=True)

with st.expander("Raw numbers"):
    st.dataframe(metrics_df.set_index("Model").style.format("{:.4f}"), use_container_width=True)

ui.section("Is the difference real?", "flask")
st.caption(
    "Paired bootstrap against the popularity control — the same 2,000 resamples "
    "score both models, so the comparison survives the fact that some customers "
    "are simply easier to predict than others."
)

VERDICT_BADGE = {"ties": "control", "beats": "shipped", "loses to": "bench"}
for model, comp in [
    ("Hybrid", results["hybrid_vs_popularity_recall"]),
    ("Item-CF", results["item_cf_vs_popularity_recall"]),
]:
    verdict = (
        "ties"
        if comp["ci_low"] <= 0 <= comp["ci_high"]
        else ("beats" if comp["mean_diff"] > 0 else "loses to")
    )
    st.markdown(
        f'<div class="ss-verdict"><strong>{model}</strong>'
        f"{ui.badge(f'{verdict} popularity', VERDICT_BADGE[verdict])}"
        f'<span class="ss-ci">Recall@{k} mean diff {comp["mean_diff"]:+.4f} · '
        f'95% CI [{comp["ci_low"]:+.4f}, {comp["ci_high"]:+.4f}] · '
        f'p={comp["p_value"]:.3f}</span></div>',
        unsafe_allow_html=True,
    )

ui.callout(
    "A 95% CI that excludes zero means the difference is unlikely to be noise "
    "given this customer sample. This is an offline counterfactual — both models "
    "are scored against the same actual future purchases — so it can't capture "
    "things a real live A/B test would (novelty effects, position bias, purchase "
    "behavior changing in response to what's shown).",
    icon="flask",
)

ui.footer()
