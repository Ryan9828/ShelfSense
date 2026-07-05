"""Offline A/B test results, straight from the same held-out week used to
train the models — no live traffic needed, since every model is scored
against real purchases that already happened. See docs/ab_test_results.md
for the full write-up this page visualizes.
"""
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Fixed categorical mapping, consistent across both charts on this page —
# color always follows the model, never its rank. Short names keep the chart's
# x-axis/legend from truncating; the fuller "control/shipped/benchmarked"
# framing lives in the surrounding prose instead.
MODEL_COLORS = {
    "Popularity": "#2a78d6",
    "Hybrid": "#1baf7a",
    "Item-CF": "#eda100",
}
MODEL_ORDER = list(MODEL_COLORS.keys())

st.set_page_config(page_title="ShelfSense — Model Comparison", page_icon="📊", layout="wide")
st.title("📊 Offline A/B test results")
st.caption(
    "All three models scored against the same real held-out week of purchases. "
    "Popularity = control baseline. Hybrid = what's shipped. Item-CF = benchmarked "
    "but not shipped (see verdict below). This is a counterfactual comparison, not "
    "a live experiment — see the note at the bottom."
)


@st.cache_data(ttl=60)
def get_eval_results() -> dict:
    resp = requests.get(f"{API_URL}/eval-results", timeout=10)
    resp.raise_for_status()
    return resp.json()


try:
    results = get_eval_results()
except requests.RequestException as e:
    st.error(f"Could not reach the API at {API_URL}: {e}")
    st.stop()

k = results["k"]
metrics_df = pd.DataFrame(
    [
        {"Model": "Popularity", "Recall@k": results[f"popularity_recall@{k}"], "NDCG@k": results[f"popularity_ndcg@{k}"]},
        {"Model": "Hybrid", "Recall@k": results[f"hybrid_recall@{k}"], "NDCG@k": results[f"hybrid_ndcg@{k}"]},
        {"Model": "Item-CF", "Recall@k": results[f"item_cf_recall@{k}"], "NDCG@k": results[f"item_cf_ndcg@{k}"]},
    ]
)

st.metric("Customers evaluated (held-out week)", f"{results['n_eval_customers']:,}")

col_recall, col_ndcg = st.columns(2)
for col, metric in [(col_recall, "Recall@k"), (col_ndcg, "NDCG@k")]:
    with col:
        chart = (
            alt.Chart(metrics_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Model:N", sort=MODEL_ORDER, title=None),
                y=alt.Y(f"{metric}:Q", title=f"{metric.replace('k', str(k))}"),
                color=alt.Color(
                    "Model:N",
                    scale=alt.Scale(domain=MODEL_ORDER, range=[MODEL_COLORS[m] for m in MODEL_ORDER]),
                    legend=alt.Legend(orient="bottom", title=None),
                ),
                tooltip=["Model", alt.Tooltip(f"{metric}:Q", format=".4f")],
            )
            .properties(title=f"{metric.replace('k', str(k))} by model", height=320)
        )
        st.altair_chart(chart, use_container_width=True)

st.subheader("Raw numbers")
st.dataframe(metrics_df.set_index("Model"), use_container_width=True)

st.subheader("Statistical significance (paired bootstrap vs. popularity)")
hp = results["hybrid_vs_popularity_recall"]
cp = results["item_cf_vs_popularity_recall"]

verdict_hybrid = "ties" if hp["ci_low"] <= 0 <= hp["ci_high"] else ("beats" if hp["mean_diff"] > 0 else "loses to")
verdict_cf = "ties" if cp["ci_low"] <= 0 <= cp["ci_high"] else ("beats" if cp["mean_diff"] > 0 else "loses to")

st.markdown(
    f"- **Hybrid {verdict_hybrid} popularity** on Recall@{k}: mean diff "
    f"{hp['mean_diff']:+.4f}, 95% CI [{hp['ci_low']:+.4f}, {hp['ci_high']:+.4f}], p={hp['p_value']:.3f}"
)
st.markdown(
    f"- **Item-CF {verdict_cf} popularity** on Recall@{k}: mean diff "
    f"{cp['mean_diff']:+.4f}, 95% CI [{cp['ci_low']:+.4f}, {cp['ci_high']:+.4f}], p={cp['p_value']:.3f}"
)

st.info(
    "A 95% CI that excludes zero means the difference is unlikely to be noise "
    "given this customer sample. This is an offline counterfactual — both models "
    "are scored against the same actual future purchases — so it can't capture "
    "things a real live A/B test would (novelty effects, position bias, purchase "
    "behavior changing in response to what's shown)."
)
