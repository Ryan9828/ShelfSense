"""ShelfSense storefront demo — entrypoint.

This is the artifact to link in a resume/portfolio — it makes the offline A/B
result tangible instead of a table of numbers, including the negative result
(item-CF loses to popularity) which is the more interesting finding.

Run: streamlit run frontend/streamlit_app.py
Requires the FastAPI service running (default http://localhost:8000, override
with the API_URL env var — set this to the deployed API's URL in production).

Pages live in frontend/views/ and are registered here with st.navigation
(instead of the auto-discovered pages/ folder) so the sidebar gets proper
labels + icons. url_path values keep the URLs the old emoji-named pages had.
Design system: .streamlit/config.toml (theme tokens) + frontend/ui.py (all
component CSS/HTML).
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ui

st.set_page_config(page_title="ShelfSense", page_icon="🛍️", layout="wide")
ui.apply_theme()

pg = st.navigation(
    [
        st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
        st.Page("views/storefront.py", title="Storefront", icon=":material/storefront:", url_path="Storefront"),
        st.Page("views/model_comparison.py", title="Model Comparison", icon=":material/monitoring:", url_path="Model_Comparison"),
        st.Page("views/try_it_yourself.py", title="Try It Yourself", icon=":material/touch_app:", url_path="Try_It_Yourself"),
    ]
)

with st.sidebar:
    st.markdown("**ShelfSense**")
    st.caption(
        "A hybrid retail recommender on 1.4M real H&M transactions, with the "
        "offline A/B test that decided which model shipped — including the one "
        "that lost."
    )
    st.markdown(f"[Source on GitHub]({ui.REPO_URL})")
    st.divider()
    st.caption(
        "The demo API runs on a free tier and sleeps when idle — the first data "
        "page after a quiet spell can take up to a minute to wake."
    )

pg.run()
