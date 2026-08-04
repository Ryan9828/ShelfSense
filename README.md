# ShelfSense — Retail Product Recommender

A hybrid recommender system for retail, built end-to-end: data pipeline,
collaborative filtering + content-based cold-start, an offline A/B test
against a popularity baseline, a FastAPI serving layer, and a Streamlit
storefront demo.

Built on the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
Kaggle dataset (real anonymized retail transactions).

**[docs/CASE_STUDY.md](docs/CASE_STUDY.md)** — the full retrospective: what
worked, what didn't, and how each failure was actually diagnosed. Worth
reading before an interview, since it's the more interesting document.

## Why this project

Recommenders are a different ML paradigm from regression/classification —
ranking metrics (Recall@K, NDCG@K), implicit feedback, and cold-start are all
problems that don't show up in a typical tabular-prediction project. This is
also the ML problem most directly relevant to retail/e-commerce employers.

## Architecture

```
Kaggle CSVs -> build_features.py -> processed parquet -> train.py -> artifacts/
                                                                          |
                                                          FastAPI (app/main.py)
                                                                          |
                                                     Streamlit demo (frontend/)
```

**Modeling approach** (`src/shelfsense/`):
- `baseline.py` — popularity model (control arm, and cold-start fallback)
- `collaborative.py` — implicit-feedback ALS (Hu, Koren & Volinsky). Benchmarked
  on every training run but **not used by the shipped hybrid** — see the finding below
- `affinity.py` — category-affinity popularity: best-sellers within a
  customer's own favorite product category. What the hybrid actually uses
  for warm customers, in place of latent-factor CF
- `content.py` — TF-IDF similarity over article metadata, used when a
  customer has too little history for category-affinity to be reliable
- `hybrid.py` — routes each customer to affinity / content / global popularity
  depending on how much purchase history they have (the actual cold-start strategy)
- `evaluate.py` — Recall@K, NDCG@K, and a paired bootstrap significance test
  used as an **offline A/B test**, scored against the same held-out week of
  real future purchases, with a 95% CI on the uplift (see `docs/ab_test_results.md`
  after running `train.py`)

**The actual finding** (real H&M data, see `docs/ab_test_results.md` for exact
numbers): ALS matrix factorization — implicit-feedback collaborative
filtering, the latent-factor kind — was implemented and
benchmarked first, and it *lost* decisively to the popularity baseline —
fashion repurchase rates are low and the catalog turns over fast, so
the user-item interaction matrix is too sparse over a single-week holdout for the
learned factors to beat "what's trending right now." Category-affinity popularity was built as
a replacement and statistically **ties** the popularity baseline (95% CI
includes zero) while still personalizing which items are shown per customer.
The ALS code stayed in the repo specifically so this comparison is
reproducible on every training run, rather than quietly deleting the
approach that didn't work.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Get the data (manual — needs your own Kaggle account)

```bash
./scripts/download_data.sh
```

See the comment header in that script for the one-time Kaggle API token setup.

## 2. Build features and train

```bash
python -m shelfsense.build_features   # subsamples + time-splits into data/processed/
python -m shelfsense.train            # fits all models, runs the offline A/B test,
                                       # writes artifacts/ and docs/ab_test_results.md
```

## 3. Run the API

```bash
uvicorn app.main:app --reload
curl http://localhost:8000/health
curl "http://localhost:8000/recommend/<customer_id>?model=hybrid&k=12"   # or model=popularity / item_cf
curl "http://localhost:8000/customers/profiles?n=5"                     # sampled customers + their history
curl "http://localhost:8000/articles/batch?ids=0808651003,0554757003"   # many articles in one call
```

`/customers/profiles` and `/articles/batch` exist for the frontend: the demo
introduces customers by their purchase behaviour rather than their 64-char
hash, and drawing three 12-item recommendation lists used to cost 36
sequential round trips.

## 4. Run the storefront demo

```bash
streamlit run frontend/streamlit_app.py
```

This is a 4-page app. Pages live in `frontend/views/` and are registered in
`streamlit_app.py` via `st.navigation`; `frontend/api.py` is the only module
that talks HTTP, `frontend/ui.py` holds the design system, and
`frontend/routing.py` mirrors the hybrid's branching rule for display.
Streamlit-native theme tokens are in `.streamlit/config.toml` — **that file
must be committed**, or a deployed app falls back to the visitor's system
theme and renders the light-surface components in dark chrome.
- **Home** — what the project is, what it's built on, the headline finding,
  how the hybrid's history-length routing works, what each page shows, and
  what the demo is not. Makes no API calls, so it loads instantly even while
  the free-tier API is waking up.
- **Storefront** — the same three models in two registers. The top half
  merchandises the selected model's 12 recommendations as an actual shop page
  (`frontend/storefront_ui.py`); switching the tab re-stocks the store from a
  different model, and items the popularity baseline would *not* have shown
  are tagged, so personalization is visible rather than asserted. The bottom
  half is the analyst view: the customer's purchase count, favourite category
  and recent items, then all three lists side by side folded to their top 6,
  each reporting how much it overlaps the baseline.

  The shop has no product photography — the Kaggle competition ships images
  but `scripts/download_data.sh` never downloads them, so each tile draws the
  garment's real `colour_group_name` as a colour field with a silhouette
  picked from its `product_group_name` / `product_type_name`. Prices are the
  dataset's own `price` rescaled by `config.PRICE_SCALE` (see
  `data.attach_retail_price`). Every value on a tile is real data.
- **Model Comparison** — the offline A/B test results as charts + a data
  table, pulled live from `/eval-results` (same numbers as
  `docs/ab_test_results.md`).
- **Try It Yourself** — search the real catalog, pick a few items, and get
  a live recommendation for a customer with *no* stored history. This
  triggers the cold-start routing interactively (`/recommend/custom`) instead
  of only being demonstrable by looking up an existing anonymized customer_id
  — the more convincing artifact for a resume link.

## Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — Docker for the API, Streamlit
Community Cloud for the frontend (same pattern as `Portfolio_Risk_Platform`).

## Tests

```bash
pytest
```

Unit tests cover ranking metrics, the bootstrap significance test, the
popularity/content/affinity models, and the hybrid routing logic (via fakes
— no dataset needed to run `pytest`).

## Interesting points

- **ALS lost to popularity, and that's in the repo, not hidden**: the
  most defensible thing about this project isn't a metric, it's that a
  negative result (ALS underperforming a trivial baseline) drove an actual
  architecture change instead of being tuned away or quietly dropped. Most
  candidates' recommender projects only show the version that "won."
- **Cold-start**: a pure-CF system has nothing to say about a customer with
  0-2 purchases — often a large share of daily traffic. The hybrid's
  history-length routing (category-affinity / content / global popularity)
  is the actual answer to "how do you handle new users," not an afterthought.
- **Why an offline A/B test, not just accuracy**: a single Recall@12 number
  invites the question "is that difference real or noise?" The bootstrap CI
  answers it, and is the same technique used to read a live experiment.
- **Why this isn't RMSE/accuracy**: recommendation is a ranking problem —
  Recall@K/NDCG@K reward getting a few relevant items into a fixed-size list,
  which is what a product page actually shows.
