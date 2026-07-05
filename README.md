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
  for warm customers, in place of item-based CF
- `content.py` — TF-IDF similarity over article metadata, used when a
  customer has too little history for category-affinity to be reliable
- `hybrid.py` — routes each customer to affinity / content / global popularity
  depending on how much purchase history they have (the actual cold-start strategy)
- `evaluate.py` — Recall@K, NDCG@K, and a paired bootstrap significance test
  used as an **offline A/B test**, scored against the same held-out week of
  real future purchases, with a 95% CI on the uplift (see `docs/ab_test_results.md`
  after running `train.py`)

**The actual finding** (real H&M data, see `docs/ab_test_results.md` for exact
numbers): item-based ALS collaborative filtering was implemented and
benchmarked first, and it *lost* decisively to the popularity baseline —
fashion repurchase rates are low and the catalog turns over fast, so
item-level co-purchase patterns are too sparse over a single-week holdout to
beat "what's trending right now." Category-affinity popularity was built as
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
```

## 4. Run the storefront demo

```bash
streamlit run frontend/streamlit_app.py
```

This is a 3-page app (Streamlit auto-discovers `frontend/pages/`):
- **Home** — the customer-lookup comparison: hybrid vs. popularity vs.
  item-CF side by side for a sampled existing customer.
- **📊 Model Comparison** — the offline A/B test results as charts + a data
  table, pulled live from `/eval-results` (same numbers as
  `docs/ab_test_results.md`).
- **🧑 Try It Yourself** — search the real catalog, pick a few items, and get
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

## Talking points for interviews

- **Item-CF lost to popularity, and that's in the repo, not hidden**: the
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
