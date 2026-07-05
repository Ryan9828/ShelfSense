# ShelfSense — Retail Product Recommender

A hybrid recommender system for retail, built end-to-end: data pipeline,
collaborative filtering + content-based cold-start, an offline A/B test
against a popularity baseline, a FastAPI serving layer, and a Streamlit
storefront demo.

Built on the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
Kaggle dataset (real anonymized retail transactions).

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
- `collaborative.py` — implicit-feedback ALS (Hu, Koren & Volinsky) for
  customers with enough purchase history
- `content.py` — TF-IDF similarity over article metadata, used when a
  customer has too little history for CF to be reliable
- `hybrid.py` — routes each customer to CF / content / popularity depending
  on how much interaction history they have (the actual cold-start strategy)
- `evaluate.py` — Recall@K, NDCG@K, and a paired bootstrap significance test
  used as an **offline A/B test**: hybrid vs. popularity baseline, scored
  against the same held-out week of real future purchases, with a 95% CI on
  the uplift (see `docs/ab_test_results.md` after running `train.py`)

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
python -m shelfsense.train            # fits all 3 models, runs the offline A/B test,
                                       # writes artifacts/ and docs/ab_test_results.md
```

## 3. Run the API

```bash
uvicorn app.main:app --reload
curl http://localhost:8000/health
curl "http://localhost:8000/recommend/<customer_id>?model=hybrid&k=12"
```

## 4. Run the storefront demo

```bash
streamlit run frontend/streamlit_app.py
```

Shows hybrid vs. popularity recommendations side by side for a sampled
customer — this is the artifact worth linking from a resume.

## Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — Docker for the API, Streamlit
Community Cloud for the frontend (same pattern as `Portfolio_Risk_Platform`).

## Tests

```bash
pytest
```

Unit tests cover ranking metrics, the bootstrap significance test, the
popularity/content models, and the hybrid routing logic (via fakes — no
dataset needed to run `pytest`).

## Talking points for interviews

- **Cold-start**: a pure-CF system has nothing to say about a customer with
  0-2 purchases — often a large share of daily traffic. The hybrid's
  history-length routing is the actual answer to "how do you handle new
  users," not an afterthought.
- **Why an offline A/B test, not just accuracy**: a single Recall@12 number
  invites the question "is that difference real or noise?" The bootstrap CI
  answers it, and is the same technique used to read a live experiment.
- **Why this isn't RMSE/accuracy**: recommendation is a ranking problem —
  Recall@K/NDCG@K reward getting a few relevant items into a fixed-size list,
  which is what a product page actually shows.
