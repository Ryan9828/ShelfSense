# Offline A/B Test: Hybrid vs. Popularity Baseline (and ALS, benchmarked)

Evaluated on 3,066 customers with at least one
purchase in the held-out 7-day window, using a paired
bootstrap (2000 resamples) over per-customer metrics.

| Model | Recall@12 | NDCG@12 |
|---|---|---|
| Popularity (control) | 0.0185 | 0.0107 |
| Hybrid (affinity + content + popularity) | 0.0155 | 0.0101 |
| ALS matrix factorization (latent-factor CF) | 0.0091 | 0.0055 |

**Hybrid vs. popularity** — Recall@12 mean diff -0.0030, 95% CI
[-0.0066, +0.0007], p=0.1200.
NDCG@12 mean diff -0.0006, 95% CI [-0.0030, +0.0017], p=0.5970.

**ALS vs. popularity** — Recall@12 mean diff -0.0094, 95% CI
[-0.0139, -0.0051], p=0.0000.

## Reading this

A 95% CI that excludes zero means the difference is unlikely to be noise
given this customer sample. This is an *offline* counterfactual comparison,
not a live experiment — both models are scored against the same actual
future purchases, so it can't capture things a real A/B test would (novelty
effects, display position bias, purchase behavior changing in response to
what's shown).

**Why ALS isn't what ships**: implicit-feedback collaborative filtering via
ALS matrix factorization (Hu, Koren & Volinsky) — a latent-factor model, not
a neighbourhood one — was implemented and benchmarked first; see the row
above. It significantly *underperforms* the popularity baseline here (this is
a real, reproducible finding, not a bug: fashion repurchase rates are low and
the catalog turns over fast, so the user-item interaction matrix is too
sparse over a single-week holdout for the learned factors to beat "what's
trending right now"). Rather than
ship a personalization layer that's worse than doing nothing, the hybrid
uses category-affinity popularity for warm customers instead — best-sellers
within *their own* favorite product category — which ties the popularity
baseline in aggregate metrics while still tailoring which items are shown
per customer. `src/shelfsense/collaborative.py` (ALS) is kept in the repo
and still benchmarked on every training run specifically to make this
comparison reproducible.
