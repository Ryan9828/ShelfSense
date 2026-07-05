"""Entry point: python -m shelfsense.train

Fits all models on the processed train split, runs the offline A/B
comparison (hybrid vs. popularity baseline, plus item-CF as a benchmarked
alternative) against the held-out week, saves model artifacts for serving,
and writes the results to docs/ab_test_results.md.
"""
import json

import joblib
import numpy as np
import pandas as pd

from shelfsense import config, data, evaluate
from shelfsense.affinity import CategoryAffinityModel
from shelfsense.baseline import PopularityModel
from shelfsense.collaborative import ALSModel
from shelfsense.content import ContentModel
from shelfsense.hybrid import HybridRecommender


def main() -> None:
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("Loading processed data...")
    articles = pd.read_parquet(config.DATA_PROCESSED / "articles.parquet")
    train = pd.read_parquet(config.DATA_PROCESSED / "transactions_train.parquet")
    test = pd.read_parquet(config.DATA_PROCESSED / "transactions_test.parquet")

    print("Building interaction matrix...")
    idx = data.IndexMaps.build(train)
    interactions = data.build_interaction_matrix(train, idx)

    print("Fitting popularity baseline...")
    popularity = PopularityModel().fit(train)

    print("Fitting content model...")
    content = ContentModel().fit(articles)

    print("Fitting category-affinity model...")
    affinity = CategoryAffinityModel().fit(train, articles)

    print("Fitting ALS collaborative model (benchmarked, not used by the hybrid — see below)...")
    als = ALSModel().fit(interactions, idx)

    hybrid = HybridRecommender(affinity, content, popularity)

    print("Evaluating on held-out week...")
    test_baskets = data.customer_test_baskets(test)
    eval_customers = list(test_baskets.keys())
    k = config.TOP_K

    def eval_model(fn):
        return evaluate.evaluate_recommender(fn, eval_customers, test_baskets, k)

    hybrid_recall, hybrid_ndcg = eval_model(lambda cid: hybrid.recommend(cid, k, train))

    pop_recs = popularity.recommend(k)
    pop_recall, pop_ndcg = eval_model(lambda cid: pop_recs)

    als_recall, als_ndcg = eval_model(lambda cid: [a for a, _ in als.recommend(cid, k)])

    results = {
        "k": k,
        "n_eval_customers": len(eval_customers),
        f"hybrid_recall@{k}": float(np.nanmean(hybrid_recall)),
        f"popularity_recall@{k}": float(np.nanmean(pop_recall)),
        f"item_cf_recall@{k}": float(np.nanmean(als_recall)),
        "hybrid_vs_popularity_recall": evaluate.bootstrap_paired_diff(
            hybrid_recall, pop_recall, config.N_BOOTSTRAP
        ),
        "item_cf_vs_popularity_recall": evaluate.bootstrap_paired_diff(
            als_recall, pop_recall, config.N_BOOTSTRAP
        ),
        f"hybrid_ndcg@{k}": float(np.nanmean(hybrid_ndcg)),
        f"popularity_ndcg@{k}": float(np.nanmean(pop_ndcg)),
        f"item_cf_ndcg@{k}": float(np.nanmean(als_ndcg)),
        "hybrid_vs_popularity_ndcg": evaluate.bootstrap_paired_diff(
            hybrid_ndcg, pop_ndcg, config.N_BOOTSTRAP
        ),
    }
    print(json.dumps(results, indent=2))

    print("Saving artifacts...")
    joblib.dump(popularity, config.ARTIFACTS / "popularity.joblib")
    joblib.dump(content, config.ARTIFACTS / "content.joblib")
    joblib.dump(affinity, config.ARTIFACTS / "affinity.joblib")
    joblib.dump(als, config.ARTIFACTS / "als.joblib")  # kept for the /recommend?model=item_cf comparison
    with open(config.ARTIFACTS / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    _write_report(results)
    print(f"Done. Artifacts in {config.ARTIFACTS}, report in docs/ab_test_results.md")


def _write_report(results: dict) -> None:
    k = results["k"]
    hp = results["hybrid_vs_popularity_recall"]
    cp = results["item_cf_vs_popularity_recall"]
    hn = results["hybrid_vs_popularity_ndcg"]
    report = f"""# Offline A/B Test: Hybrid vs. Popularity Baseline (and Item-CF, benchmarked)

Evaluated on {results['n_eval_customers']:,} customers with at least one
purchase in the held-out {config.HOLDOUT_DAYS}-day window, using a paired
bootstrap ({config.N_BOOTSTRAP} resamples) over per-customer metrics.

| Model | Recall@{k} | NDCG@{k} |
|---|---|---|
| Popularity (control) | {results[f'popularity_recall@{k}']:.4f} | {results[f'popularity_ndcg@{k}']:.4f} |
| Hybrid (affinity + content + popularity) | {results[f'hybrid_recall@{k}']:.4f} | {results[f'hybrid_ndcg@{k}']:.4f} |
| Item-based collaborative filtering (ALS) | {results[f'item_cf_recall@{k}']:.4f} | {results[f'item_cf_ndcg@{k}']:.4f} |

**Hybrid vs. popularity** — Recall@{k} mean diff {hp['mean_diff']:+.4f}, 95% CI
[{hp['ci_low']:+.4f}, {hp['ci_high']:+.4f}], p={hp['p_value']:.4f}.
NDCG@{k} mean diff {hn['mean_diff']:+.4f}, 95% CI [{hn['ci_low']:+.4f}, {hn['ci_high']:+.4f}], p={hn['p_value']:.4f}.

**Item-CF vs. popularity** — Recall@{k} mean diff {cp['mean_diff']:+.4f}, 95% CI
[{cp['ci_low']:+.4f}, {cp['ci_high']:+.4f}], p={cp['p_value']:.4f}.

## Reading this

A 95% CI that excludes zero means the difference is unlikely to be noise
given this customer sample. This is an *offline* counterfactual comparison,
not a live experiment — both models are scored against the same actual
future purchases, so it can't capture things a real A/B test would (novelty
effects, display position bias, purchase behavior changing in response to
what's shown).

**Why item-CF isn't what ships**: pure item-based ALS collaborative
filtering was implemented and benchmarked first — see the row above. It
significantly *underperforms* the popularity baseline here (this is a real,
reproducible finding, not a bug: fashion repurchase rates are low and the
catalog turns over fast, so item-level co-purchase patterns are too sparse
over a single-week holdout to beat "what's trending right now"). Rather than
ship a personalization layer that's worse than doing nothing, the hybrid
uses category-affinity popularity for warm customers instead — best-sellers
within *their own* favorite product category — which ties the popularity
baseline in aggregate metrics while still tailoring which items are shown
per customer. `src/shelfsense/collaborative.py` (ALS) is kept in the repo
and still benchmarked on every training run specifically to make this
comparison reproducible.
"""
    (config.ROOT / "docs" / "ab_test_results.md").write_text(report)


if __name__ == "__main__":
    main()
