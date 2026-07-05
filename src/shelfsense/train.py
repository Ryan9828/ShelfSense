"""Entry point: python -m shelfsense.train

Fits all three models on the processed train split, runs the offline A/B
comparison (hybrid vs. popularity baseline) against the held-out week, saves
model artifacts for serving, and writes the A/B result to docs/ab_test_results.md.
"""
import json

import joblib
import pandas as pd

from shelfsense import config, data, evaluate
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

    print("Fitting ALS collaborative model...")
    als = ALSModel().fit(interactions, idx)

    hybrid = HybridRecommender(als, content, popularity)

    print("Evaluating on held-out week...")
    test_baskets = data.customer_test_baskets(test)
    eval_customers = list(test_baskets.keys())

    hybrid_recall, hybrid_ndcg = evaluate.evaluate_recommender(
        lambda cid: hybrid.recommend(cid, config.TOP_K, train),
        eval_customers,
        test_baskets,
        config.TOP_K,
    )
    pop_recs = popularity.recommend(config.TOP_K)
    pop_recall, pop_ndcg = evaluate.evaluate_recommender(
        lambda cid: pop_recs, eval_customers, test_baskets, config.TOP_K
    )

    recall_ab = evaluate.bootstrap_paired_diff(hybrid_recall, pop_recall, config.N_BOOTSTRAP)
    ndcg_ab = evaluate.bootstrap_paired_diff(hybrid_ndcg, pop_ndcg, config.N_BOOTSTRAP)

    results = {
        "k": config.TOP_K,
        "n_eval_customers": len(eval_customers),
        f"hybrid_recall@{config.TOP_K}": float(hybrid_recall.mean()),
        f"popularity_recall@{config.TOP_K}": float(pop_recall.mean()),
        "recall_uplift_bootstrap": recall_ab,
        f"hybrid_ndcg@{config.TOP_K}": float(hybrid_ndcg.mean()),
        f"popularity_ndcg@{config.TOP_K}": float(pop_ndcg.mean()),
        "ndcg_uplift_bootstrap": ndcg_ab,
    }
    print(json.dumps(results, indent=2))

    print("Saving artifacts...")
    joblib.dump(popularity, config.ARTIFACTS / "popularity.joblib")
    joblib.dump(content, config.ARTIFACTS / "content.joblib")
    joblib.dump(als, config.ARTIFACTS / "als.joblib")
    with open(config.ARTIFACTS / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    _write_report(results)
    print(f"Done. Artifacts in {config.ARTIFACTS}, report in docs/ab_test_results.md")


def _write_report(results: dict) -> None:
    k = results["k"]
    r = results["recall_uplift_bootstrap"]
    n = results["ndcg_uplift_bootstrap"]
    report = f"""# Offline A/B Test: Hybrid vs. Popularity Baseline

Evaluated on {results['n_eval_customers']:,} customers with at least one
purchase in the held-out {config.HOLDOUT_DAYS}-day window, using a paired
bootstrap ({config.N_BOOTSTRAP} resamples) over per-customer metrics.

| Metric | Hybrid | Popularity (control) | Mean diff | 95% CI | p-value |
|---|---|---|---|---|---|
| Recall@{k} | {results[f'hybrid_recall@{k}']:.4f} | {results[f'popularity_recall@{k}']:.4f} | {r['mean_diff']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['p_value']:.4f} |
| NDCG@{k} | {results[f'hybrid_ndcg@{k}']:.4f} | {results[f'popularity_ndcg@{k}']:.4f} | {n['mean_diff']:+.4f} | [{n['ci_low']:+.4f}, {n['ci_high']:+.4f}] | {n['p_value']:.4f} |

**Reading this**: a 95% CI that excludes zero means the uplift (or
regression) is unlikely to be noise given this sample of customers. This is
an *offline* counterfactual comparison, not a live experiment — both models
are scored against the same actual future purchases, so it can't capture
things a real A/B test would (novelty effects, display position bias,
purchase behavior actually changing in response to what's shown).
"""
    (config.ROOT / "docs" / "ab_test_results.md").write_text(report)


if __name__ == "__main__":
    main()
