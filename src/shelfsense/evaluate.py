"""Ranking metrics and the offline A/B test: an honest substitute for a live
experiment when there's no production traffic to split. We recompute the same
per-customer metric for two models against the *same* held-out future
purchases, then bootstrap the distribution of the paired difference — this is
the standard way to get a confidence interval on an uplift claim ("hybrid
beats popularity by +X% Recall@12") without a live experiment.
"""
import numpy as np


def recall_at_k(recommended: list[str], actual: set[str], k: int) -> float:
    if not actual:
        return np.nan
    hits = len(set(recommended[:k]) & actual)
    return hits / len(actual)


def ndcg_at_k(recommended: list[str], actual: set[str], k: int) -> float:
    if not actual:
        return np.nan
    dcg = sum(
        1.0 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in actual
    )
    ideal_hits = min(len(actual), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else np.nan


def evaluate_recommender(recommend_fn, customer_ids, test_baskets: dict, k: int) -> np.ndarray:
    """recommend_fn(customer_id) -> list[str]. Returns one Recall@k and one
    NDCG@k value per customer (NaN where the customer has no test basket —
    dropped before aggregation, not treated as a zero)."""
    recalls, ndcgs = [], []
    for cid in customer_ids:
        actual = test_baskets.get(cid, set())
        if not actual:
            continue
        recs = recommend_fn(cid)
        recalls.append(recall_at_k(recs, actual, k))
        ndcgs.append(ndcg_at_k(recs, actual, k))
    return np.array(recalls), np.array(ndcgs)


def bootstrap_paired_diff(
    metric_a: np.ndarray, metric_b: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> dict:
    """Paired bootstrap over customers for (a - b), e.g. hybrid - popularity.
    Returns mean diff, 95% CI, and a two-sided p-value (fraction of bootstrap
    samples that cross zero)."""
    assert len(metric_a) == len(metric_b)
    rng = np.random.default_rng(seed)
    n = len(metric_a)
    diffs = metric_a - metric_b
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[sample_idx].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    p_value = 2 * min((boot_means < 0).mean(), (boot_means > 0).mean())
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(p_value),
        "n": n,
    }
