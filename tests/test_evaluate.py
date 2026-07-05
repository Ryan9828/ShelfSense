import numpy as np

from shelfsense.evaluate import bootstrap_paired_diff, ndcg_at_k, recall_at_k


def test_recall_at_k_all_hits():
    assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_recall_at_k_partial_hit():
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5


def test_recall_at_k_ignores_items_beyond_k():
    # "b" only appears past the cutoff, so it shouldn't count as a hit
    assert recall_at_k(["a", "x", "y", "b"], {"a", "b"}, k=3) == 0.5


def test_recall_at_k_empty_actual_is_nan():
    assert np.isnan(recall_at_k(["a"], set(), k=3))


def test_ndcg_at_k_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_ndcg_at_k_rewards_earlier_hits():
    late = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)
    early = ndcg_at_k(["a", "x", "y"], {"a"}, k=3)
    assert early > late


def test_bootstrap_paired_diff_identical_arrays_is_zero():
    a = np.array([0.5, 0.6, 0.7, 0.4, 0.9])
    result = bootstrap_paired_diff(a, a.copy(), n_boot=200)
    assert result["mean_diff"] == 0.0
    assert result["ci_low"] <= 0.0 <= result["ci_high"]


def test_bootstrap_paired_diff_detects_uplift():
    rng = np.random.default_rng(0)
    a = rng.normal(0.6, 0.05, size=500)  # clearly better
    b = rng.normal(0.4, 0.05, size=500)
    result = bootstrap_paired_diff(a, b, n_boot=500)
    assert result["mean_diff"] > 0
    assert result["ci_low"] > 0  # CI excludes zero -> "significant" uplift
