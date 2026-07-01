from __future__ import annotations

import numpy as np

from scripts.geometry_diagnostics import (
    bootstrap_eum,
    compute_diagnostics,
    effective_rank,
    effective_rank_entropy,
    effective_rank_participation,
    prompt_metrics,
    reward_collinearity,
    unique_pareto_stats,
)
from vpo.utils.eval_metrics import dirichlet_weights, expected_max_weighted


def metrics(tensor: np.ndarray) -> dict[str, float]:
    return compute_diagnostics(tensor, n_weights=512, seed=7, bootstrap_samples=25)["metrics"]


def test_collinear_rewards_have_high_collinearity_and_low_effective_rank():
    base = np.linspace(0.0, 1.0, 12)
    rewards = np.stack([base, 2.0 * base, 3.0 * base], axis=1)

    out = metrics(rewards)

    assert out["reward_collinearity"] > 0.99
    assert out["effective_rank_entropy"] < 1.05
    assert out["effective_rank"] == out["effective_rank_entropy"]


def test_eum_gap_uses_delta_set_not_mean_reward_gap():
    collapsed = np.array([[0.5, 0.5], [0.5, 0.5]])
    specialists = np.array([[1.0, 0.0], [0.0, 1.0]])

    collapsed_metrics = compute_diagnostics(collapsed, n_weights=256, seed=3, bootstrap_samples=0)["metrics"]
    specialist_metrics = compute_diagnostics(specialists, n_weights=256, seed=3, bootstrap_samples=0)["metrics"]

    assert np.isclose(collapsed_metrics["eum_gap"], 0.0)
    assert specialist_metrics["eum_gap"] > 0.20
    assert specialist_metrics["eum_gap"] == specialist_metrics["expected_support_delta_set"]


def test_winner_entropy_is_tie_aware_over_reward_clusters():
    duplicated_winners = np.array([[1.0, 0.0], [1.0, 0.0]])

    out = compute_diagnostics(duplicated_winners, n_weights=32, seed=0, bootstrap_samples=0)["metrics"]

    assert out["reward_cluster_count"] == 1.0
    assert out["winner_cluster_entropy"] == 0.0
    assert out["winner_cluster_entropy_normalized"] == 0.0
    assert out["dominant_cluster_mass"] == 1.0


def test_unique_pareto_fraction_deduplicates_reward_vectors():
    rewards = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]
    )

    stats = unique_pareto_stats(rewards)
    out = compute_diagnostics(rewards, n_weights=32, seed=0, bootstrap_samples=0)["metrics"]

    assert stats["unique_reward_vector_count"] == 2.0
    assert stats["unique_pareto_count"] == 1.0
    assert stats["unique_pareto_fraction"] == 0.5
    assert out["pareto_fraction"] == out["unique_pareto_fraction"]
    assert out["raw_pareto_fraction"] != out["unique_pareto_fraction"]


def test_inactive_collinearity_is_undefined_not_forced_to_one():
    constant = np.ones((2, 4, 3))

    out = compute_diagnostics(constant, n_weights=16, seed=0, bootstrap_samples=0)

    assert out["per_prompt"][0]["reward_collinearity"] is None
    assert out["per_prompt"][1]["reward_collinearity"] is None
    assert out["metrics"]["reward_collinearity"] is None
    assert out["metrics"]["reward_collinearity_defined"] == 0.0


def test_effective_rank_entropy_and_participation_are_reported():
    rewards = np.eye(3)

    assert effective_rank_entropy(rewards) > 1.0
    assert effective_rank_participation(rewards) > 1.0
    assert effective_rank(rewards) == effective_rank_entropy(rewards)


def test_fixed_target_regret_uses_train_and_target_scalarizations():
    rewards = np.array([[1.0, 0.0], [0.0, 1.0]])

    out = compute_diagnostics(
        rewards,
        n_weights=64,
        seed=5,
        bootstrap_samples=0,
        train_weights=np.array([1.0, 0.0]),
        target_weights=np.array([0.0, 1.0]),
    )["metrics"]

    assert np.isclose(out["target_regret_fixed"], 1.0)
    assert out["target_regret"] == out["target_regret_fixed"]


def test_specialist_rewards_increase_eum_gap_and_winner_entropy():
    collapsed = np.tile(np.array([[0.33, 0.33, 0.33]]), (6, 1))
    specialists = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )

    collapsed_metrics = metrics(collapsed)
    specialist_metrics = metrics(specialists)

    assert specialist_metrics["eum_gap"] > collapsed_metrics["eum_gap"]
    assert specialist_metrics["winner_cluster_entropy"] > collapsed_metrics["winner_cluster_entropy"]
    assert specialist_metrics["base_best_of_k_slope"] == specialist_metrics["best_of_k_slope"]


def test_high_entropy_off_target_specialists_have_target_regret():
    rewards = np.array(
        [
            [0.2, 1.0, 0.0],
            [0.2, 0.0, 1.0],
            [0.2, 0.9, 0.1],
            [1.0, 0.0, 0.0],
        ]
    )

    out = compute_diagnostics(
        rewards,
        n_weights=512,
        seed=7,
        bootstrap_samples=5,
        target_weights=np.array([1.0, 0.0, 0.0]),
    )["metrics"]

    assert out["winner_entropy"] > 0.0
    assert out["target_regret"] > 0.0


def test_fixed_seed_bootstrap_is_deterministic():
    rng = np.random.default_rng(123)
    rewards = rng.uniform(size=(5, 8, 4))

    first = compute_diagnostics(rewards, n_weights=256, seed=11, bootstrap_samples=40)
    second = compute_diagnostics(rewards, n_weights=256, seed=11, bootstrap_samples=40)

    assert first["metrics"]["bootstrap_eum_mean"] == second["metrics"]["bootstrap_eum_mean"]
    assert first["metrics"]["bootstrap_eum_std"] == second["metrics"]["bootstrap_eum_std"]
    assert first["metrics"]["bootstrap_stability"] == second["metrics"]["bootstrap_stability"]


def test_off_target_specialists_have_positive_target_regret():
    rewards = np.array(
        [
            [0.9, 0.9, 0.0],
            [0.0, 0.9, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    out = compute_diagnostics(
        rewards,
        n_weights=512,
        seed=5,
        bootstrap_samples=10,
        target_weights=np.array([0.0, 0.0, 1.0]),
    )["metrics"]

    assert out["winner_entropy"] > 0.0
    assert out["target_regret"] > 0.0
    assert "base_best_of_k_slope" in out


def test_bootstrap_eum_matches_repeated_resampling_formula():
    rng = np.random.default_rng(321)
    rewards = rng.uniform(size=(7, 5, 3))
    weights = rng.dirichlet(np.ones(3), size=32)
    seed = 17
    samples = 19

    optimized = bootstrap_eum(rewards, weights, seed=seed, samples=samples)

    old_rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        idx = old_rng.integers(0, rewards.shape[0], size=rewards.shape[0])
        values.append(float(np.mean([expected_max_weighted(rewards[i], weights) for i in idx])))
    old_values = np.asarray(values, dtype=np.float64)

    assert optimized["bootstrap_eum_mean"] == float(old_values.mean())
    assert optimized["bootstrap_eum_std"] == float(old_values.std(ddof=0))


def test_compute_diagnostics_matches_prompt_metric_reference():
    rng = np.random.default_rng(2026)
    rewards = rng.uniform(size=(6, 7, 4))
    seed = 23
    n_weights = 41
    bootstrap_samples = 11
    target_weights = np.array([0.1, 0.2, 0.3, 0.4])
    train_weights = np.array([0.4, 0.3, 0.2, 0.1])
    weights = dirichlet_weights(rewards.shape[2], n_weights, seed=seed)
    normalized_target = target_weights / target_weights.sum()
    normalized_train = train_weights / train_weights.sum()

    per_prompt = [
        prompt_metrics(pool, weights, normalized_target, seed + i, train_weights=normalized_train)
        for i, pool in enumerate(rewards)
    ]
    keys = sorted({k for row in per_prompt for k in row})
    expected_metrics = {
        key: float(np.mean([row[key] for row in per_prompt if row.get(key) is not None]))
        for key in keys
        if any(row.get(key) is not None for row in per_prompt)
    }
    expected_metrics.update(
        {
            "num_prompts": int(rewards.shape[0]),
            "pool_size": int(rewards.shape[1]),
            "num_objectives": int(rewards.shape[2]),
        }
    )
    expected_metrics.update(
        bootstrap_eum(rewards, weights, seed=seed, samples=bootstrap_samples)
    )

    actual = compute_diagnostics(
        rewards,
        n_weights=n_weights,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        target_weights=target_weights,
        train_weights=train_weights,
    )

    for key, expected in expected_metrics.items():
        assert np.isclose(actual["metrics"][key], expected)
    assert actual["metric_provenance"] == "scripts/geometry_diagnostics.py::frozen_geometry_diagnostics_v2"
    assert len(actual["per_prompt"]) == len(per_prompt)
    for actual_row, expected_row in zip(actual["per_prompt"], per_prompt):
        assert actual_row.keys() == expected_row.keys()
        for key, expected in expected_row.items():
            assert np.isclose(actual_row[key], expected)
