from __future__ import annotations

from scripts.assign_geometry_strata import assign_strata


def row(idx: int, **metrics):
    base = {
        "row_type": "prompt",
        "benchmark": "maze",
        "model": "model",
        "prompt_index": str(idx),
        "reward_collinearity_active": "0.1",
        "effective_rank_entropy": "2.0",
        "unique_pareto_fraction": "0.2",
        "eum_gap": "0.1",
        "winner_cluster_entropy_normalized": "0.5",
        "dominant_cluster_mass": "0.5",
        "target_regret_fixed": "0.1",
        "best_of_k_slope": "0.1",
        "best@30": "0.1",
        "bootstrap_stability": "1.0",
    }
    base.update({k: str(v) for k, v in metrics.items()})
    return base


def test_assign_strata_labels_prompt_rows():
    rows = [
        row(0, reward_collinearity_active=0.99, effective_rank_entropy=1.0),
        row(1, winner_cluster_entropy_normalized=0.01, dominant_cluster_mass=0.99),
        row(2, winner_cluster_entropy_normalized=0.99, target_regret_fixed=0.99),
        row(3, best_of_k_slope=0.99, **{"best@30": 0.99}),
    ]

    strata, thresholds = assign_strata(rows)

    labels = {item["geometry_stratum"] for item in strata}
    assert "scalar_like" in labels
    assert "dominant_candidate" in labels
    assert "off_target_diversity" in labels or "search_sufficient" in labels
    assert "reward_collinearity_active" in thresholds
