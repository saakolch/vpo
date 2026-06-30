from __future__ import annotations

import csv

from scripts.plot_short_training_curves import generate_plots


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_generate_short_training_plots(tmp_path, monkeypatch):
    import scripts.short_training_common as common

    monkeypatch.setattr(common, "repo_root", lambda: tmp_path)
    root = tmp_path / "pre_experiments" / "2_short_training"
    tables = root / "tables"
    plots = root / "plots"
    write_csv(
        tables / "checkpoint_table.csv",
        [
            {"method": "grpo", "update": 0, "best@1": 0.1, "winner_entropy": 0.2, "eum_gap": 0.1, "target_regret": 0.2, "reward_collinearity": 0.3, "effective_rank": 1.0},
            {"method": "grpo", "update": 1, "best@1": 0.2, "winner_entropy": 0.3, "eum_gap": 0.2, "target_regret": 0.1, "reward_collinearity": 0.2, "effective_rank": 1.5},
        ],
    )
    write_csv(
        tables / "train_health_table.csv",
        [
            {"method": "grpo", "update": 0, "kl_to_reference": 0.0, "grad_norm": 1.0, "clip_fraction": 0.0, "parse_format_success_rate": 0.5, "response_length_mean": 10},
            {"method": "grpo", "update": 1, "kl_to_reference": 0.1, "grad_norm": 1.5, "clip_fraction": 0.1, "parse_format_success_rate": 0.75, "response_length_mean": 12},
        ],
    )

    written = generate_plots(tables, plots)

    assert len(written) >= 10
    assert (plots / "best_at_1_over_updates.png").exists()
    assert (plots / "parse_format_success_rate_over_updates.png").exists()
