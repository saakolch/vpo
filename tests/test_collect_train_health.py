from __future__ import annotations

import json

from scripts.collect_train_health import collect, parse_metric_line


def test_parse_metric_line_extracts_training_health_keys():
    line = "step:3 - actor/ppo_kl:0.125 - actor/grad_norm:2.5 - actor/pg_clipfrac:0.01 - response_length/mean:42"

    parsed = parse_metric_line(line)

    assert parsed is not None
    assert parsed["step"] == 3
    assert parsed["actor/ppo_kl"] == 0.125
    assert parsed["actor/grad_norm"] == 2.5
    assert parsed["actor/pg_clipfrac"] == 0.01


def test_collect_reads_logs_and_rollouts(tmp_path, monkeypatch):
    import scripts.short_training_common as common

    root = tmp_path / "pre_experiments" / "2_short_training"
    monkeypatch.setattr(common, "repo_root", lambda: tmp_path)
    run_dir = root / "smoke" / "grpo_maze_seed0_short_training_a1"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "run.log").write_text(
        "step:1 - actor/ppo_kl:0.5 - actor/grad_norm:1.5 - actor/pg_clipfrac:0.25 - response_length/mean:100\n"
    )
    rollout_dir = run_dir / "rollouts" / "grpo_maze_seed0_short_training_a1"
    rollout_dir.mkdir(parents=True)
    (rollout_dir / "1.jsonl").write_text(
        json.dumps({"output": "one two three", "num_programs_executed": 1}) + "\n"
        + json.dumps({"output": "one", "num_programs_executed": 0}) + "\n"
    )

    out = collect(run_dir, 1)

    assert out["metrics"]["kl_to_reference"] == 0.5
    assert out["metrics"]["grad_norm"] == 1.5
    assert out["metrics"]["clip_fraction"] == 0.25
    assert out["metrics"]["parse_success_rate"] == 0.5
    assert out["metrics"]["rollout_response_length_mean"] == 2.0
