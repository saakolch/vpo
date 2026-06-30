from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts import eval as eval_script


def make_stage2_manifest(repo: Path) -> Path:
    root = repo / "pre_experiments" / "2_short_training"
    run_dir = root / "smoke" / "vpo_maze_seed0_short_training_a1"
    eval_path = run_dir / "eval" / "update_0" / "eval.json"
    tensor_path = eval_path.with_suffix(".npy")
    diag_path = run_dir / "diagnostics" / "diag.json"
    health_path = run_dir / "health" / "health.json"
    summary_path = run_dir / "job_summaries" / "summary.json"
    for path in (eval_path, diag_path, health_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(
        json.dumps(
            {
                "per_objective_mean": {"completion": 1.0, "gold": 0.5},
                "best_at_k": {"best@1": 0.25, "best@3": 0.75},
                "tensor_path": str(tensor_path),
            }
        )
    )
    np.save(tensor_path, np.asarray([[[1.0, 0.5], [0.0, 0.25]]]))
    diag_path.write_text(json.dumps({"metrics": {"eum_gap": 0.2, "winner_entropy": 0.7, "base_best_of_k_slope": 0.1}}))
    health_path.write_text(json.dumps({"metrics": {"kl_to_reference": 0.01, "parse_success_rate": 1.0}}))
    summary_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "train_command": "frozen_base_model_no_training",
                "eval_command": "python eval/eval_maze.py",
                "diagnostics_command": "python scripts/geometry_diagnostics.py",
                "health_command": "python scripts/collect_train_health.py",
            }
        )
    )
    manifest = root / "manifests" / "short_training_smoke_jobs.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "benchmark": "maze",
        "method": "vpo",
        "model": "Qwen/Qwen3-0.6B",
        "seed": 0,
        "update": 0,
        "checkpoint_path": "model://Qwen/Qwen3-0.6B",
        "eval_output_path": str(eval_path),
        "tensor_path": str(tensor_path),
        "diagnostics_path": str(diag_path),
        "health_path": str(health_path),
        "job_summary_path": str(summary_path),
        "artifact_paths": [str(eval_path), str(tensor_path), str(diag_path), str(health_path), str(summary_path)],
        "dataset_split": "test",
        "metric_provenance": "scripts/eval.py::short_training_official_eval_tensor",
    }
    manifest.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id": "vpo_maze_seed0_short_training_a1",
                        "checkpoints": [record],
                    }
                ]
            }
        )
    )
    return manifest


def test_short_training_eval_writes_required_tables(tmp_path, monkeypatch):
    import scripts.short_training_common as common

    monkeypatch.setattr(common, "repo_root", lambda: tmp_path)
    manifest = make_stage2_manifest(tmp_path)
    output_dir = tmp_path / "pre_experiments" / "2_short_training" / "tables"

    outputs = eval_script.write_short_training_tables(manifest, output_dir)

    assert {path.name for path in outputs.values()} == {
        "checkpoint_table.csv",
        "method_difference_table.csv",
        "diagnostic_table.csv",
        "train_health_table.csv",
        "final_short_training_summary.md",
    }
    with (output_dir / "checkpoint_table.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["benchmark"] == "maze"
    assert rows[0]["update"] == "0"
    assert rows[0]["best@1"] == "0.25"
    assert "completion" in rows[0]["per_dimension_reward_means"]

    with (output_dir / "train_health_table.csv").open(newline="") as f:
        health = list(csv.DictReader(f))
    assert health[0]["parse_format_success_rate"] == "1.0"


def test_eval_script_direct_execution_context_can_import_stage2_common(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")
    code = (
        "import runpy; "
        f"ns = runpy.run_path({str(repo / 'scripts' / 'eval.py')!r}); "
        "import scripts.short_training_common as common; "
        "assert ns['repo_root']() == common.repo_root()"
    )

    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
