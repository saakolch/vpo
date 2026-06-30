from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import eval as eval_script
from scripts import short_training_common as short_common


def row(checkpoint: Path) -> dict:
    return {
        "benchmark": "maze",
        "method": "vpo",
        "model": "Qwen/Qwen3-0.6B",
        "seed": 0,
        "checkpoint_path": str(checkpoint),
        "train_command": "METHOD=vpo TASK=maze bash train.sh",
        "eval_command": ".venv/bin/python eval/eval_maze.py --method vpo",
        "artifact_paths": ["pre_experiments/first_experiment/example.json"],
        "metric_provenance": "scripts/eval.py::test",
        "dataset_split": "test",
        "mean": 0.5,
    }


def test_eval_refuses_missing_checkpoint_paths(tmp_path):
    missing = tmp_path / "missing_checkpoint"

    with pytest.raises(ValueError, match="checkpoint_path does not exist"):
        eval_script.aggregate_rows([row(missing)])


def test_eval_refuses_missing_command_provenance(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    bad = row(checkpoint)
    bad["train_command"] = ""

    with pytest.raises(ValueError, match="train_command"):
        eval_script.aggregate_rows([bad])


def test_eval_refuses_output_roots_outside_pre_experiments(tmp_path):
    with pytest.raises(ValueError, match="outside pre_experiments"):
        eval_script.require_pre_experiment_path(tmp_path / "table.json")


def test_eval_writes_aggregate_table_under_pre_experiments(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    pre = repo / "pre_experiments" / "first_experiment"
    checkpoint = pre / "ckpt"
    checkpoint.mkdir(parents=True)
    manifest = pre / "manifest.json"
    output = pre / "table.json"
    manifest.write_text(json.dumps({"rows": [row(checkpoint)]}))
    monkeypatch.setattr(eval_script, "repo_root", lambda: repo)

    rows = eval_script.aggregate_rows(eval_script.read_manifest(manifest))
    out_path = eval_script.require_pre_experiment_path(output)
    eval_script.write_table(out_path, rows, "json")

    written = json.loads(output.read_text())
    assert written[0]["seed_count"] == 1
    assert written[0]["mean"] == 0.5
    assert written[0]["std"] == 0.0


def test_eval_can_extract_rows_from_launcher_job_manifest(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    pre = repo / "pre_experiments" / "first_experiment"
    checkpoint = pre / "ckpt"
    checkpoint.mkdir(parents=True)
    summary = pre / "eval.json"
    summary.write_text(
        json.dumps(
            {
                "per_objective_mean": {
                    "completion": 1.0,
                    "gold": 0.5,
                    "diamond": 0.25,
                    "avoid_lava": 0.75,
                },
                "best_at_k": {"best@1": 0.4, "best@3": 0.6},
                "tensor_path": str(pre / "eval.npy"),
            }
        )
    )
    manifest = pre / "jobs.json"
    job = row(checkpoint)
    job.update(
        {
            "eval_output_path": str(summary),
            "artifact_paths": [str(summary)],
            "pre_training_diagnostics": str(pre / "diag.json"),
        }
    )
    manifest.write_text(json.dumps({"jobs": [job]}))
    monkeypatch.setattr(eval_script, "repo_root", lambda: repo)

    rows = eval_script.aggregate_rows(eval_script.rows_from_job_manifest(manifest))

    assert rows[0]["mean"] == pytest.approx(0.625)
    assert rows[0]["best@1"] == pytest.approx(0.4)
    assert rows[0]["best@3"] == pytest.approx(0.6)
    assert rows[0]["attempt_label"] == ""


def test_eval_job_manifest_is_strict_about_missing_eval_by_default(tmp_path):
    manifest = tmp_path / "jobs.json"
    job = row(tmp_path / "model://unused")
    job.update({"eval_output_path": str(tmp_path / "missing.json")})
    manifest.write_text(json.dumps({"jobs": [job]}))

    with pytest.raises(ValueError, match="Missing eval artifact"):
        eval_script.rows_from_job_manifest(manifest)


def test_eval_job_manifest_can_skip_missing_eval_for_partial_tables(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    pre = repo / "pre_experiments" / "first_experiment"
    checkpoint = pre / "ckpt"
    checkpoint.mkdir(parents=True)
    complete_summary = pre / "complete.json"
    complete_summary.write_text(json.dumps({"mean_scalar_reward": 0.25}))
    complete_job = row(checkpoint)
    complete_job.update({"eval_output_path": str(complete_summary), "artifact_paths": [str(complete_summary)]})
    missing_job = row(checkpoint)
    missing_job.update({"method": "maxrl", "eval_output_path": str(pre / "missing.json")})
    manifest = pre / "jobs.json"
    manifest.write_text(json.dumps({"jobs": [complete_job, missing_job]}))
    monkeypatch.setattr(eval_script, "repo_root", lambda: repo)

    rows = eval_script.aggregate_rows(eval_script.rows_from_job_manifest(manifest, skip_missing=True))

    assert len(rows) == 1
    assert rows[0]["method"] == "vpo"
    assert rows[0]["mean"] == pytest.approx(0.25)


def test_eval_preserves_attempt_and_gpu_resource_provenance(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    pre = repo / "pre_experiments" / "first_experiment"
    checkpoint = pre / "ckpt"
    checkpoint.mkdir(parents=True)
    summary = pre / "eval.json"
    summary.write_text(json.dumps({"mean_scalar_reward": 0.75}))
    slurm = pre / "run.sbatch"
    slurm.write_text("#!/bin/bash\n#SBATCH --partition=i64m1tga800ue\n#SBATCH --gres=gpu:a800:2\n")
    manifest = pre / "jobs.json"
    job = row(checkpoint)
    job.update(
        {
            "attempt_label": "seed0_parallel_2gpu_a1",
            "job_kind": "train_eval",
            "slurm_id": "9923712",
            "slurm_path": str(slurm),
            "train_command": "METHOD=maxrl TASK=maze N_GPUS=2 bash train.sh",
            "eval_output_path": str(summary),
            "artifact_paths": [str(summary)],
        }
    )
    manifest.write_text(json.dumps({"jobs": [job]}))
    monkeypatch.setattr(eval_script, "repo_root", lambda: repo)

    rows = eval_script.aggregate_rows(eval_script.rows_from_job_manifest(manifest))

    assert rows[0]["attempt_label"] == "seed0_parallel_2gpu_a1"
    assert rows[0]["trainer_n_gpus"] == "2"
    assert rows[0]["slurm_partition"] == "i64m1tga800ue"
    assert rows[0]["slurm_gres"] == "gpu:a800:2"
    assert rows[0]["slurm_id"] == "9923712"
    assert "trainer_n_gpus=2" in rows[0]["resource_provenance"]
    assert "slurm_gres=gpu:a800:2" in rows[0]["resource_provenance"]


def test_eval_does_not_merge_distinct_resource_attempts(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    row_4gpu = row(checkpoint)
    row_4gpu.update(
        {
            "attempt_label": "a1",
            "resource_provenance": "attempt=a1; trainer_n_gpus=4; slurm_gres=gpu:a800:4",
            "trainer_n_gpus": "4",
            "mean": 0.4,
        }
    )
    row_2gpu = row(checkpoint)
    row_2gpu.update(
        {
            "attempt_label": "seed0_parallel_2gpu_a1",
            "resource_provenance": "attempt=seed0_parallel_2gpu_a1; trainer_n_gpus=2; slurm_gres=gpu:a800:2",
            "trainer_n_gpus": "2",
            "mean": 0.6,
        }
    )

    rows = eval_script.aggregate_rows([row_4gpu, row_2gpu])

    assert len(rows) == 2
    assert {result["attempt_label"] for result in rows} == {"a1", "seed0_parallel_2gpu_a1"}


def test_eval_extracts_top_level_per_component_best_at_shape(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    pre = repo / "pre_experiments" / "first_experiment"
    checkpoint = pre / "ckpt"
    checkpoint.mkdir(parents=True)
    summary = pre / "eureqa_eval.json"
    summary.write_text(
        json.dumps(
            {
                "per_component_mean": [1.0, 0.5, 0.0],
                "best_at_k/k=1/mean": 0.25,
                "best_at_k/k=3/mean": 0.75,
            }
        )
    )
    manifest = pre / "jobs.json"
    job = row(checkpoint)
    job.update({"eval_output_path": str(summary), "artifact_paths": [str(summary)]})
    manifest.write_text(json.dumps({"jobs": [job]}))
    monkeypatch.setattr(eval_script, "repo_root", lambda: repo)

    rows = eval_script.aggregate_rows(eval_script.rows_from_job_manifest(manifest))

    assert rows[0]["mean"] == pytest.approx(0.5)
    assert rows[0]["best@1"] == pytest.approx(0.25)
    assert rows[0]["best@3"] == pytest.approx(0.75)


def test_eval_extracts_nested_musique_summary_scalar(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    pre = repo / "pre_experiments" / "first_experiment"
    checkpoint = pre / "ckpt"
    checkpoint.mkdir(parents=True)
    summary = pre / "musique_eval.json"
    summary.write_text(json.dumps({"summary": {"Ew_max_pool": 0.625}}))
    manifest = pre / "jobs.json"
    job = row(checkpoint)
    job.update({"eval_output_path": str(summary), "artifact_paths": [str(summary)]})
    manifest.write_text(json.dumps({"jobs": [job]}))
    monkeypatch.setattr(eval_script, "repo_root", lambda: repo)

    rows = eval_script.aggregate_rows(eval_script.rows_from_job_manifest(manifest))

    assert rows[0]["mean"] == pytest.approx(0.625)


def test_eval_writes_short_training_table_set(tmp_path, monkeypatch):
    import numpy as np

    repo = tmp_path / "repo"
    root = repo / "pre_experiments" / "2_short_training"
    run = root / "smoke" / "grpo_maze_seed0"
    eval_dir = run / "eval" / "update_0"
    diag_dir = run / "diagnostics"
    health_dir = run / "health"
    summary_dir = run / "job_summaries"
    for path in (eval_dir, diag_dir, health_dir, summary_dir):
        path.mkdir(parents=True)

    tensor = eval_dir / "eval.npy"
    np.save(tensor, np.ones((2, 3, 4), dtype=np.float64))
    eval_json = eval_dir / "eval.json"
    eval_json.write_text(
        json.dumps(
            {
                "per_objective_mean": {"completion": 1.0, "gold": 0.5},
                "best_at_k": {"best@1": 0.25, "best@3": 0.75},
                "tensor_path": str(tensor),
                "feature_names": ["completion", "gold", "diamond", "avoid_lava"],
            }
        )
    )
    diag = diag_dir / "diag.json"
    diag.write_text(json.dumps({"metrics": {"eum_gap": 0.2, "winner_entropy": 0.3, "base_best_of_k_slope": 0.4}}))
    health = health_dir / "health.json"
    health.write_text(json.dumps({"metrics": {"kl_to_reference": 0.01, "grad_norm": 1.0, "parse_format_success_rate": 0.5}}))
    summary = summary_dir / "summary.json"
    summary.write_text(json.dumps({"status": "PASS", "train_command": "frozen_base_model_no_training", "eval_command": "python eval/eval_maze.py"}))
    manifest = root / "manifests" / "jobs.json"
    manifest.parent.mkdir(parents=True)
    record = {
        "benchmark": "maze",
        "method": "grpo",
        "model": "Qwen/Qwen3-0.6B",
        "seed": 0,
        "update": 0,
        "checkpoint_path": "model://Qwen/Qwen3-0.6B",
        "train_command": "frozen_base_model_no_training",
        "eval_command": "generated",
        "artifact_paths": [str(eval_json), str(tensor), str(diag), str(health), str(summary)],
        "dataset_split": "test",
        "metric_provenance": "scripts/eval.py::short_training_official_eval_tensor",
        "eval_output_path": str(eval_json),
        "tensor_path": str(tensor),
        "diagnostics_path": str(diag),
        "health_path": str(health),
        "job_summary_path": str(summary),
    }
    manifest.write_text(json.dumps({"jobs": [{"job_id": "j", "checkpoints": [record]}]}))
    monkeypatch.setattr(eval_script, "repo_root", lambda: repo)
    monkeypatch.setattr(short_common, "repo_root", lambda: repo)

    outputs = eval_script.write_short_training_tables(manifest, root / "tables")

    assert outputs["checkpoint"].exists()
    checkpoint_text = outputs["checkpoint"].read_text()
    assert "best@1" in checkpoint_text
    assert "parse_format_success_rate" in checkpoint_text
    assert outputs["diagnostic"].exists()
    assert outputs["train_health"].exists()
    assert outputs["summary"].read_text().startswith("# Final Short-Training Summary")
