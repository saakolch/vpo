from __future__ import annotations

from pathlib import Path

import pytest

from scripts import launch_short_training_grid as launcher
from scripts import short_training_common as common


def make_config(tmp_path: Path) -> tuple[dict, dict]:
    snapshot = tmp_path / "snapshots" / "qwen"
    snapshot.mkdir(parents=True)
    config = {
        "experiment": {"output_root": "pre_experiments/2_short_training"},
        "defaults": {"seeds": [0], "dataset_split": "test", "train_overrides": ["trainer.logger=[console]"]},
        "methods": {"active": ["grpo", "maxrl", "max_at_k", "multi_rlvr", "vpo"], "eval_aliases": {"maze": {"max_at_k": "maxrl"}}},
        "safety": {"max_gpu_hours_without_override": 4},
        "smoke": {
            "benchmark": "maze",
            "model": "Qwen/Qwen3-0.6B",
            "methods": ["grpo"],
            "seeds": [0],
            "updates": [0, 1],
            "estimated_gpu_hours": 1,
            "partition_class": "smoke",
            "data": {"train_size": 4, "test_size": 4, "train_seed": 42, "test_seed": 4242},
            "training": {"epochs": 1, "n_gpus": 1},
            "eval": {"n_chains": 2, "max_tokens": 128, "temperature": 0.7},
        },
        "benchmarks": {},
    }
    site = {
        "site": {"python": "/opt/python"},
        "environment": {"hf_home": str(tmp_path / "hf")},
        "model_snapshots": {"Qwen/Qwen3-0.6B": str(snapshot)},
        "slurm": {"smoke": {"partition": "debug", "gres": "gpu:a40:1", "cpus_per_task": 2, "mem": "8G", "time": "00:10:00"}},
    }
    return config, site


def patch_roots(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(common, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(launcher, "repo_root", lambda: tmp_path)


def test_short_training_launcher_builds_segmented_checkpoint_records(tmp_path, monkeypatch):
    patch_roots(monkeypatch, tmp_path)
    config, site = make_config(tmp_path)

    jobs = launcher.build_jobs(config, site, True, ["maze"], None, None, None, "a1")

    assert len(jobs) == 1
    job = jobs[0]
    assert job["updates"] == [0, 1]
    assert job["checkpoints"][0]["checkpoint_path"] == "model://Qwen/Qwen3-0.6B"
    assert job["checkpoints"][1]["checkpoint_path"].endswith("global_step_1/actor/huggingface_merged")
    assert job["checkpoints"][1]["previous_update"] is None


def test_short_training_launcher_writes_manifest_and_slurm(tmp_path, monkeypatch):
    patch_roots(monkeypatch, tmp_path)
    (tmp_path / "pre_experiments" / "2_short_training").mkdir(parents=True)
    config, site = make_config(tmp_path)
    jobs = launcher.build_jobs(config, site, True, ["maze"], "grpo", "0", None, "tqdm")
    manifest = launcher.manifest_path_for(config, "smoke", "tqdm")

    launcher.write_manifest(manifest, tmp_path / "cfg.yaml", tmp_path / "site.yaml", jobs, False)
    launcher.write_slurm(jobs[0], manifest, site, False)

    assert manifest.exists()
    slurm = Path(jobs[0]["slurm_path"]).read_text()
    assert "scripts/run_short_training_job.py" in slurm
    assert "--allow_main_experiments=false" in slurm
    assert "#SBATCH --partition=debug" in slurm


def test_stage2_path_guard_rejects_non_stage2_outputs(tmp_path, monkeypatch):
    patch_roots(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="outside pre_experiments/2_short_training"):
        common.require_stage2_path(tmp_path / "pre_experiments" / "first_experiment" / "bad.json")
