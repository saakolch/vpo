from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts import base_geometry_grid as base
import scripts.geometry_diagnostics as gd


def test_plan_pairs_defers_missing_models(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    (snap / "tokenizer.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"fake")
    config = {
        "diagnostics": {"dataset_split": "train"},
        "benchmarks": {
            "maze": {"models": ["available/model", "missing/model"]},
            "ultrafeedback": {"enabled": False, "models": ["missing/uf"]},
        },
    }
    site = {
        "environment": {"hf_home": str(tmp_path / "hf")},
        "model_snapshots": {"available/model": str(snap)},
    }

    available, deferred = base.plan_pairs(config, site, "configs/site/hpc2.yaml")

    assert [row["model"] for row in available] == ["available/model"]
    assert [row["model"] for row in deferred] == ["missing/model"]
    assert deferred[0]["model_status"] == "deferred"


def test_plan_pairs_defers_incomplete_indexed_snapshot(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    (snap / "tokenizer.json").write_text("{}")
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001-of-00002.safetensors"}})
    )
    config = {
        "diagnostics": {"dataset_split": "train"},
        "benchmarks": {"eureqa": {"models": ["indexed/model"]}},
    }
    site = {
        "environment": {"hf_home": str(tmp_path / "hf")},
        "model_snapshots": {"indexed/model": str(snap)},
    }

    available, deferred = base.plan_pairs(config, site, "configs/site/hpc2.yaml")

    assert available == []
    assert [row["model"] for row in deferred] == ["indexed/model"]


def test_filter_pairs_and_stage_dirs(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    root = repo / "pre_experiments" / "1_frozen_diagnostic"
    monkeypatch.setattr(gd, "repo_root", lambda: repo)
    monkeypatch.setattr(base, "repo_root", lambda: repo)
    pairs = [
        {"benchmark": "maze", "model": "a"},
        {"benchmark": "maze", "model": "b"},
        {"benchmark": "eureqa", "model": "a"},
    ]

    assert base.filter_pairs(pairs, benchmark="maze", model="a") == [{"benchmark": "maze", "model": "a"}]

    base.ensure_stage_dirs(root, {"diagnostics": {"checkpoints": [5, 20, 100]}})
    assert (root / "results" / "run_5%").is_dir()
    assert (root / "results" / "run_20%").is_dir()
    assert (root / "results" / "run_100%").is_dir()


def test_estimated_gpu_hours_uses_benchmark_override():
    config = {
        "diagnostics": {"estimated_gpu_hours_per_pair": 1},
        "benchmarks": {
            "maze": {"estimated_gpu_hours": 1},
            "musique": {"estimated_gpu_hours": 8},
        },
    }
    pairs = [
        {"benchmark": "maze", "model": "Qwen/Qwen3-0.6B"},
        {"benchmark": "musique", "model": "Qwen/Qwen3-1.7B"},
    ]

    assert base.estimated_gpu_hours(config, pairs) == 9.0


def test_ensure_data_allows_dataset_downloads_without_changing_model_offline_env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data_dir = repo / "pre_experiments" / "1_frozen_diagnostic" / "data" / "musique"
    monkeypatch.setattr(base, "repo_root", lambda: repo)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    captured = {}

    def fake_run(cmd, cwd, text, env):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["text"] = text
        captured["env"] = env

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(base.subprocess, "run", fake_run)

    base.ensure_data(
        {"diagnostics": {"dataset_split": "train"}, "benchmarks": {"musique": {"data": {}}}},
        {
            "site": {"python": "/tmp/python"},
            "environment": {
                "allow_dataset_downloads": "1",
                "hf_home": str(tmp_path / "hf_cache"),
                "pythonpath": "/tmp/repo:/tmp/site-packages",
            },
        },
        "musique",
        data_dir,
    )

    assert captured["cmd"][:2] == ["/tmp/python", "data/preprocess_musique.py"]
    assert captured["cwd"] == repo
    assert captured["text"] is True
    assert "HF_HUB_OFFLINE" not in captured["env"]
    assert "TRANSFORMERS_OFFLINE" not in captured["env"]
    assert captured["env"]["HF_HOME"] == str(tmp_path / "hf_cache")
    assert captured["env"]["PYTHONPATH"] == "/tmp/repo:/tmp/site-packages"


def test_process_tensor_stage_writes_tsv_strata_and_pngs(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    root = repo / "pre_experiments" / "1_frozen_diagnostic"
    root.mkdir(parents=True)
    monkeypatch.setattr(gd, "repo_root", lambda: repo)
    monkeypatch.setattr(base, "repo_root", lambda: repo)
    config = {
        "diagnostics": {
            "dataset_split": "train",
            "n_weights": 64,
            "bootstrap_samples": 8,
            "seed": 0,
        }
    }
    tensor = np.random.default_rng(0).uniform(size=(4, 10, 3, 4))

    base.process_tensor_stage(
        tensor=tensor,
        config=config,
        root=root,
        percent=20,
        benchmark="maze",
        model="Qwen/Qwen3-0.6B",
        phase="available",
        model_status="available",
        snapshot="/tmp/snapshot",
        download_cmd="huggingface-cli download Qwen/Qwen3-0.6B",
        verify_cmd="verify",
        total_prompts=16,
    )

    run_dir = root / "results" / "run_20%"
    assert (run_dir / "metrics.tsv").exists()
    assert (run_dir / "summary.tsv").exists()
    assert (run_dir / "strata.tsv").exists()
    assert (run_dir / "strata_thresholds.json").exists()
    assert (run_dir / "reward_collinearity_active_distribution.png").exists()
    marker = run_dir / "checkpoint_markers" / "maze__Qwen_Qwen3-0.6B__available.json"
    assert marker.exists()
    assert json.loads(marker.read_text())["limited_run"] is False

    base.process_tensor_stage(
        tensor=tensor,
        config=config,
        root=root,
        percent=20,
        benchmark="maze",
        model="Qwen/Qwen3-8B",
        phase="deferred_downloaded",
        model_status="downloaded_verified",
        snapshot="/tmp/snapshot_8b",
        download_cmd="huggingface-cli download Qwen/Qwen3-8B",
        verify_cmd="verify 8b",
        total_prompts=16,
    )

    with (run_dir / "metrics.tsv").open(newline="") as f:
        metric_rows = list(csv.DictReader(f, delimiter="\t"))
    with (run_dir / "summary.tsv").open(newline="") as f:
        summary_rows = list(csv.DictReader(f, delimiter="\t"))
    assert {"Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B"} <= {row["model"] for row in metric_rows}
    assert {"Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B"} <= {row["model"] for row in summary_rows}
    assert (run_dir / "best_of_k_slope_curves.png").exists()


def test_missing_stage_pairs_tracks_phase1_completion(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    root = repo / "pre_experiments" / "1_frozen_diagnostic"
    root.mkdir(parents=True)
    monkeypatch.setattr(gd, "repo_root", lambda: repo)
    monkeypatch.setattr(base, "repo_root", lambda: repo)
    config = {
        "diagnostics": {
            "dataset_split": "train",
            "checkpoints": [5, 20],
            "n_weights": 16,
            "bootstrap_samples": 0,
            "seed": 0,
        }
    }
    pair = {"benchmark": "maze", "model": "Qwen/Qwen3-0.6B"}
    tensor = np.random.default_rng(1).uniform(size=(2, 3, 2, 2))

    assert len(base.missing_stage_pairs(root, config, [pair])) == 2

    base.process_tensor_stage(
        tensor=tensor,
        config=config,
        root=root,
        percent=5,
        benchmark=pair["benchmark"],
        model=pair["model"],
        phase="available",
        model_status="available",
        snapshot="/tmp/snapshot",
        download_cmd="download",
        verify_cmd="verify",
        total_prompts=2,
    )
    assert len(base.missing_stage_pairs(root, config, [pair])) == 1

    base.process_tensor_stage(
        tensor=tensor,
        config=config,
        root=root,
        percent=20,
        benchmark=pair["benchmark"],
        model=pair["model"],
        phase="available",
        model_status="available",
        snapshot="/tmp/snapshot",
        download_cmd="download",
        verify_cmd="verify",
        total_prompts=2,
    )
    assert base.missing_stage_pairs(root, config, [pair]) == []


def test_missing_stage_pairs_rejects_limited_smoke(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    root = repo / "pre_experiments" / "1_frozen_diagnostic"
    root.mkdir(parents=True)
    monkeypatch.setattr(gd, "repo_root", lambda: repo)
    monkeypatch.setattr(base, "repo_root", lambda: repo)
    config = {
        "diagnostics": {
            "dataset_split": "train",
            "checkpoints": [5],
            "n_weights": 16,
            "bootstrap_samples": 0,
            "seed": 0,
        },
        "benchmarks": {"maze": {"data": {"train_size": 1000}}},
    }
    pair = {"benchmark": "maze", "model": "Qwen/Qwen3-0.6B"}
    tensor = np.random.default_rng(2).uniform(size=(16, 3, 2, 2))

    base.process_tensor_stage(
        tensor=tensor,
        config=config,
        root=root,
        percent=5,
        benchmark=pair["benchmark"],
        model=pair["model"],
        phase="available",
        model_status="available",
        snapshot="/tmp/snapshot",
        download_cmd="download",
        verify_cmd="verify",
        total_prompts=16,
        limited_run=True,
        max_prompts=16,
    )

    missing = base.missing_stage_pairs(root, config, [pair])
    assert missing[0]["reason"] == "limited_run"


def test_estimated_gpu_hours_sums_benchmark_estimates():
    config = {
        "diagnostics": {"estimated_gpu_hours_per_pair": 1},
        "benchmarks": {"maze": {"estimated_gpu_hours": 2}},
    }
    pairs = [{"benchmark": "maze"}, {"benchmark": "tool"}]

    assert base.estimated_gpu_hours(config, pairs) == 3.0


def test_progress_description_uses_short_model_name():
    assert base.progress_description("musique", "Qwen/Qwen3-1.7B", "available") == "musique/Qwen3-1.7B available"


def test_base_config_musique_context_covers_measured_train_prompts():
    config = base.load_yaml(Path("configs/geometry/base_diagnostic_grid.yaml"))
    musique_eval = config["benchmarks"]["musique"]["eval"]

    # Measured on the configured MuSiQue train split with Qwen3 tokenizer:
    # max prompt length is 7384 tokens, before reserving generation budget.
    assert musique_eval["max_model_len"] >= 7384 + musique_eval["max_tokens"]


def test_prepare_enabled_data_uses_configured_benchmarks(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    root = repo / "pre_experiments" / "1_frozen_diagnostic"
    monkeypatch.setattr(gd, "repo_root", lambda: repo)
    monkeypatch.setattr(base, "repo_root", lambda: repo)
    config = {
        "experiment": {"output_root": str(root.relative_to(repo))},
        "benchmarks": {
            "maze": {"enabled": True},
            "musique": {"enabled": True},
            "ultrafeedback": {"enabled": False},
        },
    }
    calls = []

    def fake_ensure_data(_config, _site, benchmark, data_dir):
        calls.append((benchmark, data_dir))

    monkeypatch.setattr(base, "ensure_data", fake_ensure_data)

    prepared = base.prepare_enabled_data(config, {})

    assert prepared == ["maze", "musique"]
    assert calls == [
        ("maze", root / "data" / "maze"),
        ("musique", root / "data" / "musique"),
    ]


def test_deferred_completion_verifies_before_run(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    root = repo / "pre_experiments" / "1_frozen_diagnostic"
    root.mkdir(parents=True)
    monkeypatch.setattr(gd, "repo_root", lambda: repo)
    monkeypatch.setattr(base, "repo_root", lambda: repo)
    config = {"benchmarks": {"eureqa": {"models": ["missing/model"]}}}
    site = {"environment": {"hf_home": str(tmp_path / "hf")}}
    deferred = [{
        "benchmark": "eureqa",
        "model": "missing/model",
        "dataset_split": "train",
        "phase": "deferred",
        "model_status": "deferred",
        "snapshot_path": "",
        "download_command": "huggingface-cli download missing/model",
        "verification_command": "verify missing/model",
    }]
    calls = []
    monkeypatch.setattr(base, "download_deferred_model", lambda _site, _model: True)
    monkeypatch.setattr(base, "snapshot_path", lambda _site, _model: "/tmp/snapshot")
    monkeypatch.setattr(base, "verify_model", lambda _site, _model: (False, "broken"))
    monkeypatch.setattr(base, "run_pair", lambda *args, **kwargs: calls.append((args, kwargs)))

    failed = base.complete_deferred_models(config, site, root, deferred, allow_downloads=True)

    assert calls == []
    assert failed[0]["model_status"] == "failed_verify"

    monkeypatch.setattr(base, "verify_model", lambda _site, _model: (True, "/tmp/snapshot"))
    completed = base.complete_deferred_models(config, site, root, deferred, allow_downloads=True)

    assert len(calls) == 1
    pair = calls[0][0][2]
    assert pair["phase"] == "deferred_downloaded"
    assert pair["model_status"] == "downloaded_verified"
    assert completed[0]["model_status"] == "downloaded_verified"


def test_deferred_completion_logs_failed_download(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    root = repo / "pre_experiments" / "1_frozen_diagnostic"
    root.mkdir(parents=True)
    monkeypatch.setattr(gd, "repo_root", lambda: repo)
    monkeypatch.setattr(base, "repo_root", lambda: repo)
    config = {"benchmarks": {"eureqa": {"models": ["fake/missing"]}}}
    site = {"environment": {"hf_home": str(tmp_path / "hf")}}
    deferred = [{
        "benchmark": "eureqa",
        "model": "fake/missing",
        "dataset_split": "train",
        "phase": "deferred",
        "model_status": "deferred",
        "snapshot_path": "",
        "download_command": "huggingface-cli download fake/missing",
        "verification_command": "verify fake/missing",
    }]
    calls = []
    monkeypatch.setattr(base, "download_deferred_model", lambda _site, _model: False)
    monkeypatch.setattr(base, "run_pair", lambda *args, **kwargs: calls.append((args, kwargs)))

    failed = base.complete_deferred_models(config, site, root, deferred, allow_downloads=True)

    assert calls == []
    assert failed[0]["model_status"] == "failed_download"
    assert "deferred download failed benchmark=eureqa model=fake/missing" in (root / "LOCAL_COMMAND_LOG.md").read_text()


def test_write_deferred_records_preserves_failed_status(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    root = repo / "pre_experiments" / "1_frozen_diagnostic"
    root.mkdir(parents=True)
    monkeypatch.setattr(gd, "repo_root", lambda: repo)
    monkeypatch.setattr(base, "repo_root", lambda: repo)
    failed = [{
        "benchmark": "eureqa",
        "model": "fake/missing",
        "dataset_split": "train",
        "phase": "deferred",
        "model_status": "failed_download",
        "snapshot_path": "/tmp/snapshot",
        "download_command": "download old",
        "verification_command": "verify old",
    }]
    planned = [{
        "benchmark": "eureqa",
        "model": "fake/missing",
        "dataset_split": "train",
        "phase": "deferred",
        "model_status": "deferred",
        "snapshot_path": "/tmp/snapshot",
        "download_command": "download new",
        "verification_command": "verify new",
    }]

    base.write_deferred_records(root, failed)
    base.write_deferred_records(root, planned)

    rows = base.read_tsv(root / "deferred_models.tsv")
    assert rows[0]["model_status"] == "failed_download"
    assert rows[0]["download_command"] == "download new"


def test_download_deferred_model_missing_cli_returns_false(monkeypatch):
    def fake_run(*_args, **_kwargs):
        raise FileNotFoundError("huggingface-cli")

    monkeypatch.setattr(base.subprocess, "run", fake_run)

    assert base.download_deferred_model({"environment": {"hf_home": "/tmp/hf"}}, "fake/missing") is False
