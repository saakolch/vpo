from __future__ import annotations

from pathlib import Path

from scripts import launch_geometry_grid as launcher


def make_config(stage: dict) -> tuple[dict, dict]:
    site = {"site": {"python": ".venv/bin/python"}, "slurm": {"official": {}}}
    config = {
        "experiment": {
            "output_root": "pre_experiments/first_experiment",
            "command_log": "pre_experiments/first_experiment/LOCAL_COMMAND_LOG.md",
        },
        "defaults": {"dataset_split": "test", "train_overrides": []},
        "methods": {
            "eval_aliases": {
                "musique": {"max_at_k": "maxrl"},
                "eureqa": {"max_at_k": "maxrl"},
            }
        },
        "stages": [stage],
        "_site": site,
    }
    return config, site


def test_launcher_writes_musique_official_preprocess_and_eval(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "repo_root", lambda: tmp_path)
    stage = {
        "id": 3,
        "name": "musique_qwen3_1_7b",
        "task": "musique",
        "model": "Qwen/Qwen3-1.7B",
        "data": {"max_train": -1, "max_test": -1},
        "training": {"epochs": 1, "n_gpus": 4},
        "eval": {"n_chains": 10, "num_examples": 300, "max_tokens": 512, "temperature": 0.7, "max_model_len": 6400},
    }
    config, site = make_config(stage)

    job = launcher.build_job(config, site, stage, "max_at_k", 0, "train_eval", True)
    launcher.write_slurm(job, config, site, stage, True)

    slurm = Path(job["slurm_path"]).read_text()
    assert "data/preprocess_musique.py" in slurm
    assert "--max_train -1" in slurm
    assert "eval/eval_musique.py" in job["eval_command"]
    assert "--method maxrl" in job["eval_command"]
    assert "--num-examples 300" in job["eval_command"]
    assert "--max-model-len 6400" in job["eval_command"]
    assert any(path.endswith("_weights.npy") for path in job["artifact_paths"])


def test_launcher_writes_eureqa_mixed_split_and_task_tensor_path(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "repo_root", lambda: tmp_path)
    stage = {
        "id": 4,
        "name": "eureqa_qwen3_8b",
        "task": "eureqa",
        "model": "Qwen/Qwen3-8B",
        "data": {"mode": "mixed_split", "mixed_split_seed": 0, "max_train": -1, "max_test": -1},
        "training": {"epochs": 6, "n_gpus": 4},
        "eval": {"n_chains": 10, "num_prompts": 8, "max_tokens": 2048, "temperature": 0.7, "max_model_len": 4096},
    }
    config, site = make_config(stage)

    pre_cmd = launcher.preprocess_command(site, tmp_path / "data", stage)
    job = launcher.build_job(config, site, stage, "vpo", 0, "train_eval", True)

    assert "data/preprocess_eureqa.py" in pre_cmd
    assert "--mode mixed_split" in pre_cmd
    assert "--mixed_split_seed 0" in pre_cmd
    assert "eval/eval_eureqa.py" in job["eval_command"]
    assert "--num-prompts 8" in job["eval_command"]
    assert job["tensor_path"].endswith("_tensor.npy")


def test_launcher_writes_tool_official_eval_options(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "repo_root", lambda: tmp_path)
    stage = {
        "id": 5,
        "name": "toolrl_qwen3_1_7b",
        "task": "tool",
        "model": "Qwen/Qwen3-1.7B",
        "data": {"source_dir": None},
        "training": {"epochs": 20, "n_gpus": 4},
        "eval": {"n_chains": 10, "num_prompts": 80, "max_tokens": 2048, "temperature": 0.7, "max_model_len": 6144},
    }
    config, site = make_config(stage)

    job = launcher.build_job(config, site, stage, "multi_rlvr", 0, "train_eval", True)

    assert "eval/eval_tool.py" in job["eval_command"]
    assert "--num-prompts 80" in job["eval_command"]
    assert "--num-solutions 3" in job["eval_command"]
    assert "--max-model-len 6144" in job["eval_command"]
    assert job["tensor_path"].endswith("_tensor.npy")


def test_full_grid_aliases_max_at_k_for_eval_clis():
    config = launcher.load_yaml(Path("configs/geometry/full_benchmark_grid.yaml"))

    assert launcher.method_eval_name(config, "maze", "max_at_k") == "maxrl"
    assert launcher.method_eval_name(config, "musique", "max_at_k") == "maxrl"
    assert launcher.method_eval_name(config, "eureqa", "max_at_k") == "maxrl"


def test_alternate_attempt_label_gets_distinct_slurm_path(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "repo_root", lambda: tmp_path)
    stage = {
        "id": 2,
        "name": "maze_official_qwen3_4b",
        "task": "maze",
        "model": "Qwen/Qwen3-4B",
        "data": {"train_size": 1000, "test_size": 100},
        "training": {"epochs": 50, "n_gpus": 2},
        "eval": {"n_chains": 10, "max_tokens": 1024, "temperature": 0.7},
        "_attempt_label": "seed0_parallel_2gpu_a1",
    }
    config, site = make_config(stage)

    job = launcher.build_job(config, site, stage, "maxrl", 0, "train_eval", True)

    assert "seed0_parallel_2gpu_a1" in job["slurm_path"]
    assert "N_GPUS=2" in job["train_command"]


def test_launcher_uses_site_python_for_merge_and_pythonpath(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "repo_root", lambda: tmp_path)
    stage = {
        "id": 2,
        "name": "maze_official_qwen3_4b",
        "task": "maze",
        "model": "Qwen/Qwen3-4B",
        "data": {"train_size": 1000, "test_size": 100},
        "training": {"epochs": 50, "n_gpus": 2},
        "eval": {"n_chains": 10, "max_tokens": 1024, "temperature": 0.7},
    }
    config, site = make_config(stage)
    site["site"]["python"] = "/opt/container/python3.11"
    site["environment"] = {"pythonpath": "/workspace/.venv/lib/python3.11/site-packages"}

    job = launcher.build_job(config, site, stage, "grpo", 0, "train_eval", True)
    launcher.write_slurm(job, config, site, stage, True)

    slurm = Path(job["slurm_path"]).read_text()
    assert "/opt/container/python3.11 -m verl.model_merger" in slurm
    assert "/workspace/.venv/lib/python3.11/site-packages" in slurm
