from __future__ import annotations

from pathlib import Path

from scripts import short_training_common as common
from scripts.short_training_common import train_segment_command


def test_train_segment_command_uses_resume_path_after_first_segment(tmp_path):
    config = {"defaults": {"train_overrides": []}}
    benchmark = {
        "name": "maze",
        "model": "Qwen/Qwen3-0.6B",
        "training": {"epochs": 1, "n_gpus": 1},
    }
    site = {"model_snapshots": {}}
    checkpoint_root = tmp_path / "ckpts"

    first = train_segment_command(config, benchmark, "grpo", 0, "grpo_maze_seed0_short_training_a1", tmp_path / "data", checkpoint_root, 1, None, site)
    resumed = train_segment_command(config, benchmark, "grpo", 0, "grpo_maze_seed0_short_training_a1", tmp_path / "data", checkpoint_root, 4, 1, site)

    assert "trainer.total_training_steps=1" in first
    assert "trainer.resume_mode=disable" in first
    assert "trainer.total_training_steps=4" in resumed
    assert "trainer.resume_mode=resume_path" in resumed
    assert f"trainer.resume_from_path={checkpoint_root / 'global_step_1'}" in resumed


def test_train_segment_command_keeps_method_overrides_after_common_overrides(tmp_path):
    config = {"defaults": {"train_overrides": []}}
    benchmark = {
        "name": "maze",
        "model": "Qwen/Qwen3-0.6B",
        "training": {
            "epochs": 1,
            "n_gpus": 1,
            "overrides": ["++actor_rollout_ref.rollout.n=2"],
            "method_overrides": {
                "max_at_k": [
                    "actor_rollout_ref.rollout.n=3",
                    "actor_rollout_ref.rollout.agent.num_workers=4",
                ]
            },
        },
    }
    site = {"model_snapshots": {}}
    checkpoint_root = tmp_path / "ckpts"

    command = train_segment_command(
        config,
        benchmark,
        "max_at_k",
        0,
        "max_at_k_maze_seed0_short_training_a1",
        tmp_path / "data",
        checkpoint_root,
        1,
        None,
        site,
    )

    assert command.index("++actor_rollout_ref.rollout.n=2") < command.index("actor_rollout_ref.rollout.n=3")
    assert "actor_rollout_ref.rollout.agent.num_workers=4" in command


def test_site_environment_prepends_repo_verl_and_site_pythonpath(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "repo_root", lambda: tmp_path)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("PYTHONPATH", "/already")

    env = common.site_environment(
        {
            "site": {"python": str(tmp_path / "py" / "bin" / "python")},
            "environment": {"pythonpath": "/extra:/already"},
        }
    )

    pythonpath = env["PYTHONPATH"].split(":")
    assert env["PATH"].startswith(str(tmp_path / "py" / "bin"))
    assert pythonpath[:4] == [str(tmp_path / "verl"), str(tmp_path), "/extra", "/already"]
    assert pythonpath.count("/already") == 1


def test_wait_for_gpu_quiescence_waits_for_empty_compute_apps(tmp_path, monkeypatch):
    outputs = iter(["123, 4096\n", "\n"])

    class Result:
        returncode = 0

        def __init__(self, stdout: str):
            self.stdout = stdout

    def fake_run(*args, **kwargs):
        return Result(next(outputs))

    monkeypatch.setattr(common.subprocess, "run", fake_run)
    monkeypatch.setattr(common.time, "sleep", lambda _: None)
    log_path = tmp_path / "gpu_settle.log"

    assert common.wait_for_gpu_quiescence(log_path, timeout_s=5, interval_s=0, stable_checks=1)
    text = log_path.read_text()
    assert "apps=[(123, 4096)]" in text
    assert "apps=[]" in text
