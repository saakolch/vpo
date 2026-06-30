#!/usr/bin/env python3
"""Stage-gated launcher for reward-geometry benchmark pilots."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only in missing envs.
    raise SystemExit("PyYAML is required to read configs/*.yaml") from exc


METHODS = ("grpo", "maxrl", "max_at_k", "multi_rlvr", "vpo")
MULTI_METHODS = {"multi_rlvr", "vpo"}
TASK_PREPROCESS_SCRIPTS = {
    "maze": "data/preprocess_maze.py",
    "musique": "data/preprocess_musique.py",
    "eureqa": "data/preprocess_eureqa.py",
    "tool": "data/preprocess_tool.py",
}
TASK_EVAL_SCRIPTS = {
    "maze": "eval/eval_maze.py",
    "musique": "eval/eval_musique.py",
    "eureqa": "eval/eval_eureqa.py",
    "tool": "eval/eval_tool.py",
}
TASKS_WITH_VLLM_EVAL_ARGS = {"musique", "eureqa", "tool"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_bool(value: str | bool | None) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def require_pre_experiment_path(path: Path, allow_main_experiments: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if allow_main_experiments:
        return resolved
    root = (repo_root() / "pre_experiments").resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to write outside pre_experiments/: {resolved}. "
            "Pass --allow_main_experiments=true to override."
        ) from exc
    return resolved


def stage_slug(stage: dict[str, Any]) -> str:
    return f"stage_{int(stage['id'])}_{stage['name']}"


def stage_dir(config: dict[str, Any], stage: dict[str, Any], allow: bool = False) -> Path:
    root = repo_root() / config["experiment"]["output_root"]
    return require_pre_experiment_path(root / "stages" / stage_slug(stage), allow)


def status_path(config: dict[str, Any], stage: dict[str, Any]) -> Path:
    return stage_dir(config, stage, allow=False) / "status.json"


def load_status(config: dict[str, Any], stage: dict[str, Any]) -> str:
    path = status_path(config, stage)
    if path.exists():
        with path.open() as f:
            data = json.load(f)
        return str(data.get("status", "UNKNOWN")).upper()
    return str(stage.get("status", "LOCKED")).upper()


def find_stage(config: dict[str, Any], stage_id: int) -> dict[str, Any]:
    for stage in config.get("stages", []):
        if int(stage["id"]) == stage_id:
            return stage
    raise ValueError(f"No stage with id={stage_id}")


def previous_stages_pass(config: dict[str, Any], stage_id: int) -> tuple[bool, list[str]]:
    failures = []
    for stage in sorted(config.get("stages", []), key=lambda x: int(x["id"])):
        sid = int(stage["id"])
        if sid >= stage_id:
            break
        status = load_status(config, stage)
        if status != "PASS":
            failures.append(f"stage {sid} {stage['name']} is {status}")
    return not failures, failures


def comma_list(text: str | None, default: list[str]) -> list[str]:
    if not text:
        return default
    return [item.strip() for item in text.split(",") if item.strip()]


def seed_list(text: str | None, default: list[int]) -> list[int]:
    if not text:
        return default
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def env_prefix(env: dict[str, str]) -> str:
    return " ".join(f"{key}={shlex.quote(str(value))}" for key, value in env.items())


def method_eval_name(config: dict[str, Any], task: str, method: str) -> str:
    aliases = config.get("methods", {}).get("eval_aliases", {}).get(task, {})
    return str(aliases.get(method, method))


def num_solutions(stage: dict[str, Any], method: str) -> int:
    value = stage.get("eval", {}).get("num_solutions", {}).get(method)
    if value is not None:
        return int(value)
    return 3 if method in MULTI_METHODS else 1


def slurm_class(site: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any]:
    key = str(stage.get("partition_class", "smoke"))
    return dict(site.get("slurm", {}).get(key, {}))


def slurm_header(site: dict[str, Any], stage: dict[str, Any], job_name: str, out: Path, err: Path) -> str:
    klass = slurm_class(site, stage)
    account = site.get("slurm", {}).get("account")
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={klass.get('partition', 'debug')}",
        f"#SBATCH --gres={klass.get('gres', 'gpu:a40:1')}",
        f"#SBATCH --cpus-per-task={klass.get('cpus_per_task', 8)}",
        f"#SBATCH --mem={klass.get('mem', '64G')}",
        f"#SBATCH --time={klass.get('time', '00:30:00')}",
        f"#SBATCH --output={out}",
        f"#SBATCH --error={err}",
    ]
    if account:
        lines.insert(2, f"#SBATCH --account={account}")
    return "\n".join(lines)


def base_exports(site: dict[str, Any]) -> list[str]:
    env = site.get("environment", {})
    repo = repo_root()
    python_path = site.get("site", {}).get("python", ".venv/bin/python")
    site_pythonpath = str(env.get("pythonpath", "")).strip()
    pythonpath_suffix = f":{shlex.quote(site_pythonpath)}" if site_pythonpath else ""
    return [
        "set -euo pipefail",
        f"cd {shlex.quote(str(repo))}",
        f"export PATH={shlex.quote(str(repo / Path(python_path).parent))}:$PATH",
        f"export PYTHONPATH={shlex.quote(str(repo / 'verl'))}:{shlex.quote(str(repo))}{pythonpath_suffix}${{PYTHONPATH:+:$PYTHONPATH}}",
        f"export HF_HOME={shlex.quote(str(env.get('hf_home', repo / 'pre_experiments' / 'hf_cache')))}",
        f"export TRANSFORMERS_CACHE={shlex.quote(str(env.get('transformers_cache', '$HF_HOME/transformers')))}",
        f"export HF_HUB_CACHE={shlex.quote(str(env.get('hf_hub_cache', '$HF_HOME/hub')))}",
        f"export HF_HUB_OFFLINE={shlex.quote(str(env.get('hf_hub_offline', '1')))}",
        f"export TRANSFORMERS_OFFLINE={shlex.quote(str(env.get('transformers_offline', '1')))}",
        f"export WANDB_MODE={shlex.quote(str(env.get('wandb_mode', 'disabled')))}",
        "export WANDB_DISABLED=true",
        f"export TOKENIZERS_PARALLELISM={shlex.quote(str(env.get('tokenizers_parallelism', 'false')))}",
        "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy",
        "unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES",
    ]


def runtime_model_path(site: dict[str, Any], model_id: str) -> str:
    override = site.get("model_snapshots", {}).get(model_id)
    if override:
        path = Path(str(override)).expanduser()
        if not path.exists():
            raise ValueError(f"Configured model snapshot does not exist for {model_id}: {path}")
        return str(path)
    return model_id


def runtime_log_guard(stage_root: Path, allow_main_experiments: bool) -> list[str]:
    link_target = stage_root / "runtime_links" / "logs"
    lines = [
        f"LOG_LINK_TARGET={shlex.quote(str(link_target))}",
        f"mkdir -p {shlex.quote(str(link_target))}",
    ]
    if allow_main_experiments:
        lines.extend(
            [
                "if [[ -L logs ]]; then",
                "  CURRENT_LOG_TARGET=$(readlink logs)",
                "  if [[ \"$CURRENT_LOG_TARGET\" != \"$LOG_LINK_TARGET\" ]]; then",
                "    ln -sfn \"$LOG_LINK_TARGET\" logs",
                "  fi",
                "elif [[ ! -e logs ]]; then",
                "  ln -s \"$LOG_LINK_TARGET\" logs",
                "fi",
            ]
        )
    lines.extend(
        [
            "if [[ ! -L logs ]]; then",
            "  echo 'ERROR: train.sh writes to ./logs; refusing because ./logs is not a symlink into pre_experiments/first_experiment.' >&2",
            "  echo 'Regenerate with --allow_main_experiments=true to permit creating the top-level runtime symlink.' >&2",
            "  exit 2",
            "fi",
            "CURRENT_LOG_TARGET=$(readlink logs)",
            "if [[ \"$CURRENT_LOG_TARGET\" != \"$LOG_LINK_TARGET\" ]]; then",
            "  echo \"ERROR: ./logs points to $CURRENT_LOG_TARGET, expected $LOG_LINK_TARGET.\" >&2",
            "  echo 'Regenerate with --allow_main_experiments=true after confirming no previous-stage jobs are still running.' >&2",
            "  exit 2",
            "fi",
        ]
    )
    return lines


def add_optional_arg(parts: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            parts.append(flag)
        return
    text = str(value)
    if text.lower() in {"", "none", "null"}:
        return
    parts.extend([flag, text])


def preprocess_command(site: dict[str, Any], data_dir: Path, stage: dict[str, Any]) -> str:
    task = str(stage["task"])
    if task == "synthetic":
        return ""
    script = TASK_PREPROCESS_SCRIPTS.get(task)
    if not script:
        raise ValueError(f"No official preprocess script configured for task={task}")
    data = stage.get("data", {})
    py = site.get("site", {}).get("python", ".venv/bin/python")
    parts = [py, script, "--local_save_dir", str(data_dir)]
    if task == "maze":
        parts.extend([
            "--train_size", str(int(data.get("train_size", 1000))),
            "--test_size", str(int(data.get("test_size", 100))),
            "--train_seed", str(int(data.get("train_seed", 42))),
            "--test_seed", str(int(data.get("test_seed", 4242))),
        ])
    elif task == "musique":
        add_optional_arg(parts, "--max_train", data.get("max_train"))
        add_optional_arg(parts, "--max_test", data.get("max_test"))
    elif task == "eureqa":
        add_optional_arg(parts, "--mode", data.get("mode"))
        add_optional_arg(parts, "--train_split", data.get("train_split"))
        add_optional_arg(parts, "--test_split", data.get("test_split"))
        add_optional_arg(parts, "--mixed_split_seed", data.get("mixed_split_seed"))
        add_optional_arg(parts, "--max_train", data.get("max_train"))
        add_optional_arg(parts, "--max_test", data.get("max_test"))
    elif task == "tool":
        add_optional_arg(parts, "--source_dir", data.get("source_dir"))
    return shell_join(parts)


def eval_tensor_path(task: str, output_json: Path) -> Path:
    if task == "maze":
        return output_json.with_suffix(".npy")
    return output_json.with_name(f"{output_json.stem}_tensor.npy")


def extra_eval_artifact_paths(task: str, output_json: Path) -> list[Path]:
    if task == "musique":
        return [output_json.with_name(f"{output_json.stem}_weights.npy")]
    return []


def train_command(
    config: dict[str, Any],
    stage: dict[str, Any],
    method: str,
    seed: int,
    run_name: str,
    data_dir: Path,
    checkpoint_root: Path,
    allow_main_experiments: bool,
) -> str:
    task = str(stage["task"])
    training = stage.get("training", {})
    model_arg = runtime_model_path(config.get("_site", {}), str(stage["model"]))
    env = {
        "METHOD": method,
        "TASK": task,
        "MODEL": model_arg,
        "EPOCHS": str(int(training.get("epochs", 1))),
        "N_GPUS": str(int(training.get("n_gpus", 1))),
        "SEED": str(seed),
        "TAG": run_name.replace(f"{method}_{task}_seed{seed}_", ""),
        "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true",
    }
    if task == "maze":
        env["MAZE_DATA"] = str(data_dir)
    elif task == "musique":
        env["MUSIQUE_DATA"] = str(data_dir)
    elif task == "eureqa":
        env["EUREQA_DATA"] = str(data_dir)
    elif task == "tool":
        env["TOOL_DATA"] = str(data_dir)

    overrides = list(config.get("defaults", {}).get("train_overrides", []))
    overrides.extend(stage.get("training", {}).get("overrides", []))
    overrides.extend(stage.get("training", {}).get("method_overrides", {}).get(method, []))
    if "total_training_steps" in training:
        overrides.append(f"trainer.total_training_steps={int(training['total_training_steps'])}")
    if "save_freq" in training:
        overrides.append(f"trainer.save_freq={int(training['save_freq'])}")
    overrides.extend(
        [
            f"trainer.default_local_dir={checkpoint_root}",
            f"trainer.rollout_data_dir={checkpoint_root.parents[1] / 'rollouts' / run_name}",
            f"trainer.validation_data_dir={checkpoint_root.parents[1] / 'validation' / run_name}",
            f"global_profiler.save_path={checkpoint_root.parents[1] / 'profile' / run_name}",
        ]
    )
    overrides.append("trainer.resume_mode=disable")
    return f"{env_prefix(env)} bash train.sh {shell_join(overrides)}"


def eval_command(
    config: dict[str, Any],
    site: dict[str, Any],
    stage: dict[str, Any],
    method: str,
    seed: int,
    model_arg: str,
    data_dir: Path,
    output_json: Path,
) -> str:
    task = str(stage["task"])
    script = TASK_EVAL_SCRIPTS.get(task)
    if not script:
        raise ValueError(f"No official eval script configured for task={task}")
    eval_cfg = stage.get("eval", {})
    py = site.get("site", {}).get("python", ".venv/bin/python")
    parts = [
        py,
        script,
        "--model",
        model_arg,
        "--method",
        method_eval_name(config, task, method),
        "--data-dir",
        str(data_dir),
        "--output",
        str(output_json),
        "--n-chains",
        str(int(eval_cfg.get("n_chains", 10))),
        "--max-tokens",
        str(int(eval_cfg.get("max_tokens", 512))),
        "--temperature",
        str(float(eval_cfg.get("temperature", 0.7))),
        "--seed",
        str(seed),
        "--num-solutions",
        str(num_solutions(stage, method)),
    ]
    if task in TASKS_WITH_VLLM_EVAL_ARGS:
        add_optional_arg(parts, "--max-model-len", eval_cfg.get("max_model_len"))
        add_optional_arg(parts, "--tp", eval_cfg.get("tp"))
        add_optional_arg(parts, "--gpu-mem", eval_cfg.get("gpu_mem"))
        add_optional_arg(parts, "--enable-thinking", eval_cfg.get("enable_thinking"))
    if task == "musique":
        add_optional_arg(parts, "--num-examples", eval_cfg.get("num_examples"))
        add_optional_arg(parts, "--n-weights", eval_cfg.get("n_weights"))
        add_optional_arg(parts, "--mv-k-values", eval_cfg.get("mv_k_values"))
        add_optional_arg(parts, "--mv-n-subsets", eval_cfg.get("mv_n_subsets"))
    elif task in {"eureqa", "tool"}:
        add_optional_arg(parts, "--num-prompts", eval_cfg.get("num_prompts"))
    return shell_join(parts)


def diagnostics_command(site: dict[str, Any], tensor_path: Path, output_path: Path) -> str:
    py = site.get("site", {}).get("python", ".venv/bin/python")
    return shell_join([py, "scripts/geometry_diagnostics.py", "--tensor", str(tensor_path), "--output", str(output_path)])


def job_name(stage: dict[str, Any], method: str, seed: int, kind: str, attempt_label: str = "a1") -> str:
    attempt_suffix = "" if attempt_label == "a1" else f"_{attempt_label}"
    raw = f"geo_s{stage['id']}_{kind}_{method}_seed{seed}{attempt_suffix}"
    return re.sub(r"[^A-Za-z0-9_]+", "_", raw)[:128]


def build_job(
    config: dict[str, Any],
    site: dict[str, Any],
    stage: dict[str, Any],
    method: str,
    seed: int,
    kind: str,
    allow_main_experiments: bool,
) -> dict[str, Any]:
    root = stage_dir(config, stage, allow=False)
    task = str(stage["task"])
    attempt_label = str(stage.get("_attempt_label", "a1"))
    run_name = f"{method}_{task}_seed{seed}_first_experiment_stage{stage['id']}_{kind}_{attempt_label}"
    data_root = root / "data" / task
    eval_root = root / ("frozen_eval" if kind == "frozen_eval" else "eval")
    diag_root = root / "diagnostics"
    slurm_root = root / "slurm"
    ckpt_root = root / "checkpoints" / run_name
    expected_step = stage.get("training", {}).get("total_training_steps")
    if expected_step:
        ckpt_path = ckpt_root / f"global_step_{int(expected_step)}" / "actor" / "huggingface_merged"
    else:
        ckpt_path = ckpt_root / "LATEST_ACTOR_HF_MERGED"

    output_json = eval_root / f"{run_name}.json"
    tensor_path = eval_tensor_path(task, output_json)
    diag_path = diag_root / f"{run_name}_diagnostics.json"
    summary_path = root / "job_summaries" / f"{run_name}.json"

    train_cmd = ""
    model_arg = runtime_model_path(site, str(stage["model"]))
    if kind == "train_eval":
        train_cmd = train_command(
            config,
            stage,
            method,
            seed,
            run_name,
            data_root,
            ckpt_root,
            allow_main_experiments,
        )
        model_arg = str(ckpt_path)

    eval_cmd = eval_command(config, site, stage, method, seed, model_arg, data_root, output_json)
    diag_cmd = diagnostics_command(site, tensor_path, diag_path)
    name = job_name(stage, method, seed, kind, attempt_label)
    slurm_path = slurm_root / f"{name}.sbatch"
    out_path = root / "slurm_logs" / f"{name}_%j.out"
    err_path = root / "slurm_logs" / f"{name}_%j.err"

    return {
        "schema_version": 1,
        "stage_id": int(stage["id"]),
        "stage_name": str(stage["name"]),
        "job_kind": kind,
        "attempt_label": attempt_label,
        "benchmark": task,
        "method": method,
        "eval_method": method_eval_name(config, task, method),
        "model": str(stage["model"]),
        "runtime_model": model_arg,
        "seed": int(seed),
        "dataset_split": str(config.get("defaults", {}).get("dataset_split", "test")),
        "metric_provenance": "scripts/eval.py::official_eval_json_summary",
        "checkpoint_path": f"model://{stage['model']}" if kind == "frozen_eval" else str(ckpt_path),
        "train_command": train_cmd if train_cmd else "frozen_base_model_no_training",
        "eval_command": eval_cmd,
        "diagnostics_command": diag_cmd,
        "artifact_paths": [str(p) for p in [output_json, tensor_path, *extra_eval_artifact_paths(task, output_json), diag_path, summary_path]],
        "pre_training_diagnostics": "",
        "data_dir": str(data_root),
        "eval_output_path": str(output_json),
        "tensor_path": str(tensor_path),
        "diagnostics_path": str(diag_path),
        "job_summary_path": str(summary_path),
        "slurm_path": str(slurm_path),
        "slurm_stdout": str(out_path),
        "slurm_stderr": str(err_path),
        "estimated_gpu_hours": float(stage.get("estimated_gpu_hours", 0.0)),
        "created_at": now_iso(),
    }


def write_slurm(job: dict[str, Any], config: dict[str, Any], site: dict[str, Any], stage: dict[str, Any], allow_main_experiments: bool) -> None:
    path = require_pre_experiment_path(Path(job["slurm_path"]), allow_main_experiments=False)
    root = stage_dir(config, stage, allow=False)
    for key in ("slurm_stdout", "slurm_stderr"):
        require_pre_experiment_path(Path(job[key]), allow_main_experiments=False).parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(job["eval_output_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(job["diagnostics_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(job["job_summary_path"]).parent.mkdir(parents=True, exist_ok=True)

    lines = [
        slurm_header(
            site,
            stage,
            job_name(stage, job["method"], job["seed"], job["job_kind"], job.get("attempt_label", "a1")),
            Path(job["slurm_stdout"]),
            Path(job["slurm_stderr"]),
        ),
        "",
        *base_exports(site),
        f"mkdir -p {shlex.quote(str(Path(job['data_dir'])))}",
    ]
    pre_cmd = preprocess_command(site, Path(job["data_dir"]), stage)
    if pre_cmd:
        lines.append(pre_cmd)
    if job["job_kind"] == "train_eval":
        py = site.get("site", {}).get("python", ".venv/bin/python")
        lines.extend(runtime_log_guard(root, allow_main_experiments))
        lines.append(job["train_command"])
        run_name = f"{job['method']}_{job['benchmark']}_seed{job['seed']}_first_experiment_stage{job['stage_id']}_{job['job_kind']}_{job.get('attempt_label', 'a1')}"
        checkpoint_root = root / "checkpoints" / run_name
        expected_model = str(job["checkpoint_path"])
        lines.extend(
            [
                f"CKPT_ROOT={shlex.quote(str(checkpoint_root))}",
                "LAST_STEP=$(find \"$CKPT_ROOT\" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -n 1 || true)",
                "if [[ -z \"$LAST_STEP\" || ! -d \"$LAST_STEP/actor\" ]]; then",
                "  echo \"ERROR: no actor checkpoint found under $CKPT_ROOT\" >&2",
                "  exit 3",
                "fi",
                "MERGED_DIR=\"$LAST_STEP/actor/huggingface_merged\"",
                f"{shlex.quote(str(py))} -m verl.model_merger merge --backend fsdp --local_dir \"$LAST_STEP/actor\" --target_dir \"$MERGED_DIR\"",
                f"EXPECTED_MODEL={shlex.quote(expected_model)}",
                f"EVAL_CMD={shlex.quote(str(job['eval_command']))}",
                "if [[ \"$EXPECTED_MODEL\" == *LATEST_ACTOR_HF_MERGED ]]; then",
                "  EVAL_CMD=\"${EVAL_CMD//$EXPECTED_MODEL/$MERGED_DIR}\"",
                "fi",
                "eval \"$EVAL_CMD\"",
            ]
        )
    else:
        lines.append(job["eval_command"])
    lines.extend(
        [
            f"if [[ -f {shlex.quote(str(job['tensor_path']))} ]]; then",
            f"  {job['diagnostics_command']}",
            "fi",
            "cat > " + shlex.quote(str(job["job_summary_path"])) + " <<'JSON'",
            json.dumps({k: v for k, v in job.items() if k != "artifact_paths"}, indent=2, sort_keys=True),
            "JSON",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_manifest(path: Path, jobs: list[dict[str, Any]], allow: bool) -> None:
    out = require_pre_experiment_path(path, allow)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema_version": 1, "created_at": now_iso(), "jobs": jobs}, indent=2, sort_keys=True) + "\n")


def job_key(job: dict[str, Any]) -> tuple[int, str, str, int, str]:
    return (
        int(job["stage_id"]),
        str(job["job_kind"]),
        str(job["method"]),
        int(job["seed"]),
        str(job.get("attempt_label", "a1")),
    )


def merge_existing_submission_state(path: Path, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return jobs
    try:
        existing = json.loads(path.read_text()).get("jobs", [])
    except Exception:
        return jobs
    by_key = {
        job_key(job): job
        for job in existing
        if isinstance(job, dict)
        and "stage_id" in job
        and "job_kind" in job
        and "method" in job
        and "seed" in job
    }
    merged = []
    for job in jobs:
        prior = by_key.get(job_key(job), {})
        for field in ("slurm_id", "submitted_at", "submission_status", "submission_error"):
            if prior.get(field):
                job[field] = prior[field]
        merged.append(job)
    return merged


def submit_job(path: Path) -> str:
    result = subprocess.run(["sbatch", str(path)], capture_output=True, text=True)
    if result.returncode != 0:
        message = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
        raise RuntimeError(message or f"sbatch failed with return code {result.returncode}")
    text = result.stdout.strip()
    match = re.search(r"Submitted batch job\s+(\d+)", text)
    if not match:
        raise RuntimeError(f"Could not parse sbatch output: {text}")
    return match.group(1)


def append_command_log(config: dict[str, Any], line: str) -> None:
    path = repo_root() / config["experiment"].get("command_log", "pre_experiments/first_experiment/LOCAL_COMMAND_LOG.md")
    require_pre_experiment_path(path, False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(line.rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geometry/maze_pilot.yaml")
    parser.add_argument("--site-config", default="configs/site/hpc2.yaml")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--methods", default=None, help="Comma-separated method list.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list.")
    parser.add_argument("--job-kind", choices=["frozen_eval", "train_eval", "all"], default="all")
    parser.add_argument("--attempt-label", default="a1", help="Attempt suffix used to preserve failed run artifacts.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--generate-slurm", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--allow-high-gpu-hours", action="store_true")
    parser.add_argument("--allow-deferred-methods", action="store_true")
    parser.add_argument("--allow_main_experiments", nargs="?", const=True, default=False, type=parse_bool)
    args = parser.parse_args()

    config = load_yaml(repo_root() / args.config)
    site = load_yaml(repo_root() / args.site_config)
    config["_site"] = site
    stage = find_stage(config, args.stage)
    stage["_attempt_label"] = args.attempt_label
    if stage.get("enabled") is False and not args.allow_deferred_methods:
        raise SystemExit(f"Stage {args.stage} is disabled/deferred: {stage.get('deferred_reason', 'no reason recorded')}")

    ok, failures = previous_stages_pass(config, int(stage["id"]))
    if args.submit and not ok:
        raise SystemExit("Refusing to submit locked stage; previous stages are not PASS: " + "; ".join(failures))

    max_hours = float(config.get("safety", {}).get("max_gpu_hours_without_override", 4))
    if args.submit and float(stage.get("estimated_gpu_hours", 0)) > max_hours and not args.allow_high_gpu_hours:
        raise SystemExit(
            f"Refusing to submit stage estimated at {stage.get('estimated_gpu_hours')} GPU-hours "
            f"without --allow-high-gpu-hours."
        )

    default_methods = list(config.get("methods", {}).get("active", METHODS))
    methods = comma_list(args.methods, default_methods)
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown methods: {', '.join(unknown)}")
    seeds = seed_list(args.seeds, [int(x) for x in stage.get("seeds", config.get("defaults", {}).get("seeds", [0]))])
    kinds = ["frozen_eval", "train_eval"] if args.job_kind == "all" else [args.job_kind]
    if int(stage["id"]) != 1:
        kinds = [kind for kind in kinds if kind != "frozen_eval"]
    if str(stage["task"]) == "synthetic":
        kinds = []

    jobs = [
        build_job(config, site, stage, method, seed, kind, args.allow_main_experiments)
        for kind in kinds
        for method in methods
        for seed in seeds
    ]
    suffix = "" if args.attempt_label == "a1" else f"_{args.attempt_label}"
    manifest_path = stage_dir(config, stage, allow=False) / "manifests" / f"stage_{stage['id']}_{stage['name']}_jobs{suffix}.json"
    jobs = merge_existing_submission_state(manifest_path, jobs)

    print(json.dumps({"stage": stage["name"], "jobs": len(jobs), "submit": args.submit, "dry_run": args.dry_run}, indent=2))
    if args.dry_run and not args.generate_slurm:
        for job in jobs:
            print(f"DRY-RUN {job['job_kind']} {job['method']} seed={job['seed']}: {job['slurm_path']}")
        return

    write_manifest(manifest_path, jobs, allow=False)
    for job in jobs:
        write_slurm(job, config, site, stage, args.allow_main_experiments)

    if args.submit:
        submitted = []
        failures = []
        for job in jobs:
            if job.get("slurm_id"):
                continue
            try:
                slurm_id = submit_job(Path(job["slurm_path"]))
            except Exception as exc:
                job["submission_status"] = "FAILED_TO_SUBMIT"
                job["submission_error"] = str(exc)
                failures.append(job)
                append_command_log(config, f"- {now_iso()} failed_submit stage={stage['id']} job={job['job_kind']} method={job['method']} seed={job['seed']} error={str(exc)!r} script={job['slurm_path']}")
                write_manifest(manifest_path, jobs, allow=False)
                continue
            job["slurm_id"] = slurm_id
            job["submitted_at"] = now_iso()
            job["submission_status"] = "SUBMITTED"
            job.pop("submission_error", None)
            submitted.append(slurm_id)
            append_command_log(config, f"- {now_iso()} submitted stage={stage['id']} job={job['job_kind']} method={job['method']} seed={job['seed']} slurm_id={slurm_id} script={job['slurm_path']}")
            write_manifest(manifest_path, jobs, allow=False)
        write_manifest(manifest_path, jobs, allow=False)
        if submitted:
            print("Submitted Slurm jobs: " + ", ".join(submitted))
        if failures:
            print(f"Failed submissions: {len(failures)}")
            raise SystemExit(1)
    else:
        append_command_log(config, f"- {now_iso()} generated stage={stage['id']} slurm manifest={manifest_path} jobs={len(jobs)} submit=false")
        print(f"Generated manifest: {manifest_path}")


if __name__ == "__main__":
    main()
