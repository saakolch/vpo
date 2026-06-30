#!/usr/bin/env python3
"""Shared helpers for Stage 2 short-training diagnostics."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

T = TypeVar("T")

METHODS = ("grpo", "maxrl", "max_at_k", "multi_rlvr", "vpo")
MULTI_METHODS = {"multi_rlvr", "vpo"}
TASK_EVAL_SCRIPTS = {
    "maze": "eval/eval_maze.py",
    "musique": "eval/eval_musique.py",
    "tool": "eval/eval_tool.py",
}
TASK_PREPROCESS_SCRIPTS = {
    "maze": "data/preprocess_maze.py",
    "musique": "data/preprocess_musique.py",
    "tool": "data/preprocess_tool.py",
}
TASK_DATA_ENV = {
    "maze": "MAZE_DATA",
    "musique": "MUSIQUE_DATA",
    "tool": "TOOL_DATA",
}
TASKS_WITH_VLLM_EVAL_ARGS = {"musique", "tool"}
STAGE2_ROOT = "pre_experiments/2_short_training"


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
    text = path.read_text()
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = simple_yaml_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:idx].rstrip()
    return line.rstrip()


def _parse_scalar(text: str) -> Any:
    raw = text.strip()
    if raw == "":
        return ""
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    lowered = raw.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def simple_yaml_load(text: str) -> Any:
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = _strip_comment(raw)
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent, stripped.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index
        if lines[index][1].startswith("- "):
            out = []
            while index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("- "):
                item = lines[index][1][2:].strip()
                index += 1
                if item:
                    out.append(_parse_scalar(item))
                else:
                    value, index = parse_block(index, indent + 2)
                    out.append(value)
            return out, index
        out: dict[str, Any] = {}
        while index < len(lines) and lines[index][0] == indent and not lines[index][1].startswith("- "):
            text_line = lines[index][1]
            if ":" not in text_line:
                raise ValueError(f"Cannot parse YAML line: {text_line}")
            key, raw_value = text_line.split(":", 1)
            index += 1
            if raw_value.strip():
                out[key.strip()] = _parse_scalar(raw_value)
            else:
                value, index = parse_block(index, indent + 2)
                out[key.strip()] = value
        return out, index

    value, final = parse_block(0, lines[0][0] if lines else 0)
    if final != len(lines):
        raise ValueError("Could not parse full YAML document.")
    return value


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: Any, allow_main_experiments: bool = False) -> None:
    out = require_stage2_path(path, allow_main_experiments)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def require_stage2_path(path: Path, allow_main_experiments: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if allow_main_experiments:
        return resolved
    root = (repo_root() / STAGE2_ROOT).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to write outside {STAGE2_ROOT}/: {resolved}. "
            "Pass --allow_main_experiments=true to override."
        ) from exc
    return resolved


def rel_repo(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def shell_join(parts: Iterable[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def env_prefix(env: dict[str, Any]) -> str:
    return " ".join(f"{key}={shlex.quote(str(value))}" for key, value in env.items())


def comma_list(text: str | None, default: list[str]) -> list[str]:
    if not text:
        return default
    return [item.strip() for item in text.split(",") if item.strip()]


def int_list(text: str | None, default: list[int]) -> list[int]:
    if not text:
        return default
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def progress(items: Iterable[T], desc: str = "progress") -> Iterator[T]:
    try:
        from tqdm import tqdm

        yield from tqdm(items, desc=desc)
    except Exception:  # pragma: no cover - fallback for minimal envs.
        seq = list(items)
        total = len(seq)
        for idx, item in enumerate(seq, start=1):
            print(f"{desc}: {idx}/{total}", flush=True)
            yield item


def runtime_model_path(site: dict[str, Any], model_id: str) -> str:
    override = site.get("model_snapshots", {}).get(model_id)
    if override:
        path = Path(str(override)).expanduser()
        if not path.exists():
            raise ValueError(f"Configured model snapshot does not exist for {model_id}: {path}")
        return str(path)
    return model_id


def method_eval_name(config: dict[str, Any], task: str, method: str) -> str:
    aliases = config.get("methods", {}).get("eval_aliases", {}).get(task, {})
    return str(aliases.get(method, method))


def num_solutions(method: str, eval_cfg: dict[str, Any]) -> int:
    per_method = eval_cfg.get("num_solutions", {})
    if isinstance(per_method, dict) and method in per_method:
        return int(per_method[method])
    return 3 if method in MULTI_METHODS else 1


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


def slurm_class(site: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    key = str(benchmark.get("partition_class", "smoke"))
    return dict(site.get("slurm", {}).get(key, {}))


def slurm_header(site: dict[str, Any], benchmark: dict[str, Any], job_name: str, out: Path, err: Path) -> str:
    klass = slurm_class(site, benchmark)
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
    account = site.get("slurm", {}).get("account")
    if account:
        lines.insert(2, f"#SBATCH --account={account}")
    return "\n".join(lines)


def site_exports(site: dict[str, Any]) -> list[str]:
    repo = repo_root()
    env = site.get("environment", {})
    python_path = str(site.get("site", {}).get("python", ".venv/bin/python"))
    python_parent = Path(python_path).expanduser().parent
    site_pythonpath = str(env.get("pythonpath", "")).strip()
    suffix = f":{shlex.quote(site_pythonpath)}" if site_pythonpath else ""
    return [
        "set -euo pipefail",
        f"cd {shlex.quote(str(repo))}",
        f"export PATH={shlex.quote(str(python_parent))}:$PATH",
        f"export PYTHONPATH={shlex.quote(str(repo / 'verl'))}:{shlex.quote(str(repo))}{suffix}${{PYTHONPATH:+:$PYTHONPATH}}",
        f"export HF_HOME={shlex.quote(str(env.get('hf_home', repo / STAGE2_ROOT / 'hf_cache')))}",
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


def site_environment(site: dict[str, Any]) -> dict[str, str]:
    repo = repo_root()
    env = site.get("environment", {})
    python_path = str(site.get("site", {}).get("python", ".venv/bin/python"))
    python_parent = Path(python_path).expanduser().parent
    pythonpath_entries = [str(repo / "verl"), str(repo)]
    site_pythonpath = str(env.get("pythonpath", "")).strip()
    if site_pythonpath:
        pythonpath_entries.extend(part for part in site_pythonpath.split(":") if part)
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if current_pythonpath:
        pythonpath_entries.extend(part for part in current_pythonpath.split(":") if part)

    deduped_pythonpath: list[str] = []
    for entry in pythonpath_entries:
        if entry not in deduped_pythonpath:
            deduped_pythonpath.append(entry)

    return {
        "PATH": f"{python_parent}:{os.environ.get('PATH', '')}",
        "PYTHONPATH": ":".join(deduped_pythonpath),
        "HF_HOME": str(env.get("hf_home", repo / STAGE2_ROOT / "hf_cache")),
        "TRANSFORMERS_CACHE": str(env.get("transformers_cache", "$HF_HOME/transformers")),
        "HF_HUB_CACHE": str(env.get("hf_hub_cache", "$HF_HOME/hub")),
        "HF_HUB_OFFLINE": str(env.get("hf_hub_offline", "1")),
        "TRANSFORMERS_OFFLINE": str(env.get("transformers_offline", "1")),
        "WANDB_MODE": str(env.get("wandb_mode", "disabled")),
        "WANDB_DISABLED": "true",
        "TOKENIZERS_PARALLELISM": str(env.get("tokenizers_parallelism", "false")),
    }


def apply_site_environment(site: dict[str, Any]) -> None:
    os.environ.update(site_environment(site))
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
    ):
        os.environ.pop(key, None)


def _query_gpu_compute_apps() -> list[tuple[int, int]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    apps: list[tuple[int, int]] = []
    for raw in result.stdout.splitlines():
        pieces = [part.strip() for part in raw.split(",")]
        if len(pieces) < 2 or not pieces[0]:
            continue
        try:
            apps.append((int(pieces[0]), int(float(pieces[1]))))
        except ValueError:
            continue
    return apps


def wait_for_gpu_quiescence(
    log_path: Path | None = None,
    *,
    timeout_s: float = 180.0,
    interval_s: float = 5.0,
    stable_checks: int = 2,
    desc: str = "gpu settle",
) -> bool:
    start = time.time()
    stable = 0
    max_checks = max(1, int(timeout_s / max(interval_s, 1.0)) + 1)
    for _ in progress(range(max_checks), desc=desc):
        apps = _query_gpu_compute_apps()
        total_mem = sum(mem for _, mem in apps)
        line = f"[{now_iso()}] GPU settle apps={apps} total_mem_mib={total_mem}"
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as f:
                f.write(line + "\n")
        else:
            print(line, flush=True)
        if not apps:
            stable += 1
            if stable >= stable_checks:
                return True
        else:
            stable = 0
        if time.time() - start >= timeout_s:
            break
        time.sleep(interval_s)
    return False


def preprocess_command(site: dict[str, Any], benchmark: dict[str, Any], data_dir: Path) -> str:
    task = str(benchmark["name"])
    script = TASK_PREPROCESS_SCRIPTS.get(task)
    if script is None:
        raise ValueError(f"No preprocess script for benchmark={task}")
    data = benchmark.get("data", {})
    py = site.get("site", {}).get("python", ".venv/bin/python")
    parts = [py, script, "--local_save_dir", str(data_dir)]
    if task == "maze":
        parts.extend(
            [
                "--train_size",
                str(int(data.get("train_size", 1000))),
                "--test_size",
                str(int(data.get("test_size", 100))),
                "--train_seed",
                str(int(data.get("train_seed", 42))),
                "--test_seed",
                str(int(data.get("test_seed", 4242))),
            ]
        )
    elif task == "musique":
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


def eval_command(
    config: dict[str, Any],
    site: dict[str, Any],
    benchmark: dict[str, Any],
    method: str,
    seed: int,
    model_arg: str,
    data_dir: Path,
    output_json: Path,
) -> str:
    task = str(benchmark["name"])
    script = TASK_EVAL_SCRIPTS.get(task)
    if script is None:
        raise ValueError(f"No eval script for benchmark={task}")
    eval_cfg = benchmark.get("eval", {})
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
        str(num_solutions(method, eval_cfg)),
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
    elif task == "tool":
        add_optional_arg(parts, "--num-prompts", eval_cfg.get("num_prompts"))
    return shell_join(parts)


def diagnostics_command(site: dict[str, Any], tensor_path: Path, output_path: Path) -> str:
    py = site.get("site", {}).get("python", ".venv/bin/python")
    return shell_join([py, "scripts/geometry_diagnostics.py", "--tensor", str(tensor_path), "--output", str(output_path)])


def train_segment_command(
    config: dict[str, Any],
    benchmark: dict[str, Any],
    method: str,
    seed: int,
    run_name: str,
    data_dir: Path,
    checkpoint_root: Path,
    target_update: int,
    previous_update: int | None,
    site: dict[str, Any],
) -> str:
    task = str(benchmark["name"])
    training = benchmark.get("training", {})
    env = {
        "METHOD": method,
        "TASK": task,
        "MODEL": runtime_model_path(site, str(benchmark["model"])),
        "EPOCHS": str(int(training.get("epochs", 1))),
        "N_GPUS": str(int(training.get("n_gpus", 1))),
        "SEED": str(seed),
        "TAG": run_name.replace(f"{method}_{task}_seed{seed}_", ""),
        "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true",
    }
    data_env = TASK_DATA_ENV.get(task)
    if data_env:
        env[data_env] = str(data_dir)

    overrides = list(config.get("defaults", {}).get("train_overrides", []))
    overrides.extend(training.get("overrides", []))
    overrides.extend(training.get("method_overrides", {}).get(method, []))
    overrides.extend(
        [
            f"trainer.total_training_steps={int(target_update)}",
            f"trainer.save_freq={int(target_update)}",
            "trainer.test_freq=-1",
            f"trainer.default_local_dir={checkpoint_root}",
            f"trainer.rollout_data_dir={checkpoint_root.parents[1] / 'rollouts' / run_name}",
            f"trainer.validation_data_dir={checkpoint_root.parents[1] / 'validation' / run_name}",
            f"global_profiler.save_path={checkpoint_root.parents[1] / 'profile' / run_name}",
        ]
    )
    if previous_update and previous_update > 0:
        resume_path = checkpoint_root / f"global_step_{int(previous_update)}"
        overrides.extend(["trainer.resume_mode=resume_path", f"trainer.resume_from_path={resume_path}"])
    else:
        overrides.append("trainer.resume_mode=disable")
    return f"{env_prefix(env)} bash train.sh {shell_join(overrides)}"


def merge_command(site: dict[str, Any], actor_dir: Path, target_dir: Path) -> str:
    py = site.get("site", {}).get("python", ".venv/bin/python")
    repo = repo_root()
    site_pythonpath = str(site.get("environment", {}).get("pythonpath", "")).strip()
    pythonpath_parts = [str(repo / "verl"), str(repo)]
    if site_pythonpath:
        pythonpath_parts.append(site_pythonpath)
    command = shell_join(
        [
            py,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(actor_dir),
            "--target_dir",
            str(target_dir),
        ]
    )
    return f"PYTHONPATH={shlex.quote(':'.join(pythonpath_parts))}${{PYTHONPATH:+:$PYTHONPATH}} {command}"


def run_command(command: str, log_path: Path | None = None) -> int:
    start = time.time()
    print(f"[{now_iso()}] START {command}", flush=True)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(f"\n[{now_iso()}] START {command}\n")
            proc = subprocess.run(command, shell=True, executable="/bin/bash", stdout=f, stderr=subprocess.STDOUT)
            f.write(f"[{now_iso()}] END returncode={proc.returncode} elapsed_s={time.time() - start:.1f}\n")
    else:
        proc = subprocess.run(command, shell=True, executable="/bin/bash")
    elapsed = time.time() - start
    print(f"[{now_iso()}] END returncode={proc.returncode} elapsed_s={elapsed:.1f}", flush=True)
    return int(proc.returncode)


def submit_sbatch(path: Path) -> str:
    result = subprocess.run(["sbatch", str(path)], capture_output=True, text=True)
    if result.returncode != 0:
        message = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
        raise RuntimeError(message or f"sbatch failed with return code {result.returncode}")
    text = result.stdout.strip()
    pieces = text.split()
    if pieces and pieces[-1].isdigit():
        return pieces[-1]
    raise RuntimeError(f"Could not parse sbatch output: {text}")


def append_line(path: Path, line: str, allow_main_experiments: bool = False) -> None:
    out = require_stage2_path(path, allow_main_experiments)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(line.rstrip() + "\n")
