#!/usr/bin/env python3
"""Frozen base-model reward-geometry diagnostic pipeline.

This script deliberately never calls ``train.sh``. It plans and runs no-RL
base-model sampling/evaluation jobs, writes partial diagnostic stages, and
tracks deferred model downloads separately from available-model runs.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required to read configs/*.yaml") from exc

from scripts.assign_geometry_strata import assign_strata
from scripts.geometry_diagnostics import (
    compute_diagnostics,
    parse_bool,
    require_pre_experiment_path,
    rows_from_diagnostics,
    write_tsv,
)
from scripts.plot_metrics import generate_plots
from scripts.progress_monitor import checkpoint_dir_name, crossed_checkpoints, write_checkpoint_marker


TASK_PREPROCESS_SCRIPTS = {
    "maze": "data/preprocess_maze.py",
    "musique": "data/preprocess_musique.py",
    "eureqa": "data/preprocess_eureqa.py",
    "tool": "data/preprocess_tool.py",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def output_root(config: dict[str, Any]) -> Path:
    root = repo_root() / str(config["experiment"]["output_root"])
    return require_pre_experiment_path(root)


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def shell_join(parts: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(str(p)) for p in parts)


def progress_description(benchmark: str, model: str, phase: str) -> str:
    short_model = model.rsplit("/", 1)[-1]
    return f"{benchmark}/{short_model} {phase}"


def snapshot_path(site: dict[str, Any], model_id: str) -> str:
    raw = site.get("model_snapshots", {}).get(model_id, "")
    if raw:
        return str(Path(str(raw)).expanduser())
    hf_home = Path(str(site.get("environment", {}).get("hf_home", ""))).expanduser()
    if not hf_home:
        return ""
    cache_name = "models--" + model_id.replace("/", "--")
    candidates = []
    for base in (hf_home, hf_home / "hub"):
        snapshots = base / cache_name / "snapshots"
        if snapshots.exists():
            candidates.extend([p for p in snapshots.iterdir() if p.is_dir()])
    if not candidates:
        return ""
    return str(sorted(candidates)[-1])


def snapshot_has_local_weights(path: Path) -> bool:
    index_files = list(path.glob("*.safetensors.index.json")) + list(path.glob("pytorch_model*.bin.index.json"))
    for index_path in index_files:
        try:
            index = json.loads(index_path.read_text())
        except json.JSONDecodeError:
            return False
        shard_names = sorted(set((index.get("weight_map") or {}).values()))
        if shard_names:
            return all((path / name).exists() and (path / name).is_file() for name in shard_names)
    return any(
        p.exists() and p.is_file()
        for pattern in ("*.safetensors", "pytorch_model*.bin", "model*.bin")
        for p in path.glob(pattern)
    )


def model_snapshot_is_available(site: dict[str, Any], model_id: str) -> bool:
    raw = snapshot_path(site, model_id)
    if not raw:
        return False
    path = Path(raw)
    if not path.exists() or not (path / "config.json").exists():
        return False
    has_tokenizer = (path / "tokenizer.json").exists() or (path / "tokenizer_config.json").exists()
    has_weights = snapshot_has_local_weights(path)
    return has_tokenizer and has_weights


def download_command(site: dict[str, Any], model_id: str) -> str:
    hf_home = site.get("environment", {}).get("hf_home", os.environ.get("HF_HOME", ""))
    prefix = [f"HF_HOME={hf_home}"] if hf_home else []
    return shell_join([*prefix, "huggingface-cli", "download", model_id])


def verification_command(site_config: str, model_id: str) -> str:
    return shell_join([
        ".venv/bin/python",
        "scripts/base_geometry_grid.py",
        "--site-config",
        site_config,
        "verify-model",
        "--model-id",
        model_id,
    ])


def configured_pairs(config: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = []
    benchmarks = config.get("benchmarks", {})
    for benchmark, bcfg in benchmarks.items():
        if benchmark.lower() == "ultrafeedback" or bcfg.get("enabled") is False:
            continue
        for model_id in bcfg.get("models", []):
            pairs.append({"benchmark": benchmark, "model": str(model_id), "benchmark_config": bcfg})
    return pairs


def plan_pairs(config: dict[str, Any], site: dict[str, Any], site_config: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    available = []
    deferred = []
    for pair in configured_pairs(config):
        model_id = pair["model"]
        snap = snapshot_path(site, model_id)
        base = {
            "benchmark": pair["benchmark"],
            "model": model_id,
            "dataset_split": config.get("diagnostics", {}).get("dataset_split", "train"),
            "snapshot_path": snap,
            "download_command": download_command(site, model_id),
            "verification_command": verification_command(site_config, model_id),
        }
        if model_snapshot_is_available(site, model_id):
            available.append({**pair, **base, "phase": "available", "model_status": "available"})
        else:
            deferred.append({**base, "phase": "deferred", "model_status": "deferred"})
    return available, deferred


def filter_pairs(
    pairs: list[dict[str, Any]],
    *,
    benchmark: str | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    out = []
    for pair in pairs:
        if benchmark and pair.get("benchmark") != benchmark:
            continue
        if model and pair.get("model") != model:
            continue
        out.append(pair)
    return out


def ensure_stage_dirs(root: Path, config: dict[str, Any]) -> None:
    for percent in config.get("diagnostics", {}).get("checkpoints", [5, 20, 100]):
        run_dir = require_pre_experiment_path(root / "results" / checkpoint_dir_name(int(percent)))
        run_dir.mkdir(parents=True, exist_ok=True)


def write_deferred_records(root: Path, records: list[dict[str, Any]]) -> Path:
    path = root / "deferred_models.tsv"
    rows = records or [{
        "benchmark": "",
        "model": "",
        "dataset_split": "",
        "phase": "deferred",
        "model_status": "none",
        "snapshot_path": "",
        "download_command": "",
        "verification_command": "",
    }]
    existing = {
        (
            row.get("benchmark", ""),
            row.get("model", ""),
            row.get("dataset_split", ""),
        ): row
        for row in read_tsv(path)
    }
    merged_rows = []
    for row in rows:
        current = dict(row)
        old = existing.get(
            (
                str(current.get("benchmark", "")),
                str(current.get("model", "")),
                str(current.get("dataset_split", "")),
            )
        )
        if old and old.get("model_status") not in {"", "none", "deferred"}:
            current["phase"] = old.get("phase", current.get("phase", ""))
            current["model_status"] = old["model_status"]
        merged_rows.append(current)
    rows = merged_rows
    write_tsv(path, rows)
    return path


def write_available_plan(root: Path, records: list[dict[str, Any]]) -> Path:
    path = root / "available_models.tsv"
    rows = [
        {
            "benchmark": row["benchmark"],
            "model": row["model"],
            "dataset_split": row["dataset_split"],
            "phase": row["phase"],
            "model_status": row["model_status"],
            "snapshot_path": row["snapshot_path"],
            "download_command": row["download_command"],
            "verification_command": row["verification_command"],
        }
        for row in records
    ]
    if not rows:
        rows = [{
            "benchmark": "",
            "model": "",
            "dataset_split": "",
            "phase": "available",
            "model_status": "none",
            "snapshot_path": "",
            "download_command": "",
            "verification_command": "",
        }]
    write_tsv(path, rows)
    return path


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("row_type", "")),
        str(row.get("benchmark", "")),
        str(row.get("model", "")),
        str(row.get("stage_percent", "")),
        str(row.get("phase", "")),
        str(row.get("prompt_index", "")),
    )


def merge_tsv(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = read_tsv(path)
    by_key = {row_key(row): dict(row) for row in existing}
    for row in rows:
        by_key[row_key(row)] = row
    merged = list(by_key.values())
    write_tsv(path, merged)
    return merged


def diagnostic_weights(value: Any) -> np.ndarray | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        parts = [x.strip() for x in value.split(",") if x.strip()]
        return np.asarray([float(x) for x in parts], dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def expected_prompt_count(config: dict[str, Any], root: Path, benchmark: str) -> int | None:
    split = str(config.get("diagnostics", {}).get("dataset_split", "train"))
    bcfg = config.get("benchmarks", {}).get(benchmark, {})
    data = bcfg.get("data", {})
    if benchmark == "maze" and data.get("train_size") is not None and split == "train":
        return int(data["train_size"])
    max_key = "max_train" if split == "train" else "max_test"
    if data.get(max_key) is not None and int(data.get(max_key, -1)) > 0:
        return int(data[max_key])
    parquet_path = root / "data" / benchmark / f"{split}.parquet"
    if parquet_path.exists():
        try:
            import datasets

            return len(datasets.Dataset.from_parquet(str(parquet_path)))
        except Exception:
            return None
    return None


def missing_stage_pairs(root: Path, config: dict[str, Any], pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing = []
    for percent in config.get("diagnostics", {}).get("checkpoints", [5, 20, 100]):
        summary_path = root / "results" / checkpoint_dir_name(int(percent)) / "summary.tsv"
        rows = read_tsv(summary_path)
        seen = {
            (row.get("benchmark", ""), row.get("model", ""), row.get("phase", ""))
            for row in rows
            if row.get("row_type", "aggregate") == "aggregate"
        }
        for pair in pairs:
            key = (str(pair["benchmark"]), str(pair["model"]), "available")
            if key not in seen:
                missing.append({
                    "benchmark": pair["benchmark"],
                    "model": pair["model"],
                    "stage_percent": int(percent),
                    "expected_phase": "available",
                    "reason": "summary_missing",
                })
                continue
            marker_path = (
                root
                / "results"
                / checkpoint_dir_name(int(percent))
                / "checkpoint_markers"
                / f"{slug(str(pair['benchmark']))}__{slug(str(pair['model']))}__available.json"
            )
            if not marker_path.exists():
                missing.append({
                    "benchmark": pair["benchmark"],
                    "model": pair["model"],
                    "stage_percent": int(percent),
                    "expected_phase": "available",
                    "reason": "checkpoint_marker_missing",
                })
                continue
            try:
                marker = json.loads(marker_path.read_text())
            except json.JSONDecodeError:
                marker = {}
            if marker.get("limited_run") is True:
                missing.append({
                    "benchmark": pair["benchmark"],
                    "model": pair["model"],
                    "stage_percent": int(percent),
                    "expected_phase": "available",
                    "reason": "limited_run",
                })
                continue
            expected_total = expected_prompt_count(config, root, str(pair["benchmark"]))
            if expected_total is not None and int(marker.get("total", 0)) < expected_total:
                missing.append({
                    "benchmark": pair["benchmark"],
                    "model": pair["model"],
                    "stage_percent": int(percent),
                    "expected_phase": "available",
                    "reason": "partial_total",
                    "observed_total": int(marker.get("total", 0)),
                    "expected_total": expected_total,
                })
    return missing


def process_tensor_stage(
    *,
    tensor: np.ndarray,
    config: dict[str, Any],
    root: Path,
    percent: int,
    benchmark: str,
    model: str,
    phase: str,
    model_status: str,
    snapshot: str,
    download_cmd: str,
    verify_cmd: str,
    total_prompts: int | None = None,
    limited_run: bool = False,
    max_prompts: int | None = None,
) -> None:
    diagnostics_cfg = config.get("diagnostics", {})
    run_dir = root / "results" / checkpoint_dir_name(percent)
    tensor_dir = run_dir / "reward_tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = tensor_dir / f"{slug(benchmark)}__{slug(model)}__{phase}.npy"
    np.save(tensor_path, tensor)

    bootstrap = 0 if int(percent) == 5 else int(diagnostics_cfg.get("bootstrap_samples", 200))
    diagnostics = compute_diagnostics(
        tensor,
        n_weights=int(diagnostics_cfg.get("n_weights", 2048)),
        seed=int(diagnostics_cfg.get("seed", 0)),
        bootstrap_samples=bootstrap,
        target_weights=diagnostic_weights(diagnostics_cfg.get("target_weights")),
        train_weights=diagnostic_weights(diagnostics_cfg.get("train_weights")),
    )
    diagnostics["tensor_path"] = str(tensor_path)
    (run_dir / "diagnostics_json").mkdir(parents=True, exist_ok=True)
    (run_dir / "diagnostics_json" / f"{slug(benchmark)}__{slug(model)}__{phase}.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    )

    metric_rows, summary_rows = rows_from_diagnostics(
        diagnostics,
        benchmark=benchmark,
        model=model,
        dataset_split=str(diagnostics_cfg.get("dataset_split", "train")),
        stage_percent=percent,
        phase=phase,
        model_status=model_status,
        snapshot_path=snapshot,
        download_command=download_cmd,
        verification_command=verify_cmd,
    )
    for row in [*metric_rows, *summary_rows]:
        row["limited_run"] = bool(limited_run)
        row["max_prompts"] = "" if max_prompts is None else int(max_prompts)
    all_metrics = merge_tsv(run_dir / "metrics.tsv", metric_rows)
    merge_tsv(run_dir / "summary.tsv", summary_rows)
    strata_rows, thresholds = assign_strata(all_metrics)
    write_tsv(run_dir / "strata.tsv", strata_rows)
    (run_dir / "strata_thresholds.json").write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n")
    generate_plots(run_dir / "metrics.tsv", run_dir)
    write_checkpoint_marker(
        run_dir,
        percent=percent,
        processed=int(tensor.shape[0]),
        total=int(total_prompts or tensor.shape[0]),
        phase=phase,
        benchmark=benchmark,
        model=model,
    )
    marker_dir = require_pre_experiment_path(run_dir / "checkpoint_markers")
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_path = marker_dir / f"{slug(benchmark)}__{slug(model)}__{phase}.json"
    marker_path.write_text(
        json.dumps(
            {
                "percent": int(percent),
                "processed": int(tensor.shape[0]),
                "total": int(total_prompts or tensor.shape[0]),
                "phase": phase,
                "model_status": model_status,
                "benchmark": benchmark,
                "model": model,
                "limited_run": bool(limited_run),
                "max_prompts": max_prompts,
                "snapshot_path": snapshot,
                "download_command": download_cmd,
                "verification_command": verify_cmd,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def preprocess_command(config: dict[str, Any], site: dict[str, Any], benchmark: str, data_dir: Path) -> list[str]:
    py = site.get("site", {}).get("python", ".venv/bin/python")
    bcfg = config.get("benchmarks", {}).get(benchmark, {})
    data = bcfg.get("data", {})
    parts = [py, TASK_PREPROCESS_SCRIPTS[benchmark], "--local_save_dir", str(data_dir)]
    if benchmark == "maze":
        parts.extend([
            "--train_size", str(int(data.get("train_size", 1000))),
            "--test_size", str(int(data.get("test_size", 100))),
            "--train_seed", str(int(data.get("train_seed", 42))),
            "--test_seed", str(int(data.get("test_seed", 4242))),
        ])
    elif benchmark == "musique":
        parts.extend(["--max_train", str(int(data.get("max_train", -1))), "--max_test", str(int(data.get("max_test", -1)))])
    elif benchmark == "eureqa":
        for key in ("mode", "train_split", "test_split", "mixed_split_seed", "max_train", "max_test"):
            if key in data and data[key] is not None:
                flag = "--" + key
                parts.extend([flag, str(data[key])])
    elif benchmark == "tool" and data.get("source_dir"):
        parts.extend(["--source_dir", str(data["source_dir"])])
    return parts


def preprocess_environment(site: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    site_env = site.get("environment", {})
    cache_keys = {
        "hf_home": "HF_HOME",
        "transformers_cache": "TRANSFORMERS_CACHE",
        "hf_hub_cache": "HF_HUB_CACHE",
        "pythonpath": "PYTHONPATH",
    }
    for source_key, env_key in cache_keys.items():
        value = site_env.get(source_key)
        if value:
            env[env_key] = str(value)
    if parse_bool(str(site_env.get("allow_dataset_downloads", "0"))):
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            env.pop(key, None)
    return env


def ensure_data(config: dict[str, Any], site: dict[str, Any], benchmark: str, data_dir: Path) -> None:
    split = config.get("diagnostics", {}).get("dataset_split", "train")
    if (data_dir / f"{split}.parquet").exists():
        return
    cmd = preprocess_command(config, site, benchmark, data_dir)
    result = subprocess.run(cmd, cwd=repo_root(), text=True, env=preprocess_environment(site))
    if result.returncode != 0:
        raise RuntimeError(f"Preprocess failed for {benchmark}: {shell_join(cmd)}")


def prepare_enabled_data(config: dict[str, Any], site: dict[str, Any]) -> list[str]:
    root = output_root(config)
    prepared = []
    for benchmark, bcfg in config.get("benchmarks", {}).items():
        if benchmark not in TASK_PREPROCESS_SCRIPTS or not bcfg.get("enabled", True):
            continue
        ensure_data(config, site, benchmark, root / "data" / benchmark)
        prepared.append(benchmark)
    return prepared


def render_prompt(task, row: dict[str, Any], m: int) -> list[dict[str, str]]:
    prompt = [dict(msg) for msg in row["prompt"]]
    if m > 1:
        prompt[-1]["content"] = task.rewrite_multi_solution(prompt[-1]["content"], m)
    return prompt


def score_outputs(task, row: dict[str, Any], texts: list[str], m: int) -> np.ndarray:
    gt = row["reward_model"]["ground_truth"]
    if isinstance(gt, str):
        gt = json.loads(gt)
    extra = dict(row.get("extra_info") or {})
    if m > 1:
        scored = [task.score_multi(text, m, gt, extra)["sub_scores"] for text in texts]
        return np.asarray(scored, dtype=np.float64)
    scored = [[task.score_one(text, gt, extra)["sub_scores"]] for text in texts]
    return np.asarray(scored, dtype=np.float64)


def run_pair(
    config: dict[str, Any],
    site: dict[str, Any],
    pair: dict[str, Any],
    *,
    max_prompts: int | None = None,
) -> None:
    from vpo.eval_harness import EvalHarness, bootstrap_eval_process

    bootstrap_eval_process()
    import datasets
    import vpo_tasks  # noqa: F401
    from vllm import SamplingParams
    from vpo.task import resolve

    root = output_root(config)
    benchmark = pair["benchmark"]
    model = pair["model"]
    diagnostics_cfg = config.get("diagnostics", {})
    split = diagnostics_cfg.get("dataset_split", "train")
    data_dir = root / "data" / benchmark
    ensure_data(config, site, benchmark, data_dir)
    ds = datasets.Dataset.from_parquet(str(data_dir / f"{split}.parquet"))
    full_total = len(ds)
    limited_run = max_prompts is not None and max_prompts < full_total
    if max_prompts:
        ds = ds.select(range(min(max_prompts, full_total)))

    task = resolve(benchmark)
    n_chains = int(diagnostics_cfg.get("n_chains", 10))
    m = int(diagnostics_cfg.get("num_solutions", 3))
    max_tokens = int(pair["benchmark_config"].get("eval", {}).get("max_tokens", diagnostics_cfg.get("max_tokens", 1024)))
    temperature = float(diagnostics_cfg.get("temperature", 0.7))
    max_model_len = int(pair["benchmark_config"].get("eval", {}).get("max_model_len", diagnostics_cfg.get("max_model_len", 4096)))
    gpu_mem = float(pair["benchmark_config"].get("eval", {}).get("gpu_mem", diagnostics_cfg.get("gpu_mem", 0.85)))
    tp = int(pair["benchmark_config"].get("eval", {}).get("tp", diagnostics_cfg.get("tp", 1)))
    seed = int(diagnostics_cfg.get("seed", 0))

    harness = EvalHarness(pair["snapshot_path"], max_model_len=max_model_len, gpu_mem=gpu_mem, tp=tp, seed=seed)
    params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=float(diagnostics_cfg.get("top_p", 1.0)), n=n_chains, seed=seed)
    checkpoints = [int(x) for x in diagnostics_cfg.get("checkpoints", [5, 20, 100])]
    done: set[int] = set()
    rewards = []
    raw_dir = root / "raw_outputs" / slug(benchmark) / slug(model) / pair["phase"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = raw_dir / f"outputs__{attempt_id}.jsonl"
    total = len(ds)
    batch_size = int(diagnostics_cfg.get("batch_size", 4))
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    with raw_path.open("a") as raw_f:
        progress = (
            tqdm(
                total=total,
                desc=progress_description(benchmark, model, pair["phase"]),
                unit="prompt",
                dynamic_ncols=True,
                file=sys.stderr,
                leave=True,
            )
            if tqdm is not None
            else nullcontext()
        )
        with progress as pbar:
            for start in range(0, total, batch_size):
                chunk = ds.select(range(start, min(start + batch_size, total)))
                prompts = [harness.apply_chat_template(render_prompt(task, row, m), enable_thinking=False) for row in chunk]
                outputs = harness.llm.generate(prompts, params)
                for offset, (row, output) in enumerate(zip(chunk, outputs)):
                    texts = [item.text for item in output.outputs]
                    rewards.append(score_outputs(task, row, texts, m))
                    raw_f.write(json.dumps({"prompt_index": start + offset, "outputs": texts}) + "\n")
                processed = len(rewards)
                if pbar is not None:
                    pbar.update(len(chunk))
                for pct in crossed_checkpoints(processed, total, checkpoints, done):
                    done.add(pct)
                    process_tensor_stage(
                        tensor=np.asarray(rewards, dtype=np.float64),
                        config=config,
                        root=root,
                        percent=pct,
                        benchmark=benchmark,
                        model=model,
                        phase=pair["phase"],
                        model_status=pair["model_status"],
                        snapshot=pair["snapshot_path"],
                        download_cmd=pair["download_command"],
                        verify_cmd=pair["verification_command"],
                        total_prompts=total,
                        limited_run=limited_run,
                        max_prompts=max_prompts,
                    )


def verify_model(site: dict[str, Any], model_id: str) -> tuple[bool, str]:
    snap = snapshot_path(site, model_id)
    if not snap:
        return False, f"no snapshot configured for {model_id}"
    path = Path(snap)
    if not path.exists():
        return False, f"snapshot does not exist: {path}"
    if not model_snapshot_is_available(site, model_id):
        return False, f"snapshot is missing config, tokenizer, or local weight files: {path}"
    try:
        from transformers import AutoConfig, AutoTokenizer

        AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
        AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
    except Exception as exc:
        return False, f"failed local config/tokenizer load: {exc}"
    return True, str(path)


def append_log(root: Path, line: str) -> None:
    path = root / "LOCAL_COMMAND_LOG.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(line.rstrip() + "\n")


def record_phase1_complete(root: Path, deferred: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    deferred_text = ", ".join(f"{row['benchmark']}:{row['model']}" for row in deferred) or "none"
    (root / "STATUS.md").write_text(
        "# Frozen Diagnostic Status\n\n"
        f"Last updated: {now_iso()}\n\n"
        "Available configured models are complete; deferred model download/run is starting.\n\n"
        f"Deferred model-benchmark pairs: {deferred_text}\n"
    )
    (root / "SOURCE_LEDGER.md").write_text(
        "# Frozen Diagnostic Source Ledger\n\n"
        f"Last updated: {now_iso()}\n\n"
        "| Source | Use |\n|---|---|\n"
        "| configs/geometry/base_diagnostic_grid.yaml | Frozen diagnostic benchmark/model configuration. |\n"
        "| configs/site/hpc2.yaml | Local model snapshots and cache roots. |\n"
        "| vpo_tasks/* | Official prompt/reward implementations reused without modification. |\n"
    )
    (root / "NEXT_ACTION.md").write_text(
        "# Next Action\n\n"
        "Download and verify deferred models, then run frozen diagnostics for verified deferred pairs.\n"
    )
    append_log(root, f"- {now_iso()} available configured models complete; deferred={deferred_text}")


def run_planning(config: dict[str, Any], site: dict[str, Any], site_config: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    ensure_stage_dirs(root, config)
    available, deferred = plan_pairs(config, site, site_config)
    write_available_plan(root, available)
    write_deferred_records(root, deferred)
    append_log(root, f"- {now_iso()} planned frozen diagnostics available={len(available)} deferred={len(deferred)}")
    return available, deferred


def estimated_gpu_hours(config: dict[str, Any], pairs: list[dict[str, Any]]) -> float:
    total = 0.0
    for pair in pairs:
        bcfg = config.get("benchmarks", {}).get(pair["benchmark"], {})
        total += float(bcfg.get("estimated_gpu_hours", config.get("diagnostics", {}).get("estimated_gpu_hours_per_pair", 0)))
    return total


def download_deferred_model(site: dict[str, Any], model_id: str) -> bool:
    hf_home = site.get("environment", {}).get("hf_home")
    env = os.environ.copy()
    if hf_home:
        env["HF_HOME"] = str(hf_home)
    try:
        result = subprocess.run(["huggingface-cli", "download", model_id], cwd=repo_root(), env=env, text=True)
    except FileNotFoundError:
        return False
    return result.returncode == 0


def complete_deferred_models(
    config: dict[str, Any],
    site: dict[str, Any],
    root: Path,
    deferred: list[dict[str, Any]],
    *,
    allow_downloads: bool,
    max_prompts: int | None = None,
) -> list[dict[str, Any]]:
    updated = []
    for row in deferred:
        current = dict(row)
        if not allow_downloads:
            current["model_status"] = "deferred"
            updated.append(current)
            continue
        if not download_deferred_model(site, current["model"]):
            current["model_status"] = "failed_download"
            append_log(root, f"- {now_iso()} deferred download failed benchmark={current['benchmark']} model={current['model']}")
            updated.append(current)
            continue
        current["snapshot_path"] = snapshot_path(site, current["model"])
        ok, message = verify_model(site, current["model"])
        if not ok:
            current["model_status"] = "failed_verify"
            append_log(root, f"- {now_iso()} deferred verification failed benchmark={current['benchmark']} model={current['model']} message={message}")
            updated.append(current)
            continue
        pair = {
            "benchmark": current["benchmark"],
            "model": current["model"],
            "benchmark_config": config["benchmarks"][current["benchmark"]],
            "dataset_split": current["dataset_split"],
            "snapshot_path": current["snapshot_path"],
            "download_command": current["download_command"],
            "verification_command": current["verification_command"],
            "phase": "deferred_downloaded",
            "model_status": "downloaded_verified",
        }
        append_log(root, f"- {now_iso()} deferred verified benchmark={current['benchmark']} model={current['model']}")
        run_pair(config, site, pair, max_prompts=max_prompts)
        current["phase"] = "deferred_downloaded"
        current["model_status"] = "downloaded_verified"
        updated.append(current)
    write_deferred_records(root, updated)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geometry/base_diagnostic_grid.yaml")
    parser.add_argument("--site-config", default="configs/site/hpc2.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("plan")

    sub.add_parser("prepare-data")

    sub.add_parser("finalize-phase1")

    verify = sub.add_parser("verify-model")
    verify.add_argument("--model-id", required=True)

    process = sub.add_parser("process-tensor")
    process.add_argument("--tensor", required=True)
    process.add_argument("--benchmark", required=True)
    process.add_argument("--model", required=True)
    process.add_argument("--stage-percent", type=int, required=True)
    process.add_argument("--phase", default="available")
    process.add_argument("--model-status", default="available")
    process.add_argument("--snapshot-path", default="")
    process.add_argument("--download-command", default="")
    process.add_argument("--verification-command", default="")

    run = sub.add_parser("run")
    run.add_argument("--phase", choices=["available", "deferred", "all"], default="available")
    run.add_argument("--benchmark", default=None)
    run.add_argument("--model", default=None)
    run.add_argument("--max-prompts", type=int, default=None)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--allow-downloads", action="store_true")
    run.add_argument("--allow-high-gpu-hours", action="store_true")

    args = parser.parse_args()
    config = load_yaml(repo_root() / args.config)
    site = load_yaml(repo_root() / args.site_config)
    root = output_root(config)

    if args.command == "verify-model":
        ok, message = verify_model(site, args.model_id)
        print(json.dumps({"ok": ok, "model": args.model_id, "message": message}, indent=2))
        raise SystemExit(0 if ok else 1)

    if args.command == "process-tensor":
        process_tensor_stage(
            tensor=np.load(args.tensor),
            config=config,
            root=root,
            percent=args.stage_percent,
            benchmark=args.benchmark,
            model=args.model,
            phase=args.phase,
            model_status=args.model_status,
            snapshot=args.snapshot_path,
            download_cmd=args.download_command,
            verify_cmd=args.verification_command,
        )
        return

    available, deferred = run_planning(config, site, args.site_config)
    if args.command == "plan":
        print(json.dumps({"available": len(available), "deferred": len(deferred), "output_root": str(root)}, indent=2))
        return

    if args.command == "prepare-data":
        prepared = prepare_enabled_data(config, site)
        print(json.dumps({"ok": True, "prepared": prepared}, indent=2))
        return

    if args.command == "finalize-phase1":
        missing = missing_stage_pairs(root, config, available)
        if missing:
            print(json.dumps({"ok": False, "missing": missing}, indent=2))
            raise SystemExit(1)
        record_phase1_complete(root, deferred)
        print(json.dumps({"ok": True, "available": len(available), "deferred": len(deferred)}, indent=2))
        return

    available_to_run = filter_pairs(available, benchmark=args.benchmark, model=args.model)
    deferred_to_run = filter_pairs(deferred, benchmark=args.benchmark, model=args.model)
    is_filtered = bool(args.benchmark or args.model)

    max_hours = float(config.get("safety", {}).get("max_gpu_hours_without_override", 4))
    if args.phase == "available":
        planned_for_guard = available_to_run
    elif args.phase == "deferred":
        planned_for_guard = deferred_to_run
    else:
        planned_for_guard = available_to_run + deferred_to_run
    hours = estimated_gpu_hours(config, planned_for_guard)
    if hours > max_hours and not args.allow_high_gpu_hours:
        raise SystemExit(
            f"Refusing to run estimated {hours:.2f} GPU-hours without --allow-high-gpu-hours."
        )

    if args.command == "run" and args.dry_run:
        dry_available = available_to_run if args.phase in {"available", "all"} else []
        dry_deferred = deferred_to_run if args.phase in {"deferred", "all"} else []
        print(json.dumps({
            "available": len(dry_available),
            "deferred": len(dry_deferred),
            "dry_run": True,
            "phase": args.phase,
            "benchmark": args.benchmark or "",
            "model": args.model or "",
        }, indent=2))
        return

    if args.phase in {"available", "all"}:
        for pair in available_to_run:
            run_pair(config, site, pair, max_prompts=args.max_prompts)

    if args.phase in {"available", "all"} and not is_filtered:
        record_phase1_complete(root, deferred)
    elif args.phase in {"available", "all"} and is_filtered:
        append_log(
            root,
            f"- {now_iso()} filtered available run complete benchmark={args.benchmark or '*'} model={args.model or '*'} pairs={len(available_to_run)}",
        )

    if args.phase in {"deferred", "all"}:
        complete_deferred_models(
            config,
            site,
            root,
            deferred_to_run,
            allow_downloads=args.allow_downloads,
            max_prompts=args.max_prompts,
        )


if __name__ == "__main__":
    main()
