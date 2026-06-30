#!/usr/bin/env python3
"""Generate Stage 2 short-training diagnostics manifests and Slurm scripts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.short_training_common import (
    METHODS,
    STAGE2_ROOT,
    append_line,
    comma_list,
    int_list,
    load_yaml,
    now_iso,
    parse_bool,
    progress,
    repo_root,
    require_stage2_path,
    runtime_model_path,
    shell_join,
    site_exports,
    slurm_header,
    submit_sbatch,
    write_json,
)


def stage2_root(config: dict[str, Any]) -> Path:
    return repo_root() / config.get("experiment", {}).get("output_root", STAGE2_ROOT)


def normalized_smoke_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    smoke = dict(config["smoke"])
    smoke["name"] = str(smoke.get("benchmark", smoke.get("name", "maze")))
    return smoke


def selected_benchmarks(config: dict[str, Any], smoke: bool, names: list[str]) -> list[dict[str, Any]]:
    if smoke:
        return [normalized_smoke_benchmark(config)]
    benchmarks = config.get("benchmarks", {})
    missing = [name for name in names if name not in benchmarks]
    if missing:
        raise ValueError(f"Unknown benchmarks: {', '.join(missing)}")
    return [dict(benchmarks[name]) for name in names]


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def run_name(benchmark: str, method: str, seed: int, attempt_label: str) -> str:
    return f"{method}_{benchmark}_seed{seed}_short_training_{attempt_label}"


def job_name(job: dict[str, Any]) -> str:
    return slug(f"s2_{job['benchmark']}_{job['method']}_seed{job['seed']}_{job['attempt_label']}")[:128]


def checkpoint_records(
    config: dict[str, Any],
    site: dict[str, Any],
    benchmark: dict[str, Any],
    method: str,
    seed: int,
    attempt_label: str,
    root: Path,
    run_dir: Path,
    checkpoint_root: Path,
) -> list[dict[str, Any]]:
    task = str(benchmark["name"])
    records = []
    updates = sorted({int(update) for update in benchmark.get("updates", [])})
    previous_positive: int | None = None
    for update in updates:
        eval_dir = run_dir / "eval" / f"update_{update}"
        output_json = eval_dir / f"{run_name(task, method, seed, attempt_label)}_u{update}.json"
        tensor_path = output_json.with_suffix(".npy") if task == "maze" else output_json.with_name(f"{output_json.stem}_tensor.npy")
        diag_path = run_dir / "diagnostics" / f"{run_name(task, method, seed, attempt_label)}_u{update}_diagnostics.json"
        health_path = run_dir / "health" / f"{run_name(task, method, seed, attempt_label)}_u{update}_health.json"
        summary_path = run_dir / "job_summaries" / f"{run_name(task, method, seed, attempt_label)}_u{update}.json"
        if update == 0:
            checkpoint_path = f"model://{benchmark['model']}"
            model_arg = runtime_model_path(site, str(benchmark["model"]))
            train_command = "frozen_base_model_no_training"
        else:
            checkpoint_path = str(checkpoint_root / f"global_step_{update}" / "actor" / "huggingface_merged")
            model_arg = checkpoint_path
            train_command = "generated_by_scripts/run_short_training_job.py"
        records.append(
            {
                "schema_version": 1,
                "benchmark": task,
                "method": method,
                "eval_method": config.get("methods", {}).get("eval_aliases", {}).get(task, {}).get(method, method),
                "model": str(benchmark["model"]),
                "runtime_model": model_arg,
                "seed": int(seed),
                "update": int(update),
                "previous_update": previous_positive,
                "checkpoint_path": checkpoint_path,
                "train_command": train_command,
                "eval_command": "generated_by_scripts/run_short_training_job.py",
                "diagnostics_command": "generated_by_scripts/run_short_training_job.py",
                "health_command": "generated_by_scripts/run_short_training_job.py",
                "data_dir": str(run_dir / "data" / task),
                "eval_output_path": str(output_json),
                "tensor_path": str(tensor_path),
                "diagnostics_path": str(diag_path),
                "health_path": str(health_path),
                "job_summary_path": str(summary_path),
                "artifact_paths": [
                    str(output_json),
                    str(tensor_path),
                    str(diag_path),
                    str(health_path),
                    str(summary_path),
                ],
                "dataset_split": str(config.get("defaults", {}).get("dataset_split", "test")),
                "metric_provenance": "scripts/eval.py::short_training_official_eval_tensor",
            }
        )
        if update > 0:
            previous_positive = update
    return records


def build_job(
    config: dict[str, Any],
    site: dict[str, Any],
    benchmark: dict[str, Any],
    method: str,
    seed: int,
    attempt_label: str,
    mode: str,
) -> dict[str, Any]:
    root = stage2_root(config)
    task = str(benchmark["name"])
    run = run_name(task, method, seed, attempt_label)
    stage_dir = root / ("smoke" if mode == "smoke" else task)
    run_dir = stage_dir / run
    checkpoint_root = run_dir / "checkpoints"
    slurm_path = root / "slurm" / f"{slug(run)}.sbatch"
    stdout_path = root / "logs" / f"{slug(run)}_%j.out"
    stderr_path = root / "logs" / f"{slug(run)}_%j.err"
    job = {
        "schema_version": 1,
        "job_id": slug(run),
        "mode": mode,
        "attempt_label": attempt_label,
        "benchmark": task,
        "method": method,
        "model": str(benchmark["model"]),
        "runtime_model": runtime_model_path(site, str(benchmark["model"])),
        "seed": int(seed),
        "updates": sorted({int(x) for x in benchmark.get("updates", [])}),
        "estimated_gpu_hours": float(benchmark.get("estimated_gpu_hours", 0)),
        "run_name": run,
        "run_dir": str(run_dir),
        "data_dir": str(run_dir / "data" / task),
        "checkpoint_root": str(checkpoint_root),
        "log_path": str(run_dir / "logs" / f"{run}.log"),
        "slurm_path": str(slurm_path),
        "slurm_stdout": str(stdout_path),
        "slurm_stderr": str(stderr_path),
        "created_at": now_iso(),
        "benchmark_config": benchmark,
        "checkpoints": [],
    }
    job["checkpoints"] = checkpoint_records(
        config,
        site,
        benchmark,
        method,
        seed,
        attempt_label,
        root,
        run_dir,
        checkpoint_root,
    )
    return job


def build_jobs(
    config: dict[str, Any],
    site: dict[str, Any],
    smoke: bool,
    benchmarks: list[str],
    methods_arg: str | None,
    seeds_arg: str | None,
    updates_arg: str | None,
    attempt_label: str,
) -> list[dict[str, Any]]:
    mode = "smoke" if smoke else "benchmark"
    benchmark_defs = selected_benchmarks(config, smoke, benchmarks)
    jobs: list[dict[str, Any]] = []
    for benchmark in benchmark_defs:
        default_methods = list(benchmark.get("methods", config.get("methods", {}).get("active", METHODS)))
        methods = comma_list(methods_arg, default_methods)
        unknown = [method for method in methods if method not in METHODS]
        if unknown:
            raise ValueError(f"Unknown methods: {', '.join(unknown)}")
        seeds = int_list(seeds_arg, [int(x) for x in benchmark.get("seeds", config.get("defaults", {}).get("seeds", [0]))])
        if updates_arg:
            benchmark = dict(benchmark)
            benchmark["updates"] = int_list(updates_arg, [])
        for method in methods:
            for seed in seeds:
                jobs.append(build_job(config, site, benchmark, method, seed, attempt_label, mode))
    return jobs


def write_slurm(job: dict[str, Any], manifest_path: Path, site: dict[str, Any], allow_main_experiments: bool) -> None:
    path = require_stage2_path(Path(job["slurm_path"]), False)
    path.parent.mkdir(parents=True, exist_ok=True)
    for key in ("slurm_stdout", "slurm_stderr"):
        require_stage2_path(Path(job[key]), False).parent.mkdir(parents=True, exist_ok=True)
    py = site.get("site", {}).get("python", ".venv/bin/python")
    runner = shell_join(
        [
            py,
            "scripts/run_short_training_job.py",
            "--manifest",
            str(manifest_path),
            "--job-id",
            job["job_id"],
            "--allow_main_experiments=true" if allow_main_experiments else "--allow_main_experiments=false",
        ]
    )
    lines = [
        slurm_header(site, job["benchmark_config"], job_name(job), Path(job["slurm_stdout"]), Path(job["slurm_stderr"])),
        "",
        *site_exports(site),
        runner,
        "",
    ]
    path.write_text("\n".join(lines))


def manifest_path_for(config: dict[str, Any], mode: str, attempt_label: str) -> Path:
    suffix = "" if attempt_label == "a1" else f"_{attempt_label}"
    return stage2_root(config) / "manifests" / f"short_training_{mode}_jobs{suffix}.json"


def write_manifest(path: Path, config_path: Path, site_path: Path, jobs: list[dict[str, Any]], allow: bool) -> None:
    write_json(
        path,
        {
            "schema_version": 1,
            "created_at": now_iso(),
            "config_path": str(config_path),
            "site_config_path": str(site_path),
            "output_root": STAGE2_ROOT,
            "jobs": jobs,
        },
        allow,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geometry/short_training_dynamics.yaml")
    parser.add_argument("--site-config", default="configs/site/hpc2.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmarks", default="maze,musique,tool")
    parser.add_argument("--methods", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--checkpoint-updates", default=None)
    parser.add_argument("--attempt-label", default="a1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--generate-slurm", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--allow-high-gpu-hours", action="store_true")
    parser.add_argument("--allow_main_experiments", nargs="?", const=True, default=False, type=parse_bool)
    args = parser.parse_args()

    config_path = repo_root() / args.config
    site_path = repo_root() / args.site_config
    config = load_yaml(config_path)
    site = load_yaml(site_path)
    mode = "smoke" if args.smoke else "benchmark"
    jobs = build_jobs(
        config,
        site,
        args.smoke,
        comma_list(args.benchmarks, ["maze", "musique", "tool"]),
        args.methods,
        args.seeds,
        args.checkpoint_updates,
        args.attempt_label,
    )
    max_hours = float(config.get("safety", {}).get("max_gpu_hours_without_override", 4))
    too_large = [job for job in jobs if float(job.get("estimated_gpu_hours", 0)) > max_hours]
    if args.submit and too_large and not args.allow_high_gpu_hours:
        labels = ", ".join(f"{job['benchmark']}/{job['method']}" for job in too_large)
        raise SystemExit(f"Refusing to submit jobs above {max_hours} GPU-hours without --allow-high-gpu-hours: {labels}")

    manifest_path = manifest_path_for(config, mode, args.attempt_label)
    print(json.dumps({"mode": mode, "jobs": len(jobs), "manifest": str(manifest_path), "submit": args.submit}, indent=2))
    if args.dry_run and not args.generate_slurm:
        for job in jobs:
            print(f"DRY-RUN {job['job_id']} updates={job['updates']} slurm={job['slurm_path']}")
        return

    write_manifest(manifest_path, config_path, site_path, jobs, False)
    for job in progress(jobs, desc="write slurm"):
        write_slurm(job, manifest_path, site, bool(args.allow_main_experiments))

    command_log = stage2_root(config) / "LOCAL_COMMAND_LOG.md"
    append_line(command_log, f"- {now_iso()} generated mode={mode} manifest={manifest_path} jobs={len(jobs)} submit={args.submit}")
    if args.submit:
        for job in progress(jobs, desc="submit"):
            slurm_id = submit_sbatch(Path(job["slurm_path"]))
            job["slurm_id"] = slurm_id
            job["submitted_at"] = now_iso()
            append_line(command_log, f"- {now_iso()} submitted job_id={job['job_id']} slurm_id={slurm_id} script={job['slurm_path']}")
        write_manifest(manifest_path, config_path, site_path, jobs, False)
        print("Submitted: " + ", ".join(str(job.get("slurm_id")) for job in jobs))
    else:
        print(f"Generated manifest: {manifest_path}")


if __name__ == "__main__":
    main()
