#!/usr/bin/env python3
"""Execute one Stage 2 short-training job from a generated manifest."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.short_training_common import (
    apply_site_environment,
    diagnostics_command,
    eval_command,
    merge_command,
    now_iso,
    parse_bool,
    preprocess_command,
    progress,
    read_json,
    repo_root,
    require_stage2_path,
    run_command,
    shell_join,
    train_segment_command,
    wait_for_gpu_quiescence,
    write_json,
)


def find_job(manifest: dict[str, Any], job_id: str) -> dict[str, Any]:
    for job in manifest.get("jobs", []):
        if job.get("job_id") == job_id:
            return job
    raise ValueError(f"No job_id={job_id!r} in manifest")


def stage2_root_from_manifest(manifest: dict[str, Any]) -> Path:
    return repo_root() / str(manifest.get("output_root", "pre_experiments/2_short_training"))


def ensure_logs_symlink(job: dict[str, Any], allow_main_experiments: bool) -> None:
    repo = repo_root()
    link = repo / "logs"
    target = Path(job["run_dir"]) / "runtime_links" / "logs"
    require_stage2_path(target, False)
    target.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if not allow_main_experiments:
        raise RuntimeError(
            "train.sh writes to ./logs; refusing to retarget the top-level logs symlink "
            "without --allow_main_experiments=true."
        )
    if link.exists() or link.is_symlink():
        if not link.is_symlink():
            raise RuntimeError(f"Refusing to replace non-symlink logs path: {link}")
        link.unlink()
    link.symlink_to(target)


def ensure_dirs(job: dict[str, Any]) -> None:
    for key in ("run_dir", "data_dir", "checkpoint_root"):
        require_stage2_path(Path(job[key]), False).mkdir(parents=True, exist_ok=True)
    for record in job.get("checkpoints", []):
        for key in ("eval_output_path", "diagnostics_path", "health_path", "job_summary_path"):
            require_stage2_path(Path(record[key]), False).parent.mkdir(parents=True, exist_ok=True)


def write_summary(record: dict[str, Any], status: str, extra: dict[str, Any], allow: bool = False) -> None:
    payload = dict(record)
    payload.update(extra)
    payload["status"] = status
    write_json(Path(record["job_summary_path"]), payload, allow)


def require_success(command: str, log_path: Path, record: dict[str, Any], extra: dict[str, Any]) -> None:
    rc = run_command(command, log_path)
    if rc != 0:
        write_summary(record, "FAIL", {**extra, "failed_command": command, "returncode": rc})
        raise SystemExit(rc)


def run_record(
    manifest: dict[str, Any],
    job: dict[str, Any],
    record: dict[str, Any],
    allow_main_experiments: bool,
) -> None:
    config = read_json_like_yaml(manifest["_config_path"])
    site = read_json_like_yaml(manifest["_site_path"])
    benchmark = job["benchmark_config"]
    method = str(job["method"])
    seed = int(job["seed"])
    update = int(record["update"])
    run_name = str(job["run_name"])
    data_dir = Path(job["data_dir"])
    checkpoint_root = Path(job["checkpoint_root"])
    log_path = Path(job["log_path"])

    train_cmd = "frozen_base_model_no_training"
    merge_cmd = ""
    if update > 0:
        train_cmd = train_segment_command(
            config,
            benchmark,
            method,
            seed,
            run_name,
            data_dir,
            checkpoint_root,
            update,
            record.get("previous_update"),
            site,
        )
        actor_dir = checkpoint_root / f"global_step_{update}" / "actor"
        merged_dir = actor_dir / "huggingface_merged"
        if not actor_dir.exists():
            require_success(train_cmd, log_path, record, {"train_command": train_cmd})
        merge_cmd = merge_command(site, actor_dir, merged_dir)
        if not merged_dir.exists():
            require_success(merge_cmd, log_path, record, {"train_command": train_cmd, "merge_command": merge_cmd})
        wait_for_gpu_quiescence(log_path, desc=f"{run_name} u{update} gpu settle")

    model_arg = str(record["runtime_model"]) if update == 0 else str(Path(record["checkpoint_path"]))
    eval_cmd = eval_command(config, site, benchmark, method, seed, model_arg, data_dir, Path(record["eval_output_path"]))
    if not Path(record["eval_output_path"]).exists():
        require_success(eval_cmd, log_path, record, {"train_command": train_cmd, "merge_command": merge_cmd, "eval_command": eval_cmd})

    diag_cmd = diagnostics_command(site, Path(record["tensor_path"]), Path(record["diagnostics_path"]))
    if Path(record["tensor_path"]).exists() and not Path(record["diagnostics_path"]).exists():
        require_success(
            diag_cmd,
            log_path,
            record,
            {"train_command": train_cmd, "merge_command": merge_cmd, "eval_command": eval_cmd, "diagnostics_command": diag_cmd},
        )

    py = site.get("site", {}).get("python", ".venv/bin/python")
    health_cmd = shell_join(
        [
            py,
            "scripts/collect_train_health.py",
            "--run-dir",
            Path(job["run_dir"]),
            "--update",
            update,
            "--output",
            Path(record["health_path"]),
        ]
    )
    if not Path(record["health_path"]).exists():
        require_success(
            health_cmd,
            log_path,
            record,
            {
                "train_command": train_cmd,
                "merge_command": merge_cmd,
                "eval_command": eval_cmd,
                "diagnostics_command": diag_cmd,
                "health_command": health_cmd,
            },
        )

    write_summary(
        record,
        "PASS",
        {
            "train_command": train_cmd,
            "merge_command": merge_cmd,
            "eval_command": eval_cmd,
            "diagnostics_command": diag_cmd,
            "health_command": health_cmd,
            "completed_at": now_iso(),
        },
        allow_main_experiments,
    )


def read_json_like_yaml(path: str | Path) -> dict[str, Any]:
    from scripts.short_training_common import load_yaml

    return load_yaml(Path(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--allow_main_experiments", nargs="?", const=True, default=False, type=parse_bool)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = read_json(manifest_path)
    manifest["_config_path"] = manifest.get("config_path")
    manifest["_site_path"] = manifest.get("site_config_path")
    site = read_json_like_yaml(manifest["_site_path"])
    apply_site_environment(site)
    job = find_job(manifest, args.job_id)
    ensure_dirs(job)
    ensure_logs_symlink(job, bool(args.allow_main_experiments))

    prep = preprocess_command(site, job["benchmark_config"], Path(job["data_dir"]))
    data_dir = Path(job["data_dir"])
    if not (data_dir / "train.parquet").exists() or not (data_dir / "test.parquet").exists():
        require_success(
            prep,
            Path(job["log_path"]),
            job["checkpoints"][0],
            {"preprocess_command": prep},
        )

    for record in progress(job.get("checkpoints", []), desc=f"{args.job_id} updates"):
        run_record(manifest, job, record, bool(args.allow_main_experiments))


if __name__ == "__main__":
    main()
