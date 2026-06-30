#!/usr/bin/env python3
"""Canonical result-table producer for staged benchmark artifacts.

This script does not run task evaluation. It validates result manifests and
aggregates them into one table with command, checkpoint, artifact, split, and
metric provenance preserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np


REQUIRED_FIELDS = (
    "benchmark",
    "method",
    "model",
    "seed",
    "checkpoint_path",
    "train_command",
    "eval_command",
    "artifact_paths",
    "metric_provenance",
    "dataset_split",
)
OPTIONAL_BEST_FIELDS = ("best@1", "best@3", "best@10", "best@30")
OPTIONAL_PROVENANCE_FIELDS = (
    "attempt_label",
    "job_kind",
    "resource_provenance",
    "trainer_n_gpus",
    "slurm_partition",
    "slurm_gres",
    "slurm_id",
    "slurm_path",
)
STAGE2_TABLES = (
    "checkpoint_table.csv",
    "method_difference_table.csv",
    "diagnostic_table.csv",
    "train_health_table.csv",
    "final_short_training_summary.md",
)
STAGE2_DIAGNOSTIC_FIELDS = (
    "best@1",
    "best@3",
    "best@10",
    "best@30",
    "eum_gap",
    "winner_entropy",
    "dominant_candidate_mass",
    "target_regret",
    "base_best_of_k_slope",
    "reward_collinearity",
    "effective_rank",
    "pareto_fraction",
)


def parse_bool(value: str | bool | None) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def repo_root() -> Path:
    return _REPO_ROOT


def stage2_path_guard():
    try:
        from scripts.short_training_common import require_stage2_path
    except ModuleNotFoundError:
        from short_training_common import require_stage2_path

    return require_stage2_path


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
            "Pass --allow_main_experiments to override."
        ) from exc
    return resolved


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("rows") or data.get("results") or data.get("jobs")
    else:
        rows = None
    if not isinstance(rows, list):
        raise ValueError("Manifest must be a JSON list or an object with rows/results/jobs.")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Every manifest row must be an object.")
    return rows


def validate_checkpoint_path(raw_path: str) -> str:
    if not raw_path:
        raise ValueError("Missing checkpoint_path.")
    if raw_path.startswith(("hf://", "model://")):
        return raw_path
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise ValueError(f"checkpoint_path does not exist: {raw_path}")
    return str(path)


def validate_row(row: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if row.get(field) in (None, "", [])]
    if missing:
        raise ValueError(f"Manifest row is missing required fields: {', '.join(missing)}")
    if "mean" not in row and "metric_value" not in row:
        raise ValueError("Manifest row must contain mean or metric_value.")
    if not row.get("train_command") or not row.get("eval_command"):
        raise ValueError("Manifest row must contain train_command and eval_command provenance.")
    artifacts = row.get("artifact_paths")
    if isinstance(artifacts, str):
        artifacts = [artifacts]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("artifact_paths must be a non-empty list or string.")

    out = dict(row)
    out["checkpoint_path"] = validate_checkpoint_path(str(row["checkpoint_path"]))
    out["artifact_paths"] = [str(x) for x in artifacts]
    out["mean"] = float(row.get("mean", row.get("metric_value")))
    out["seed"] = int(row["seed"])
    for key in OPTIONAL_BEST_FIELDS:
        if key in out and out[key] is not None:
            out[key] = float(out[key])
    for key in OPTIONAL_PROVENANCE_FIELDS:
        if key not in out or out[key] is None:
            out[key] = ""
    if out["trainer_n_gpus"] not in ("", None):
        out["trainer_n_gpus"] = str(out["trainer_n_gpus"])
    return out


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def unique_join(values: list[Any]) -> str:
    seen: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.append(text)
    return " | ".join(seen)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validated = [validate_row(row) for row in rows]
    grouped: dict[tuple[str, str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in validated:
        key = (
            str(row["benchmark"]),
            str(row["method"]),
            str(row["model"]),
            str(row["dataset_split"]),
            str(row["metric_provenance"]),
            str(row.get("attempt_label", "")),
            str(row.get("resource_provenance", "")),
        )
        grouped[key].append(row)

    table: list[dict[str, Any]] = []
    for (
        benchmark,
        method,
        model,
        dataset_split,
        metric_provenance,
        attempt_label,
        resource_provenance,
    ), group in sorted(grouped.items()):
        group = sorted(group, key=lambda x: x["seed"])
        means = [float(row["mean"]) for row in group]
        artifact_paths: list[str] = []
        for row in group:
            artifact_paths.extend(row["artifact_paths"])
        output_row: dict[str, Any] = {
            "benchmark": benchmark,
            "method": method,
            "model": model,
            "seed": ",".join(str(row["seed"]) for row in group),
            "seed_count": len(group),
            "mean": sum(means) / len(means),
            "std": sample_std(means),
            "checkpoint_path": unique_join([row["checkpoint_path"] for row in group]),
            "train_command": unique_join([row["train_command"] for row in group]),
            "eval_command": unique_join([row["eval_command"] for row in group]),
            "artifact_paths": unique_join(artifact_paths),
            "metric_provenance": metric_provenance,
            "dataset_split": dataset_split,
            "attempt_label": attempt_label,
            "job_kind": unique_join([row.get("job_kind", "") for row in group if row.get("job_kind")]),
            "resource_provenance": resource_provenance,
            "trainer_n_gpus": unique_join([row.get("trainer_n_gpus", "") for row in group if row.get("trainer_n_gpus")]),
            "slurm_partition": unique_join([row.get("slurm_partition", "") for row in group if row.get("slurm_partition")]),
            "slurm_gres": unique_join([row.get("slurm_gres", "") for row in group if row.get("slurm_gres")]),
            "slurm_id": unique_join([row.get("slurm_id", "") for row in group if row.get("slurm_id")]),
            "slurm_path": unique_join([row.get("slurm_path", "") for row in group if row.get("slurm_path")]),
        }
        for key in OPTIONAL_BEST_FIELDS:
            vals = [float(row[key]) for row in group if key in row and row[key] is not None]
            output_row[key] = sum(vals) / len(vals) if vals else None
        diagnostics = [
            str(row["pre_training_diagnostics"])
            for row in group
            if row.get("pre_training_diagnostics")
        ]
        output_row["pre_training_diagnostics"] = unique_join(diagnostics) if diagnostics else ""
        table.append(output_row)
    return table


def _mean_list(values: list[Any]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("Cannot average an empty metric list.")
    return sum(vals) / len(vals)


def _mean_named_components(summary: dict[str, Any], feature_names: list[Any]) -> float | None:
    if not feature_names:
        return None
    values = []
    for name in feature_names:
        key = f"{name}/mean"
        if key not in summary:
            return None
        values.append(summary[key])
    return _mean_list(values)


def mean_from_eval_summary(summary: dict[str, Any]) -> float:
    if "mean_scalar_reward" in summary:
        return float(summary["mean_scalar_reward"])
    nested = summary.get("summary")
    if isinstance(nested, dict):
        if "Ew_max_pool" in nested:
            return float(nested["Ew_max_pool"])
        feature_mean = _mean_named_components(nested, list(summary.get("feature_names", [])))
        if feature_mean is not None:
            return feature_mean
        if "sum_vec/mean" in nested and summary.get("feature_names"):
            return float(nested["sum_vec/mean"]) / len(summary["feature_names"])
    per_objective = summary.get("per_objective_mean")
    if isinstance(per_objective, dict) and per_objective:
        return _mean_list(list(per_objective.values()))
    per_component = summary.get("per_component_mean")
    if isinstance(per_component, list) and per_component:
        return _mean_list(per_component)
    if "completion_rate" in summary:
        return float(summary["completion_rate"])
    raise ValueError("Could not derive a canonical mean from eval summary.")


def best_at_from_eval_summary(summary: dict[str, Any]) -> dict[str, float]:
    raw = summary.get("best_at_k")
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    out: dict[str, float] = {}
    sources = [summary]
    nested = summary.get("summary")
    if isinstance(nested, dict):
        sources.append(nested)
    for source in sources:
        for key, value in source.items():
            if key.startswith("best_at_k/k=") and key.endswith("/mean"):
                k = key.removeprefix("best_at_k/k=").removesuffix("/mean")
                out[f"best@{k}"] = float(value)
    curve = summary.get("best_at_k_curve")
    if isinstance(curve, list):
        for k in (1, 3, 10, 30):
            if k <= len(curve) and f"best@{k}" not in out:
                out[f"best@{k}"] = float(curve[k - 1])
    return out


def parse_train_n_gpus(train_command: str) -> str:
    match = re.search(r"(?:^|\s)N_GPUS=(?P<value>[^\s]+)", train_command)
    if not match:
        return ""
    return match.group("value").strip("'\"")


def parse_slurm_directive(path: Path, name: str) -> str:
    if not path.is_file():
        return ""
    prefix = f"#SBATCH --{name}="
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    return ""


def resource_fields_from_job(job: dict[str, Any]) -> dict[str, str]:
    train_command = str(job.get("train_command", ""))
    slurm_path = Path(str(job.get("slurm_path", ""))).expanduser()
    trainer_n_gpus = str(job.get("trainer_n_gpus") or parse_train_n_gpus(train_command))
    slurm_partition = str(job.get("slurm_partition") or parse_slurm_directive(slurm_path, "partition"))
    slurm_gres = str(job.get("slurm_gres") or parse_slurm_directive(slurm_path, "gres"))
    parts = [
        f"attempt={job.get('attempt_label')}" if job.get("attempt_label") else "",
        f"trainer_n_gpus={trainer_n_gpus}" if trainer_n_gpus else "",
        f"slurm_partition={slurm_partition}" if slurm_partition else "",
        f"slurm_gres={slurm_gres}" if slurm_gres else "",
    ]
    return {
        "attempt_label": str(job.get("attempt_label", "")),
        "job_kind": str(job.get("job_kind", "")),
        "trainer_n_gpus": trainer_n_gpus,
        "slurm_partition": slurm_partition,
        "slurm_gres": slurm_gres,
        "slurm_id": str(job.get("slurm_id", "")),
        "slurm_path": str(job.get("slurm_path", "")),
        "resource_provenance": "; ".join(part for part in parts if part),
    }


def rows_from_job_manifest(path: Path, skip_missing: bool = False) -> list[dict[str, Any]]:
    manifest = read_json(path)
    jobs = manifest.get("jobs") if isinstance(manifest, dict) else None
    if not isinstance(jobs, list):
        raise ValueError("Job manifest must be an object with a jobs list.")

    rows: list[dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("Every job manifest entry must be an object.")
        eval_path = Path(str(job.get("eval_output_path", ""))).expanduser()
        if not eval_path.exists():
            if skip_missing:
                continue
            raise ValueError(f"Missing eval artifact for job {job.get('method')}: {eval_path}")
        summary = read_json(eval_path)
        best_at = best_at_from_eval_summary(summary) if isinstance(summary, dict) else {}
        artifact_paths = list(job.get("artifact_paths", []))
        tensor_path = summary.get("tensor_path") if isinstance(summary, dict) else None
        if tensor_path and tensor_path not in artifact_paths:
            artifact_paths.append(str(tensor_path))
        row = {
            "benchmark": job.get("benchmark"),
            "method": job.get("method"),
            "model": job.get("model"),
            "seed": job.get("seed"),
            "checkpoint_path": job.get("checkpoint_path"),
            "train_command": job.get("train_command"),
            "eval_command": job.get("eval_command"),
            "artifact_paths": artifact_paths,
            "metric_provenance": f"{job.get('metric_provenance', 'scripts/eval.py::eval_summary')}::{job.get('job_kind', 'unknown_job_kind')}",
            "dataset_split": job.get("dataset_split", "test"),
            "mean": mean_from_eval_summary(summary),
            "pre_training_diagnostics": job.get("pre_training_diagnostics", ""),
        }
        row.update(resource_fields_from_job(job))
        for key in OPTIONAL_BEST_FIELDS:
            if isinstance(best_at, dict) and key in best_at:
                row[key] = best_at[key]
        rows.append(row)
    return rows


def per_dimension_rewards(summary: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    raw_means = summary.get("per_objective_mean")
    if isinstance(raw_means, dict):
        means.update({str(k): float(v) for k, v in raw_means.items()})
    raw_stds = summary.get("per_objective_std")
    if isinstance(raw_stds, dict):
        stds.update({str(k): float(v) for k, v in raw_stds.items()})

    nested = summary.get("summary")
    if isinstance(nested, dict):
        names = summary.get("feature_names", [])
        if isinstance(names, list):
            for name in names:
                mean_key = f"{name}/mean"
                std_key = f"{name}/std"
                if mean_key in nested:
                    means[str(name)] = float(nested[mean_key])
                if std_key in nested:
                    stds[str(name)] = float(nested[std_key])
    raw_components = summary.get("per_component_mean")
    names = summary.get("feature_names", [])
    if isinstance(raw_components, list):
        for idx, value in enumerate(raw_components):
            name = str(names[idx]) if isinstance(names, list) and idx < len(names) else f"component_{idx}"
            means[name] = float(value)
    return means, stds


def read_diagnostic_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    data = read_json(path)
    metrics = data.get("metrics") if isinstance(data, dict) else None
    if not isinstance(metrics, dict):
        return {}
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
    return out


def stage2_rows_from_manifest(path: Path, skip_missing: bool = False) -> list[dict[str, Any]]:
    manifest = read_json(path)
    jobs = manifest.get("jobs") if isinstance(manifest, dict) else None
    if not isinstance(jobs, list):
        raise ValueError("Stage 2 manifest must contain jobs list.")

    rows: list[dict[str, Any]] = []
    for job in jobs:
        for record in job.get("checkpoint_records", []):
            eval_path = Path(str(record.get("eval_output_path", ""))).expanduser()
            summary_path = Path(str(record.get("job_summary_path", ""))).expanduser()
            if not eval_path.exists():
                if skip_missing:
                    continue
                raise ValueError(f"Missing Stage 2 eval artifact for {job.get('job_id')} update={record.get('update')}: {eval_path}")
            checkpoint_path = str(record.get("checkpoint_path", ""))
            if checkpoint_path and not checkpoint_path.startswith("model://") and not Path(checkpoint_path).exists():
                if skip_missing:
                    continue
                raise ValueError(f"Missing Stage 2 checkpoint for {job.get('job_id')} update={record.get('update')}: {checkpoint_path}")
            eval_summary = read_json(eval_path)
            if not isinstance(eval_summary, dict):
                raise ValueError(f"Eval summary must be an object: {eval_path}")
            job_summary = read_json(summary_path) if summary_path.exists() else {}
            if not isinstance(job_summary, dict):
                job_summary = {}
            best_at = best_at_from_eval_summary(eval_summary)
            means, stds = per_dimension_rewards(eval_summary)
            diagnostics = read_diagnostic_metrics(Path(str(record.get("diagnostics_path", ""))).expanduser())
            artifact_paths = list(record.get("artifact_paths", []))
            tensor_path = eval_summary.get("tensor_path")
            if tensor_path and str(tensor_path) not in artifact_paths:
                artifact_paths.append(str(tensor_path))
            row: dict[str, Any] = {
                "benchmark": job.get("benchmark"),
                "method": job.get("method"),
                "model": job.get("model"),
                "seed": int(job.get("seed", 0)),
                "update": int(record.get("update", 0)),
                "checkpoint_path": checkpoint_path,
                "train_command": job_summary.get("train_command", "frozen_base_model_no_training" if int(record.get("update", 0)) == 0 else ""),
                "eval_command": job_summary.get("eval_command", ""),
                "diagnostics_command": job_summary.get("diagnostics_command", ""),
                "artifact_paths": " | ".join(str(x) for x in artifact_paths),
                "dataset_split": job.get("dataset_split", "test"),
                "metric_provenance": job.get("metric_provenance", "scripts/eval.py::stage2"),
                "mean": mean_from_eval_summary(eval_summary),
                "std": 0.0,
                "per_dimension_reward_means": json.dumps(means, sort_keys=True),
                "per_dimension_reward_stds": json.dumps(stds, sort_keys=True),
                "attempt_label": job.get("attempt_label", ""),
                "job_id": job.get("job_id", ""),
                "job_summary_path": str(summary_path),
            }
            for key in OPTIONAL_BEST_FIELDS:
                row[key] = best_at.get(key)
            row.update({k: diagnostics.get(k) for k in STAGE2_DIAGNOSTIC_FIELDS if k in diagnostics})
            rows.append(row)
    return sorted(rows, key=lambda x: (str(x["benchmark"]), str(x["method"]), int(x["seed"]), int(x["update"])))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen or ["status"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def method_difference_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ["mean", "best@1", "best@3", "best@10", "best@30", "eum_gap", "winner_entropy", "target_regret", "reward_collinearity", "effective_rank"]
    grouped: dict[tuple[str, str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["benchmark"]), str(row["model"]), int(row["seed"]), int(row["update"]))
        grouped[key][str(row["method"])] = row
    out = []
    for (benchmark, model, seed, update), by_method in sorted(grouped.items()):
        vpo = by_method.get("vpo")
        if not vpo:
            continue
        for baseline in ("grpo", "multi_rlvr", "max_at_k", "maxrl"):
            base = by_method.get(baseline)
            if not base:
                continue
            diff_row = {
                "benchmark": benchmark,
                "model": model,
                "seed": seed,
                "update": update,
                "comparison": f"VPO - {baseline}",
            }
            for metric in metrics:
                if vpo.get(metric) in (None, "") or base.get(metric) in (None, ""):
                    continue
                diff_row[f"{metric}_vpo"] = float(vpo[metric])
                diff_row[f"{metric}_{baseline}"] = float(base[metric])
                diff_row[f"{metric}_diff"] = float(vpo[metric]) - float(base[metric])
            out.append(diff_row)
    return out


def diagnostic_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_fields = ["benchmark", "method", "model", "seed", "update", "checkpoint_path"]
    fields = list(base_fields) + [f for f in STAGE2_DIAGNOSTIC_FIELDS if f not in OPTIONAL_BEST_FIELDS]
    return [{field: row.get(field, "") for field in fields} for row in rows]


def write_stage2_summary(path: Path, rows: list[dict[str, Any]], health_rows: list[dict[str, Any]]) -> None:
    completed = len(rows)
    failed_or_missing = 0
    by_benchmark: dict[str, int] = defaultdict(int)
    for row in rows:
        by_benchmark[str(row["benchmark"])] += 1
    lines = [
        "# Stage 2 Short-Training Summary",
        "",
        f"- Completed checkpoint eval rows: {completed}",
        f"- Health rows: {len(health_rows)}",
        f"- Missing rows skipped during aggregation: {failed_or_missing}",
        "",
        "## Rows By Benchmark",
        "",
    ]
    for benchmark, count in sorted(by_benchmark.items()):
        lines.append(f"- {benchmark}: {count}")
    lines.extend(
        [
            "",
            "## Early Patterns",
            "",
            "No paper-level claim is made here. Interpret rows as short-training diagnostics only.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_stage2_tables(manifest_path: Path, output_dir: Path, skip_missing: bool = False) -> dict[str, Path]:
    require_stage2_path = stage2_path_guard()
    out_dir = require_stage2_path(output_dir)
    rows = stage2_rows_from_manifest(manifest_path, skip_missing=skip_missing)
    checkpoint_path = out_dir / "checkpoint_table.csv"
    diagnostic_path = out_dir / "diagnostic_table.csv"
    diff_path = out_dir / "method_difference_table.csv"
    health_path = out_dir / "train_health_table.csv"
    summary_path = out_dir / "final_short_training_summary.md"

    checkpoint_fields = [
        "benchmark",
        "method",
        "model",
        "seed",
        "update",
        "checkpoint_path",
        "train_command",
        "eval_command",
        "artifact_paths",
        "dataset_split",
        "metric_provenance",
        "mean",
        "std",
        "best@1",
        "best@3",
        "best@10",
        "best@30",
        "per_dimension_reward_means",
        "per_dimension_reward_stds",
        "eum_gap",
        "winner_entropy",
        "dominant_candidate_mass",
        "target_regret",
        "base_best_of_k_slope",
        "reward_collinearity",
        "effective_rank",
        "pareto_fraction",
        "attempt_label",
        "job_id",
        "job_summary_path",
    ]
    write_csv_rows(checkpoint_path, rows, checkpoint_fields)
    write_csv_rows(diagnostic_path, diagnostic_table_rows(rows))
    write_csv_rows(diff_path, method_difference_rows(rows))
    try:
        from scripts.collect_train_health import health_rows_from_stage2_manifest

        health_rows = health_rows_from_stage2_manifest(manifest_path)
    except Exception:
        health_rows = []
    health_fields = [
        "benchmark",
        "method",
        "model",
        "seed",
        "update",
        "checkpoint_path",
        "kl_to_reference",
        "grad_norm",
        "clip_fraction",
        "response_length_mean",
        "response_length_std",
        "response_length_clip_ratio",
        "response_aborted_ratio",
        "parse_format_success_rate",
        "train_reward_mean",
        "rollout_reward_mean",
        "train_log_path",
        "rollout_path",
    ]
    write_csv_rows(health_path, health_rows, health_fields)
    write_stage2_summary(summary_path, rows, health_rows)
    return {
        "checkpoint_table": checkpoint_path,
        "diagnostic_table": diagnostic_path,
        "method_difference_table": diff_path,
        "train_health_table": health_path,
        "summary": summary_path,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    else:
        fieldnames = [
        "benchmark",
        "method",
        "model",
        "seed",
        "seed_count",
        "mean",
        "std",
        "checkpoint_path",
        "train_command",
        "eval_command",
        "artifact_paths",
        "metric_provenance",
        "dataset_split",
        "best@1",
        "best@3",
        "best@10",
        "best@30",
        "pre_training_diagnostics",
        "attempt_label",
        "job_kind",
        "resource_provenance",
        "trainer_n_gpus",
        "slurm_partition",
        "slurm_gres",
        "slurm_id",
        "slurm_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("| status |\n| --- |\n| no rows |\n")
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        vals = [str(row.get(h, "")).replace("|", "\\|") for h in headers]
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_table(path: Path, rows: list[dict[str, Any]], fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    elif fmt == "csv":
        write_csv(path, rows)
    elif fmt == "md":
        write_markdown(path, rows)
    else:
        raise ValueError(f"Unsupported output format: {fmt}")


def _flatten_metrics(data: dict[str, Any] | None, prefix: str = "") -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten_metrics(value, f"{name}."))
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[name] = float(value)
    return out


def _json_or_empty(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(str(path)).expanduser()
    if not p.exists():
        return {}
    data = read_json(p)
    return data if isinstance(data, dict) else {}


def _tensor_stats(path: str | Path | None, feature_names: list[Any]) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(str(path)).expanduser()
    if not p.exists():
        return {}
    tensor = np.load(p)
    if tensor.ndim < 2:
        return {}
    flat = tensor.reshape(-1, tensor.shape[-1])
    names = [str(x) for x in feature_names] if feature_names else [f"dim_{i}" for i in range(flat.shape[-1])]
    means = json.dumps(
        {names[i] if i < len(names) else f"dim_{i}": float(flat[:, i].mean()) for i in range(flat.shape[-1])},
        sort_keys=True,
    )
    stds = json.dumps(
        {names[i] if i < len(names) else f"dim_{i}": float(flat[:, i].std()) for i in range(flat.shape[-1])},
        sort_keys=True,
    )
    return {
        "per_dimension_reward_mean": means,
        "per_dimension_reward_std": stds,
        "per_dimension_reward_means": means,
        "per_dimension_reward_stds": stds,
    }


def _best_value(best_at: dict[str, float], key: str) -> float | None:
    if key in best_at:
        return best_at[key]
    if key.replace("@", "@") in best_at:
        return best_at[key.replace("@", "@")]
    return None


def short_training_rows_from_manifest(path: Path, skip_missing: bool = False) -> list[dict[str, Any]]:
    manifest = read_json(path)
    jobs = manifest.get("jobs") if isinstance(manifest, dict) else None
    if not isinstance(jobs, list):
        raise ValueError("Short-training manifest must contain a jobs list.")
    rows: list[dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        for record in job.get("checkpoints", []):
            summary = _json_or_empty(record.get("job_summary_path"))
            meta = {**record, **summary}
            eval_path = Path(str(meta.get("eval_output_path", ""))).expanduser()
            if not eval_path.exists():
                if skip_missing:
                    continue
                raise ValueError(f"Missing eval artifact for {job.get('job_id')} update={record.get('update')}: {eval_path}")
            eval_summary = read_json(eval_path)
            if not isinstance(eval_summary, dict):
                raise ValueError(f"Eval artifact must be JSON object: {eval_path}")
            best_at = best_at_from_eval_summary(eval_summary)
            diag = _json_or_empty(meta.get("diagnostics_path")).get("metrics", {})
            health = _json_or_empty(meta.get("health_path")).get("metrics", {})
            row = {
                "benchmark": meta.get("benchmark"),
                "method": meta.get("method"),
                "model": meta.get("model"),
                "seed": int(meta.get("seed", 0)),
                "update": int(meta.get("update", 0)),
                "checkpoint_path": meta.get("checkpoint_path"),
                "train_command": meta.get("train_command"),
                "eval_command": meta.get("eval_command"),
                "artifact_paths": unique_join(meta.get("artifact_paths", [])),
                "dataset_split": meta.get("dataset_split", "test"),
                "metric_provenance": meta.get("metric_provenance", "scripts/eval.py::short_training"),
                "mean": mean_from_eval_summary(eval_summary),
                "best@1": _best_value(best_at, "best@1"),
                "best@3": _best_value(best_at, "best@3"),
                "best@10": _best_value(best_at, "best@10"),
                "best@30": _best_value(best_at, "best@30"),
                "status": meta.get("status", "UNKNOWN"),
            }
            feature_names = list(eval_summary.get("feature_names", []))
            if not feature_names and isinstance(eval_summary.get("per_objective_mean"), dict):
                feature_names = list(eval_summary["per_objective_mean"].keys())
            row.update(_tensor_stats(meta.get("tensor_path"), feature_names))
            for key, value in _flatten_metrics(diag).items():
                row[key] = value
            for key, value in _flatten_metrics(health).items():
                row[key] = value
            rows.append(row)
    return rows


def diagnostic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "reward_collinearity",
        "effective_rank",
        "pareto_fraction",
        "eum_gap",
        "winner_entropy",
        "dominant_candidate_mass",
        "target_regret",
        "base_best_of_k_slope",
    ]
    base_cols = ["benchmark", "method", "model", "seed", "update", "checkpoint_path"]
    out = []
    for row in rows:
        item = {**{col: row.get(col, "") for col in base_cols}, **{key: row.get(key, "") for key in keys}}
        if item.get("parse_format_success_rate") in ("", None):
            item["parse_format_success_rate"] = row.get("parse_success_rate", row.get("format_success_rate", ""))
        out.append(item)
    return out


def train_health_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "kl_to_reference",
        "grad_norm",
        "clip_fraction",
        "response_length_mean",
        "response_length_std",
        "response_length_clip_ratio",
        "parse_success_rate",
        "format_success_rate",
        "parse_format_success_rate",
        "rollout_response_length_mean",
        "rollout_response_length_std",
    ]
    base_cols = ["benchmark", "method", "model", "seed", "update", "checkpoint_path"]
    out = []
    for row in rows:
        item = {**{col: row.get(col, "") for col in base_cols}, **{key: row.get(key, "") for key in keys}}
        if item.get("parse_format_success_rate") in ("", None):
            item["parse_format_success_rate"] = row.get("parse_success_rate", row.get("format_success_rate", ""))
        out.append(item)
    return out


def method_difference_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "mean",
        "best@1",
        "best@3",
        "best@10",
        "best@30",
        "eum_gap",
        "winner_entropy",
        "target_regret",
        "reward_collinearity",
        "effective_rank",
    ]
    baselines = ("grpo", "multi_rlvr", "max_at_k", "maxrl")
    grouped: dict[tuple[Any, Any, Any, Any], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row.get("benchmark"), row.get("model"), row.get("seed"), row.get("update"))
        grouped[key][str(row.get("method"))] = row
    out: list[dict[str, Any]] = []
    for (benchmark, model, seed, update), methods in sorted(grouped.items()):
        vpo = methods.get("vpo")
        if not vpo:
            continue
        for baseline in baselines:
            other = methods.get(baseline)
            if not other:
                continue
            row = {
                "benchmark": benchmark,
                "model": model,
                "seed": seed,
                "update": update,
                "comparison": f"VPO - {baseline}",
            }
            for metric in metrics:
                if vpo.get(metric) in ("", None) or other.get(metric) in ("", None):
                    row[f"{metric}_diff"] = ""
                else:
                    row[f"{metric}_diff"] = float(vpo[metric]) - float(other[metric])
            out.append(row)
    return out


def write_short_training_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    complete = [row for row in rows if row.get("status") == "PASS"]
    failed = [row for row in rows if row.get("status") not in ("PASS", "UNKNOWN")]
    lines = [
        "# Final Short-Training Summary",
        "",
        f"- Rows aggregated by scripts/eval.py: {len(rows)}",
        f"- Completed rows: {len(complete)}",
        f"- Failed rows: {len(failed)}",
        "- These diagnostics are early-training/plumbing evidence, not final benchmark claims.",
        "",
    ]
    by_benchmark: dict[str, int] = defaultdict(int)
    for row in complete:
        by_benchmark[str(row.get("benchmark"))] += 1
    if by_benchmark:
        lines.append("## Completed Rows By Benchmark")
        lines.append("")
        for key, count in sorted(by_benchmark.items()):
            lines.append(f"- {key}: {count}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def write_short_training_tables(manifest_path: Path, output_dir: Path, skip_missing: bool = False) -> dict[str, Path]:
    require_stage2_path = stage2_path_guard()
    out_dir = require_stage2_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = short_training_rows_from_manifest(manifest_path, skip_missing=skip_missing)
    paths = {
        "checkpoint": out_dir / "checkpoint_table.csv",
        "method_difference": out_dir / "method_difference_table.csv",
        "diagnostic": out_dir / "diagnostic_table.csv",
        "train_health": out_dir / "train_health_table.csv",
        "summary": out_dir / "final_short_training_summary.md",
    }
    write_csv(paths["checkpoint"], rows)
    write_csv(paths["method_difference"], method_difference_rows(rows))
    write_csv(paths["diagnostic"], diagnostic_rows(rows))
    write_csv(paths["train_health"], train_health_rows(rows))
    write_short_training_summary(paths["summary"], rows)
    return paths


def infer_format(path: Path, requested: str | None) -> str:
    if requested:
        return requested
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".md", ".markdown"}:
        return "md"
    return "json"


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="Result manifest JSON.")
    source.add_argument("--job-manifest", help="Launcher job manifest whose eval artifacts have completed.")
    source.add_argument("--stage2-manifest", help="Stage 2 short-training manifest; writes canonical table set.")
    source.add_argument("--short-training-manifest", help="Alias for --stage2-manifest.")
    parser.add_argument("--output", default=None, help="Output result table path, or output directory for Stage 2 tables.")
    parser.add_argument("--output-dir", default=None, help="Output directory for Stage 2 tables.")
    parser.add_argument("--format", choices=["json", "csv", "md"], default=None)
    parser.add_argument("--skip-missing", action="store_true", help="Skip job-manifest entries whose final eval artifact is not present yet.")
    parser.add_argument("--allow_main_experiments", nargs="?", const=True, default=False, type=parse_bool)
    args = parser.parse_args()

    short_manifest = args.stage2_manifest or args.short_training_manifest
    if short_manifest:
        out_dir = Path(args.output_dir or args.output or (repo_root() / "pre_experiments" / "2_short_training" / "tables"))
        outputs = write_short_training_tables(Path(short_manifest), out_dir, skip_missing=args.skip_missing)
        for name, path in outputs.items():
            print(f"Wrote {name}: {path}")
        return

    if not args.output:
        raise SystemExit("--output is required unless a Stage 2 manifest is used.")
    output = require_pre_experiment_path(Path(args.output), args.allow_main_experiments)
    if args.manifest:
        source_rows = read_manifest(Path(args.manifest))
    else:
        source_rows = rows_from_job_manifest(Path(args.job_manifest), skip_missing=args.skip_missing)
    rows = aggregate_rows(source_rows)
    write_table(output, rows, infer_format(output, args.format))
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
