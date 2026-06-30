#!/usr/bin/env python3
"""Collect Stage 2 training-health metrics from veRL logs and rollout JSONL."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.short_training_common import parse_bool, progress, require_stage2_path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"(?:^|\s)step:(?P<step>\d+)\s+-\s+(?P<body>.*)")


def strip_noise(line: str) -> str:
    text = ANSI_RE.sub("", line)
    text = re.sub(r"^\([^)]* pid=\d+\)\s*", "", text)
    return text.strip()


def parse_value(text: str) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def parse_metric_line(line: str) -> dict[str, Any] | None:
    clean = strip_noise(line)
    match = STEP_RE.search(clean)
    if not match:
        return None
    metrics: dict[str, Any] = {"step": int(match.group("step"))}
    for item in match.group("body").split(" - "):
        if ":" not in item:
            continue
        key, raw = item.rsplit(":", 1)
        value = parse_value(raw.strip())
        if value is not None:
            metrics[key.strip()] = value
    return metrics


def read_log_metrics(run_dir: Path, update: int) -> tuple[dict[str, Any], list[str]]:
    candidates = list((run_dir / "logs").glob("*.log")) + list((run_dir / "runtime_links" / "logs").glob("*.log"))
    if (run_dir / "logs").is_file():
        candidates.append(run_dir / "logs")
    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    for path in progress(sorted(set(candidates)), desc="health logs"):
        if not path.is_file():
            continue
        sources.append(str(path))
        for line in path.read_text(errors="replace").splitlines():
            parsed = parse_metric_line(line)
            if parsed and int(parsed["step"]) <= update:
                rows.append(parsed)
    if not rows:
        return {"log_step": None}, sources
    row = sorted(rows, key=lambda item: int(item["step"]))[-1]
    return {
        "log_step": int(row["step"]),
        "kl_to_reference": row.get("actor/ppo_kl", row.get("actor/kl_loss")),
        "grad_norm": row.get("actor/grad_norm"),
        "clip_fraction": row.get("actor/pg_clipfrac"),
        "response_length_mean": row.get("response_length/mean"),
        "response_length_std": row.get("response_length/std"),
        "response_length_clip_ratio": row.get("response_length/clip_ratio"),
        "reward_mean": row.get("critic/rewards/mean", row.get("critic/score/mean")),
        "pool_best_individual_mean": row.get("pool/best_individual/mean"),
    }, sources


def _bool_from_fields(row: dict[str, Any]) -> float | None:
    if "num_parsed" in row:
        return float(float(row.get("num_parsed") or 0.0) > 0)
    if "num_programs_executed" in row:
        return float(float(row.get("num_programs_executed") or 0.0) > 0)
    if "format" in row:
        return float(float(row.get("format") or 0.0) > 0)
    if "sub_scores" in row:
        return 1.0
    return None


def read_rollout_metrics(run_dir: Path, update: int) -> tuple[dict[str, Any], list[str]]:
    paths = sorted((run_dir / "rollouts").rglob(f"{int(update)}.jsonl"))
    successes: list[float] = []
    format_successes: list[float] = []
    output_lengths: list[int] = []
    sources: list[str] = []
    for path in progress(paths, desc="health rollouts"):
        sources.append(str(path))
        with path.open(errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                parsed = _bool_from_fields(row)
                if parsed is not None:
                    successes.append(parsed)
                if "format" in row:
                    format_successes.append(float(float(row.get("format") or 0.0) > 0))
                output = row.get("output")
                if isinstance(output, str):
                    output_lengths.append(len(output.split()))
    out: dict[str, Any] = {
        "parse_success_rate": float(np.mean(successes)) if successes else None,
        "format_success_rate": float(np.mean(format_successes)) if format_successes else None,
        "parse_format_success_rate": float(np.mean(format_successes or successes)) if (format_successes or successes) else None,
        "rollout_response_length_mean": float(np.mean(output_lengths)) if output_lengths else None,
        "rollout_response_length_std": float(np.std(output_lengths)) if output_lengths else None,
        "rollout_rows": len(successes) if successes else 0,
    }
    return out, sources


def collect(run_dir: Path, update: int) -> dict[str, Any]:
    log_metrics, log_sources = read_log_metrics(run_dir, update)
    rollout_metrics, rollout_sources = read_rollout_metrics(run_dir, update)
    return {
        "schema_version": 1,
        "update": int(update),
        "metrics": {**log_metrics, **rollout_metrics},
        "sources": {"logs": log_sources, "rollouts": rollout_sources},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow_main_experiments", nargs="?", const=True, default=False, type=parse_bool)
    args = parser.parse_args()

    output = require_stage2_path(Path(args.output), args.allow_main_experiments)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(collect(Path(args.run_dir), args.update), indent=2, sort_keys=True) + "\n")
    print(f"Wrote train-health metrics to {output}")


if __name__ == "__main__":
    main()
