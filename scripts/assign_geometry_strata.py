#!/usr/bin/env python3
"""Assign reward-geometry strata from per-prompt metrics TSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.geometry_diagnostics import parse_bool, require_pre_experiment_path, write_tsv


METRIC_COLUMNS = [
    "reward_collinearity",
    "effective_rank",
    "pareto_fraction",
    "eum",
    "eum_gap",
    "winner_entropy_normalized",
    "dominant_candidate_mass",
    "target_regret",
    "best_of_k_slope",
    "best@30",
    "bootstrap_stability",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_thresholds(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    prompt_rows = [row for row in rows if row.get("row_type") == "prompt"]
    for key in METRIC_COLUMNS:
        vals = np.asarray([v for row in prompt_rows if (v := _float(row, key)) is not None], dtype=np.float64)
        if vals.size == 0:
            continue
        thresholds[key] = {
            "q25": float(np.quantile(vals, 0.25)),
            "q50": float(np.quantile(vals, 0.50)),
            "q75": float(np.quantile(vals, 0.75)),
        }
    return thresholds


def _ge(row: dict[str, Any], thresholds: dict[str, dict[str, float]], key: str, quantile: str) -> bool:
    value = _float(row, key)
    return value is not None and key in thresholds and value >= thresholds[key][quantile]


def _le(row: dict[str, Any], thresholds: dict[str, dict[str, float]], key: str, quantile: str) -> bool:
    value = _float(row, key)
    return value is not None and key in thresholds and value <= thresholds[key][quantile]


def assign_label(row: dict[str, Any], thresholds: dict[str, dict[str, float]]) -> str:
    if _ge(row, thresholds, "reward_collinearity", "q75") or _le(row, thresholds, "effective_rank", "q25"):
        return "scalar_like"
    if _le(row, thresholds, "winner_entropy_normalized", "q25") and _ge(row, thresholds, "dominant_candidate_mass", "q75"):
        return "dominant_candidate"
    if (
        _ge(row, thresholds, "winner_entropy_normalized", "q75")
        and _ge(row, thresholds, "eum_gap", "q75")
        and _le(row, thresholds, "target_regret", "q25")
    ):
        return "aligned_diversity"
    if _ge(row, thresholds, "winner_entropy_normalized", "q75") and _ge(row, thresholds, "target_regret", "q75"):
        return "off_target_diversity"
    if _ge(row, thresholds, "best_of_k_slope", "q75") or _ge(row, thresholds, "best@30", "q75"):
        return "search_sufficient"
    if _ge(row, thresholds, "pareto_fraction", "q75") and _le(row, thresholds, "bootstrap_stability", "q25"):
        return "noisy_frontier"
    return "unclassified"


def assign_strata(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    thresholds = compute_thresholds(rows)
    out = []
    for row in rows:
        if row.get("row_type") != "prompt":
            continue
        out.append(
            {
                "benchmark": row.get("benchmark", ""),
                "model": row.get("model", ""),
                "dataset_split": row.get("dataset_split", ""),
                "stage_percent": row.get("stage_percent", ""),
                "phase": row.get("phase", ""),
                "model_status": row.get("model_status", ""),
                "prompt_index": row.get("prompt_index", ""),
                "geometry_stratum": assign_label(row, thresholds),
            }
        )
    return out, thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds-output", required=True)
    parser.add_argument("--allow_main_experiments", nargs="?", const=True, default=False, type=parse_bool)
    args = parser.parse_args()

    rows, thresholds = assign_strata(read_tsv(Path(args.metrics)))
    write_tsv(Path(args.output), rows)
    thresholds_path = require_pre_experiment_path(Path(args.thresholds_output), args.allow_main_experiments)
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_path.write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
