#!/usr/bin/env python3
"""Reward-geometry diagnostics for staged VPO benchmark pilots.

The script accepts a reward tensor and writes a JSON diagnostics artifact under
``pre_experiments/`` by default. Accepted tensor shapes are:

* ``(pool, objectives)``
* ``(prompts, pool, objectives)``
* ``(prompts, chains, solutions, objectives)``
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from vpo.utils.eval_metrics import (
    best_of_k_curve,
    dirichlet_weights,
    expected_max_weighted,
    mean_weighted,
    pareto_frac,
)


BEST_AT_K = (1, 3, 10, 30)
PROMPT_METRIC_BATCH_SIZE = 256


def parse_bool(value: str | bool | None) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def normalize_tensor(tensor: np.ndarray) -> np.ndarray:
    arr = np.asarray(tensor, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        pass
    elif arr.ndim == 4:
        arr = arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2], arr.shape[3])
    else:
        raise ValueError(
            "Expected reward tensor with shape (pool,obj), (prompt,pool,obj), "
            "or (prompt,chain,solution,obj)."
        )
    if arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[2] == 0:
        raise ValueError("Reward tensor must have non-empty prompt, pool, and objective axes.")
    if not np.isfinite(arr).all():
        raise ValueError("Reward tensor contains non-finite values.")
    return arr


def reward_collinearity(flat: np.ndarray) -> float:
    matrix = np.asarray(flat, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("reward_collinearity expects a 2D matrix.")
    if matrix.shape[1] < 2:
        return 1.0
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    keep = centered.std(axis=0) > 1e-12
    if int(keep.sum()) < 2:
        return 1.0
    corr = np.corrcoef(centered[:, keep], rowvar=False)
    idx = np.triu_indices(corr.shape[0], k=1)
    if idx[0].size == 0:
        return 1.0
    return float(np.mean(np.abs(corr[idx])))


def effective_rank(flat: np.ndarray) -> float:
    matrix = np.asarray(flat, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    total = float(singular_values.sum())
    if total <= 1e-12:
        return 0.0
    probs = singular_values / total
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return float(math.exp(entropy))


def entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log(probs)))


def prompt_metrics(
    pool: np.ndarray,
    weights: np.ndarray,
    target_weights: np.ndarray,
    seed: int,
) -> dict[str, float]:
    weighted = pool @ weights.T
    winners = np.argmax(weighted, axis=0)
    counts = np.bincount(winners, minlength=pool.shape[0]).astype(np.float64)
    winner_entropy = entropy_from_counts(counts)
    max_entropy = math.log(pool.shape[0]) if pool.shape[0] > 1 else 1.0
    dominant_mass = float(counts.max() / max(1.0, counts.sum()))

    target_scores = pool @ target_weights
    uniform_scores = pool.mean(axis=1)
    target_best = float(target_scores.max())
    uniform_pick = int(np.argmax(uniform_scores))
    target_regret = float(target_best - target_scores[uniform_pick])

    rng = np.random.default_rng(seed)
    perm = rng.permutation(pool.shape[0])
    scalar = pool.mean(axis=1)
    curve = best_of_k_curve(scalar, perm)
    k_hi = min(30, len(curve))
    best_slope = 0.0 if k_hi <= 1 else float((curve[k_hi - 1] - curve[0]) / (k_hi - 1))

    out = {
        "reward_collinearity": reward_collinearity(pool),
        "effective_rank": effective_rank(pool),
        "pareto_fraction": float(pareto_frac(pool)),
        "eum": float(expected_max_weighted(pool, weights)),
        "eum_gap": float(expected_max_weighted(pool, weights) - mean_weighted(pool, weights)),
        "winner_entropy": winner_entropy,
        "winner_entropy_normalized": float(winner_entropy / max_entropy) if max_entropy > 0 else 0.0,
        "dominant_candidate_mass": dominant_mass,
        "target_regret": target_regret,
        "base_best_of_k_slope": best_slope,
        "best_of_k_slope": best_slope,
        "base_best_of_k_slope": best_slope,
    }
    for k in BEST_AT_K:
        if k <= len(curve):
            out[f"best@{k}"] = float(curve[k - 1])
    return out


def bootstrap_eum(
    rewards: np.ndarray,
    weights: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    prompt_eum = _batched_prompt_eum(rewards, weights)
    return bootstrap_eum_from_values(prompt_eum, seed=seed, samples=samples)


def bootstrap_eum_from_values(
    prompt_eum: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    prompt_eum = np.asarray(prompt_eum, dtype=np.float64)
    if prompt_eum.ndim != 1:
        raise ValueError("prompt_eum must be a 1D array.")
    if prompt_eum.size == 0:
        raise ValueError("prompt_eum must be non-empty.")
    if samples <= 0 or prompt_eum.shape[0] == 1:
        return {
            "bootstrap_samples": int(max(samples, 0)),
            "bootstrap_eum_mean": float(prompt_eum.mean()),
            "bootstrap_eum_std": 0.0,
            "bootstrap_stability": 1.0,
        }

    rng = np.random.default_rng(seed)
    n_prompts = prompt_eum.shape[0]
    idx = rng.integers(0, n_prompts, size=(samples, n_prompts))
    arr = prompt_eum[idx].mean(axis=1)
    std = float(arr.std(ddof=0))
    return {
        "bootstrap_samples": int(samples),
        "bootstrap_eum_mean": float(arr.mean()),
        "bootstrap_eum_std": std,
        "bootstrap_stability": float(1.0 / (1.0 + std)),
    }


def _batched_prompt_eum(rewards: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.empty(rewards.shape[0], dtype=np.float64)
    weights_t = weights.T
    for start in range(0, rewards.shape[0], PROMPT_METRIC_BATCH_SIZE):
        end = min(start + PROMPT_METRIC_BATCH_SIZE, rewards.shape[0])
        weighted = rewards[start:end] @ weights_t
        values[start:end] = weighted.max(axis=1).mean(axis=1)
    return values


def batched_prompt_metrics(
    rewards: np.ndarray,
    weights: np.ndarray,
    target_weights: np.ndarray,
    seed: int,
) -> tuple[list[dict[str, float]], np.ndarray]:
    n_prompts, pool_size, _ = rewards.shape
    weights_t = weights.T
    prompt_eum = np.empty(n_prompts, dtype=np.float64)
    prompt_mean_weighted = np.empty(n_prompts, dtype=np.float64)
    winner_entropy = np.empty(n_prompts, dtype=np.float64)
    dominant_mass = np.empty(n_prompts, dtype=np.float64)
    max_entropy = math.log(pool_size) if pool_size > 1 else 1.0

    for start in range(0, n_prompts, PROMPT_METRIC_BATCH_SIZE):
        end = min(start + PROMPT_METRIC_BATCH_SIZE, n_prompts)
        weighted = rewards[start:end] @ weights_t
        prompt_eum[start:end] = weighted.max(axis=1).mean(axis=1)
        prompt_mean_weighted[start:end] = weighted.mean(axis=(1, 2))
        winners = np.argmax(weighted, axis=1)
        for offset, row in enumerate(winners):
            counts = np.bincount(row, minlength=pool_size).astype(np.float64)
            prompt_idx = start + offset
            winner_entropy[prompt_idx] = entropy_from_counts(counts)
            dominant_mass[prompt_idx] = float(counts.max() / max(1.0, counts.sum()))

    target_scores = rewards @ target_weights
    uniform_scores = rewards.mean(axis=2)
    per_prompt: list[dict[str, float]] = []
    for i, pool in enumerate(rewards):
        target_best = float(target_scores[i].max())
        uniform_pick = int(np.argmax(uniform_scores[i]))
        target_regret = float(target_best - target_scores[i, uniform_pick])

        rng = np.random.default_rng(seed + i)
        perm = rng.permutation(pool_size)
        curve = best_of_k_curve(uniform_scores[i], perm)
        k_hi = min(30, len(curve))
        best_slope = 0.0 if k_hi <= 1 else float((curve[k_hi - 1] - curve[0]) / (k_hi - 1))

        row = {
            "reward_collinearity": reward_collinearity(pool),
            "effective_rank": effective_rank(pool),
            "pareto_fraction": float(pareto_frac(pool)),
            "eum": float(prompt_eum[i]),
            "eum_gap": float(prompt_eum[i] - prompt_mean_weighted[i]),
            "winner_entropy": float(winner_entropy[i]),
            "winner_entropy_normalized": float(winner_entropy[i] / max_entropy) if max_entropy > 0 else 0.0,
            "dominant_candidate_mass": float(dominant_mass[i]),
            "target_regret": target_regret,
            "base_best_of_k_slope": best_slope,
            "best_of_k_slope": best_slope,
        }
        for k in BEST_AT_K:
            if k <= len(curve):
                row[f"best@{k}"] = float(curve[k - 1])
        per_prompt.append(row)

    return per_prompt, prompt_eum


def compute_diagnostics(
    tensor: np.ndarray,
    n_weights: int = 2048,
    seed: int = 0,
    bootstrap_samples: int = 200,
    target_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    rewards = normalize_tensor(tensor)
    n_prompts, pool_size, n_obj = rewards.shape
    weights = dirichlet_weights(n_obj, n_weights, seed=seed)
    if target_weights is None:
        target_weights = np.ones(n_obj, dtype=np.float64) / n_obj
    else:
        target_weights = np.asarray(target_weights, dtype=np.float64)
        if target_weights.shape != (n_obj,):
            raise ValueError(f"Expected target weights of shape ({n_obj},), got {target_weights.shape}.")
        total = float(target_weights.sum())
        if total <= 0:
            raise ValueError("Target weights must have positive sum.")
        target_weights = target_weights / total

    flat = rewards.reshape(n_prompts * pool_size, n_obj)
    per_prompt, prompt_eum = batched_prompt_metrics(rewards, weights, target_weights, seed)
    keys = sorted({k for row in per_prompt for k in row})
    aggregate = {
        key: float(np.mean([row[key] for row in per_prompt if key in row]))
        for key in keys
    }
    aggregate.update(
        {
            "reward_collinearity": reward_collinearity(flat),
            "effective_rank": effective_rank(flat),
            "num_prompts": int(n_prompts),
            "pool_size": int(pool_size),
            "num_objectives": int(n_obj),
        }
    )
    aggregate.update(bootstrap_eum_from_values(prompt_eum, seed=seed, samples=bootstrap_samples))
    return {
        "schema_version": 1,
        "metrics": aggregate,
        "per_prompt": per_prompt,
        "settings": {
            "n_weights": int(n_weights),
            "seed": int(seed),
            "bootstrap_samples": int(bootstrap_samples),
            "target_weights": target_weights.tolist(),
        },
    }


PREFERRED_TSV_COLUMNS = [
    "schema_version",
    "row_type",
    "benchmark",
    "model",
    "dataset_split",
    "stage_percent",
    "phase",
    "model_status",
    "snapshot_path",
    "download_command",
    "verification_command",
    "prompt_index",
    "num_prompts",
    "pool_size",
    "num_objectives",
]


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def rows_from_diagnostics(
    diagnostics: dict[str, Any],
    *,
    benchmark: str = "",
    model: str = "",
    dataset_split: str = "",
    stage_percent: int | str = "",
    phase: str = "available",
    model_status: str = "available",
    snapshot_path: str = "",
    download_command: str = "",
    verification_command: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build TSV-ready per-prompt/aggregate rows from diagnostics output."""
    metrics = diagnostics.get("metrics", {})
    settings = diagnostics.get("settings", {})
    common = {
        "schema_version": diagnostics.get("schema_version", 1),
        "benchmark": benchmark,
        "model": model,
        "dataset_split": dataset_split,
        "stage_percent": stage_percent,
        "phase": phase,
        "model_status": model_status,
        "snapshot_path": snapshot_path,
        "download_command": download_command,
        "verification_command": verification_command,
        "n_weights": settings.get("n_weights", ""),
        "seed": settings.get("seed", ""),
        "bootstrap_samples": settings.get("bootstrap_samples", ""),
    }
    rows: list[dict[str, Any]] = []
    for i, prompt_row in enumerate(diagnostics.get("per_prompt", [])):
        row = {
            **common,
            "row_type": "prompt",
            "prompt_index": i,
            "num_prompts": metrics.get("num_prompts", ""),
            "pool_size": metrics.get("pool_size", ""),
            "num_objectives": metrics.get("num_objectives", ""),
        }
        row.update({k: _jsonable_value(v) for k, v in prompt_row.items()})
        rows.append(row)

    aggregate = {
        **common,
        "row_type": "aggregate",
        "prompt_index": "",
    }
    aggregate.update({k: _jsonable_value(v) for k, v in metrics.items()})
    rows.append(aggregate)
    return rows, [aggregate]


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to TSV with stable column ordering."""
    out_path = require_pre_experiment_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = set()
    for row in rows:
        keys.update(row.keys())
    columns = [c for c in PREFERRED_TSV_COLUMNS if c in keys]
    columns.extend(sorted(k for k in keys if k not in columns))
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k, "") for k in columns})


def parse_target_weights(text: str | None) -> np.ndarray | None:
    if not text:
        return None
    return np.asarray([float(x.strip()) for x in text.split(",") if x.strip()], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor", required=True, help="Path to .npy reward tensor.")
    parser.add_argument("--output", required=True, help="Diagnostics JSON path.")
    parser.add_argument("--target-weights", default=None, help="Comma-separated deployment weights.")
    parser.add_argument("--n-weights", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--metrics-tsv", default=None, help="Optional metrics TSV path.")
    parser.add_argument("--summary-tsv", default=None, help="Optional summary TSV path.")
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--dataset-split", default="")
    parser.add_argument("--stage-percent", default="")
    parser.add_argument("--phase", default="available")
    parser.add_argument("--model-status", default="available")
    parser.add_argument("--snapshot-path", default="")
    parser.add_argument("--download-command", default="")
    parser.add_argument("--verification-command", default="")
    parser.add_argument("--allow_main_experiments", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    out_path = require_pre_experiment_path(Path(args.output), args.allow_main_experiments)
    tensor = np.load(args.tensor)
    diagnostics = compute_diagnostics(
        tensor,
        n_weights=args.n_weights,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        target_weights=parse_target_weights(args.target_weights),
    )
    diagnostics["tensor_path"] = str(Path(args.tensor).expanduser())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    if args.metrics_tsv or args.summary_tsv:
        metric_rows, summary_rows = rows_from_diagnostics(
            diagnostics,
            benchmark=args.benchmark,
            model=args.model,
            dataset_split=args.dataset_split,
            stage_percent=args.stage_percent,
            phase=args.phase,
            model_status=args.model_status,
            snapshot_path=args.snapshot_path,
            download_command=args.download_command,
            verification_command=args.verification_command,
        )
        if args.metrics_tsv:
            write_tsv(Path(args.metrics_tsv), metric_rows)
        if args.summary_tsv:
            write_tsv(Path(args.summary_tsv), summary_rows)
    if args.print_json:
        print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
