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
    dirichlet_weights,
)


BEST_AT_K = (1, 3, 10, 30)
PROMPT_METRIC_BATCH_SIZE = 256
METRIC_PROVENANCE = "scripts/geometry_diagnostics.py::frozen_geometry_diagnostics_v2"
UNIQUE_EPSILON = 1e-12
ACTIVE_TOL = 1e-12


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


def _normalize_weights(weights: np.ndarray, n_obj: int, label: str) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64)
    if arr.shape != (n_obj,):
        raise ValueError(f"Expected {label} weights of shape ({n_obj},), got {arr.shape}.")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError(f"{label} weights must have positive sum.")
    return arr / total


def reward_collinearity(flat: np.ndarray) -> float | None:
    matrix = np.asarray(flat, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("reward_collinearity expects a 2D matrix.")
    if matrix.shape[1] < 2:
        return None
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    keep = centered.std(axis=0) > ACTIVE_TOL
    if int(keep.sum()) < 2:
        return None
    corr = np.corrcoef(centered[:, keep], rowvar=False)
    idx = np.triu_indices(corr.shape[0], k=1)
    if idx[0].size == 0:
        return None
    vals = np.abs(corr[idx])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(np.mean(vals))


def _singular_values(flat: np.ndarray) -> np.ndarray:
    matrix = np.asarray(flat, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    return np.linalg.svd(centered, compute_uv=False)


def effective_rank_entropy(flat: np.ndarray) -> float:
    singular_values = _singular_values(flat)
    total = float(singular_values.sum())
    if total <= ACTIVE_TOL:
        return 0.0
    probs = singular_values / total
    entropy = -float(np.sum(probs * np.log(probs + ACTIVE_TOL)))
    return float(math.exp(entropy))


def effective_rank_participation(flat: np.ndarray) -> float:
    singular_values = _singular_values(flat)
    denom = float(np.sum(singular_values ** 2))
    if denom <= ACTIVE_TOL:
        return 0.0
    total = float(singular_values.sum())
    return float((total * total) / denom)


def effective_rank(flat: np.ndarray) -> float:
    """Compatibility alias for entropy effective rank."""
    return effective_rank_entropy(flat)


def entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log(probs)))


def reward_cluster_ids(pool: np.ndarray, epsilon: float = UNIQUE_EPSILON) -> tuple[np.ndarray, int]:
    if epsilon <= 0:
        rounded = np.asarray(pool, dtype=np.float64)
    else:
        rounded = np.round(np.asarray(pool, dtype=np.float64) / epsilon).astype(np.int64)
    mapping: dict[tuple[Any, ...], int] = {}
    ids = np.empty(pool.shape[0], dtype=np.int64)
    for i, row in enumerate(rounded):
        key = tuple(row.tolist())
        if key not in mapping:
            mapping[key] = len(mapping)
        ids[i] = mapping[key]
    return ids, len(mapping)


def unique_reward_vectors(pool: np.ndarray, epsilon: float = UNIQUE_EPSILON) -> np.ndarray:
    matrix = np.asarray(pool, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("unique_reward_vectors expects a 2D matrix.")
    if epsilon <= 0:
        rounded = matrix
    else:
        rounded = np.round(matrix / epsilon).astype(np.int64)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return matrix[np.sort(idx)]


def pareto_count(matrix: np.ndarray, epsilon: float = UNIQUE_EPSILON) -> int:
    arr = np.asarray(matrix, dtype=np.float64)
    n = arr.shape[0]
    if n == 0:
        return 0
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        diff = arr - arr[i]
        dominates_i = np.all(diff >= -epsilon, axis=1) & np.any(diff > epsilon, axis=1)
        dominates_i[i] = False
        if bool(np.any(dominates_i)):
            dominated[i] = True
    return int((~dominated).sum())


def pareto_fraction(matrix: np.ndarray, epsilon: float = UNIQUE_EPSILON) -> float:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape[0] <= 1:
        return 1.0
    return float(pareto_count(arr, epsilon=epsilon) / arr.shape[0])


def unique_pareto_stats(pool: np.ndarray, epsilon: float = UNIQUE_EPSILON) -> dict[str, float]:
    unique = unique_reward_vectors(pool, epsilon=epsilon)
    unique_count = int(unique.shape[0])
    unique_pareto_count = pareto_count(unique, epsilon=epsilon)
    return {
        "unique_reward_vector_count": float(unique_count),
        "unique_pareto_count": float(unique_pareto_count),
        "unique_pareto_fraction": float(unique_pareto_count / max(1, unique_count)),
        "raw_pareto_fraction": pareto_fraction(pool, epsilon=epsilon),
    }


def winner_cluster_distribution(
    pool: np.ndarray,
    weighted: np.ndarray,
    epsilon: float = UNIQUE_EPSILON,
) -> tuple[float, float, float, int]:
    cluster_ids, n_clusters = reward_cluster_ids(pool, epsilon=epsilon)
    max_scores = weighted.max(axis=0, keepdims=True)
    winners = np.isclose(weighted, max_scores, rtol=0.0, atol=epsilon)
    cluster_wins = np.zeros((n_clusters, weighted.shape[1]), dtype=bool)
    for cluster_idx in range(n_clusters):
        cluster_wins[cluster_idx] = winners[cluster_ids == cluster_idx].any(axis=0)
    ties = cluster_wins.sum(axis=0).astype(np.float64)
    ties[ties <= 0.0] = 1.0
    mass = (cluster_wins / ties).sum(axis=1)
    entropy = entropy_from_counts(mass)
    max_entropy = math.log(n_clusters) if n_clusters > 1 else 1.0
    dominant_mass = float(mass.max() / max(1.0, mass.sum()))
    normalized = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    return entropy, normalized, dominant_mass, n_clusters


def best_of_k_from_order(scores: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(np.asarray(scores, dtype=np.float64))


def expected_support_values(pool: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    weighted = pool @ weights.T
    expected_support = float(weighted.max(axis=0).mean())
    best_mean_support = float((pool @ weights.mean(axis=0)).max())
    return expected_support, float(expected_support - best_mean_support)


def prompt_metrics(
    pool: np.ndarray,
    weights: np.ndarray,
    target_weights: np.ndarray,
    seed: int,
    train_weights: np.ndarray | None = None,
) -> dict[str, float | None]:
    del seed  # Candidate order is the stored generation order for Layer A.
    if train_weights is None:
        train_weights = np.ones(pool.shape[1], dtype=np.float64) / pool.shape[1]
    weighted = pool @ weights.T
    eum, eum_gap = expected_support_values(pool, weights)
    winner_entropy, winner_entropy_normalized, dominant_mass, n_clusters = winner_cluster_distribution(pool, weighted)

    target_scores = pool @ target_weights
    train_scores = pool @ train_weights
    target_best = float(target_scores.max())
    train_pick = int(np.argmax(train_scores))
    target_regret = float(target_best - target_scores[train_pick])

    curve = best_of_k_from_order(train_scores)
    k_hi = min(30, len(curve))
    best_slope = 0.0 if k_hi <= 1 else float((curve[k_hi - 1] - curve[0]) / (k_hi - 1))
    collinearity = reward_collinearity(pool)
    pareto = unique_pareto_stats(pool)

    out = {
        "reward_collinearity": collinearity,
        "reward_collinearity_active": collinearity,
        "reward_collinearity_defined": 1.0 if collinearity is not None else 0.0,
        "effective_rank": effective_rank_entropy(pool),
        "effective_rank_entropy": effective_rank_entropy(pool),
        "effective_rank_participation": effective_rank_participation(pool),
        "pareto_fraction": pareto["unique_pareto_fraction"],
        **pareto,
        "eum": eum,
        "expected_support_mean": eum,
        "eum_gap": eum_gap,
        "expected_support_delta_set": eum_gap,
        "winner_entropy": winner_entropy,
        "winner_cluster_entropy": winner_entropy,
        "winner_entropy_normalized": winner_entropy_normalized,
        "winner_cluster_entropy_normalized": winner_entropy_normalized,
        "dominant_candidate_mass": dominant_mass,
        "dominant_cluster_mass": dominant_mass,
        "reward_cluster_count": float(n_clusters),
        "target_regret": target_regret,
        "target_regret_fixed": target_regret,
        "base_best_of_k_slope": best_slope,
        "best_of_k_slope": best_slope,
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
    values, _ = _batched_prompt_expected_support(rewards, weights)
    return values


def _batched_prompt_expected_support(rewards: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.empty(rewards.shape[0], dtype=np.float64)
    gaps = np.empty(rewards.shape[0], dtype=np.float64)
    weights_t = weights.T
    mean_weight = weights.mean(axis=0)
    for start in range(0, rewards.shape[0], PROMPT_METRIC_BATCH_SIZE):
        end = min(start + PROMPT_METRIC_BATCH_SIZE, rewards.shape[0])
        weighted = rewards[start:end] @ weights_t
        values[start:end] = weighted.max(axis=1).mean(axis=1)
        gaps[start:end] = values[start:end] - (rewards[start:end] @ mean_weight).max(axis=1)
    return values, gaps


def batched_prompt_metrics(
    rewards: np.ndarray,
    weights: np.ndarray,
    target_weights: np.ndarray,
    train_weights: np.ndarray,
    seed: int,
) -> tuple[list[dict[str, float | None]], np.ndarray]:
    n_prompts, pool_size, _ = rewards.shape
    prompt_eum, prompt_eum_gap = _batched_prompt_expected_support(rewards, weights)
    per_prompt: list[dict[str, float | None]] = []
    for i, pool in enumerate(rewards):
        row = prompt_metrics(pool, weights, target_weights, seed + i, train_weights=train_weights)
        row["eum"] = float(prompt_eum[i])
        row["expected_support_mean"] = float(prompt_eum[i])
        row["eum_gap"] = float(prompt_eum_gap[i])
        row["expected_support_delta_set"] = float(prompt_eum_gap[i])
        per_prompt.append(row)

    return per_prompt, prompt_eum


def _mean_prompt_metric(rows: list[dict[str, float | None]], key: str) -> float | None:
    vals = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            vals.append(numeric)
    if not vals:
        return None
    return float(np.mean(vals))


def compute_diagnostics(
    tensor: np.ndarray,
    n_weights: int = 2048,
    seed: int = 0,
    bootstrap_samples: int = 200,
    target_weights: np.ndarray | None = None,
    train_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    rewards = normalize_tensor(tensor)
    n_prompts, pool_size, n_obj = rewards.shape
    weights = dirichlet_weights(n_obj, n_weights, seed=seed)
    if target_weights is None:
        target_weights = np.ones(n_obj, dtype=np.float64) / n_obj
    else:
        target_weights = _normalize_weights(target_weights, n_obj, "Target")
    if train_weights is None:
        train_weights = np.ones(n_obj, dtype=np.float64) / n_obj
    else:
        train_weights = _normalize_weights(train_weights, n_obj, "Train")

    flat = rewards.reshape(n_prompts * pool_size, n_obj)
    per_prompt, prompt_eum = batched_prompt_metrics(rewards, weights, target_weights, train_weights, seed)
    keys = sorted({k for row in per_prompt for k in row})
    aggregate = {key: _mean_prompt_metric(per_prompt, key) for key in keys}
    global_collinearity = reward_collinearity(flat)
    aggregate.update(
        {
            "reward_collinearity_global_active": global_collinearity,
            "effective_rank_global_entropy": effective_rank_entropy(flat),
            "effective_rank_global_participation": effective_rank_participation(flat),
            "num_prompts": int(n_prompts),
            "pool_size": int(pool_size),
            "num_objectives": int(n_obj),
        }
    )
    aggregate.update(bootstrap_eum_from_values(prompt_eum, seed=seed, samples=bootstrap_samples))
    return {
        "schema_version": 2,
        "metric_provenance": METRIC_PROVENANCE,
        "metrics": aggregate,
        "per_prompt": per_prompt,
        "settings": {
            "n_weights": int(n_weights),
            "seed": int(seed),
            "bootstrap_samples": int(bootstrap_samples),
            "target_weights": target_weights.tolist(),
            "train_weights": train_weights.tolist(),
            "unique_epsilon": UNIQUE_EPSILON,
            "metric_provenance": METRIC_PROVENANCE,
        },
    }


PREFERRED_TSV_COLUMNS = [
    "schema_version",
    "row_type",
    "benchmark",
    "model",
    "dataset_split",
    "stage_percent",
    "metric_provenance",
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
        "metric_provenance": diagnostics.get("metric_provenance", settings.get("metric_provenance", "")),
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


def parse_train_weights(text: str | None) -> np.ndarray | None:
    return parse_target_weights(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor", required=True, help="Path to .npy reward tensor.")
    parser.add_argument("--output", required=True, help="Diagnostics JSON path.")
    parser.add_argument("--target-weights", default=None, help="Comma-separated deployment weights.")
    parser.add_argument("--train-weights", default=None, help="Comma-separated train scalarization weights.")
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
        train_weights=parse_train_weights(args.train_weights),
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
