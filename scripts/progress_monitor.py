#!/usr/bin/env python3
"""Progress checkpoint helpers for frozen diagnostic runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.geometry_diagnostics import parse_bool, require_pre_experiment_path


def checkpoint_targets(total: int, percents: list[int]) -> dict[int, int]:
    if total <= 0:
        raise ValueError("total must be positive")
    out: dict[int, int] = {}
    for pct in percents:
        if pct <= 0 or pct > 100:
            raise ValueError(f"checkpoint percent must be in 1..100, got {pct}")
        out[int(pct)] = max(1, min(total, int(math.ceil(total * pct / 100.0))))
    return out


def crossed_checkpoints(
    processed: int,
    total: int,
    percents: list[int],
    completed: set[int] | None = None,
) -> list[int]:
    done = completed or set()
    targets = checkpoint_targets(total, percents)
    return [pct for pct in sorted(targets) if pct not in done and processed >= targets[pct]]


def checkpoint_dir_name(percent: int) -> str:
    return f"run_{int(percent)}%"


def write_checkpoint_marker(
    output_dir: Path,
    *,
    percent: int,
    processed: int,
    total: int,
    phase: str,
    benchmark: str,
    model: str,
) -> Path:
    out_dir = require_pre_experiment_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "checkpoint.json"
    marker.write_text(
        json.dumps(
            {
                "percent": int(percent),
                "processed": int(processed),
                "total": int(total),
                "phase": phase,
                "benchmark": benchmark,
                "model": model,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return marker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=int, required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--percents", default="5,20,100")
    parser.add_argument("--completed", default="")
    parser.add_argument("--allow_main_experiments", nargs="?", const=True, default=False, type=parse_bool)
    args = parser.parse_args()

    percents = [int(x.strip()) for x in args.percents.split(",") if x.strip()]
    completed = {int(x.strip()) for x in args.completed.split(",") if x.strip()}
    print(json.dumps(crossed_checkpoints(args.processed, args.total, percents, completed)))


if __name__ == "__main__":
    main()
