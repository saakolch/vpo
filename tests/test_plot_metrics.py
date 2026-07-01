from __future__ import annotations

import struct
from pathlib import Path

from scripts.geometry_diagnostics import write_tsv
from scripts.plot_metrics import BG, HEIGHT, WIDTH, draw_text, generate_plots, new_canvas
import scripts.geometry_diagnostics as gd


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    return struct.unpack(">II", data[16:24])


def metric_row(benchmark: str, model: str, prompt_index: int, offset: float) -> dict[str, object]:
    return {
        "row_type": "prompt",
        "benchmark": benchmark,
        "model": model,
        "dataset_split": "train",
        "stage_percent": 100,
        "prompt_index": prompt_index,
        "metric_provenance": "scripts/geometry_diagnostics.py::frozen_geometry_diagnostics_v2",
        "reward_collinearity_active": 0.1 + offset,
        "effective_rank_entropy": 1.0 + offset,
        "effective_rank_participation": 1.2 + offset,
        "unique_pareto_fraction": 0.2 + offset,
        "eum": 0.3 + offset,
        "eum_gap": 0.1 + offset,
        "winner_cluster_entropy_normalized": 0.5 + offset,
        "dominant_cluster_mass": 0.4 + offset,
        "target_regret_fixed": 0.05 + offset,
        "best@1": 0.1 + offset,
        "best@3": 0.2 + offset,
        "best@10": 0.3 + offset,
        "best@30": 0.4 + offset,
    }


def summary_row(benchmark: str, model: str, offset: float) -> dict[str, object]:
    row = metric_row(benchmark, model, 0, offset)
    row.update(
        {
            "row_type": "aggregate",
            "prompt_index": "",
            "verification_command": "scripts/base_geometry_grid.py verify-model",
            "eum": 0.6 + offset,
            "eum_gap": 0.2,
        }
    )
    return row


def test_draw_text_marks_canvas():
    canvas = new_canvas()
    draw_text(canvas, 10, 10, "maze | Qwen3-0.6B", scale=2)

    assert any(canvas[i:i + 3] != bytes(BG) for i in range(0, len(canvas), 3))


def test_plot_metrics_writes_pngs_from_tsv(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    pre = repo / "pre_experiments" / "plots"
    pre.mkdir(parents=True)
    monkeypatch.setattr(gd, "repo_root", lambda: repo)
    metrics = pre / "metrics.tsv"
    write_tsv(
        metrics,
        [
            metric_row("maze", "Qwen/Qwen3-0.6B", 0, 0.0),
            metric_row("tool", "Qwen/Qwen3-1.7B", 1, 0.1),
        ],
    )
    write_tsv(
        pre / "summary.tsv",
        [
            summary_row("maze", "Qwen/Qwen3-0.6B", 0.0),
            summary_row("tool", "Qwen/Qwen3-1.7B", 0.1),
        ],
    )

    written = generate_plots(metrics, pre)

    assert written
    expected = {
        "reward_collinearity_active_distribution.png",
        "effective_rank_entropy_distribution.png",
        "effective_rank_participation_distribution.png",
        "unique_pareto_fraction_histogram.png",
        "eum_distribution.png",
        "eum_gap_delta_set_distribution.png",
        "winner_cluster_entropy_histogram.png",
        "dominant_cluster_mass_histogram.png",
        "target_regret_fixed_distribution.png",
        "best_of_k_slope_curves.png",
    }
    assert {Path(path).name for path in written} == expected
    assert pre / "accuracy_metrics_latest.png" not in written
    assert not (pre / "accuracy_metrics_latest.png").exists()
    for path in written:
        assert png_size(Path(path)) == (WIDTH, HEIGHT)
        assert Path(path).stat().st_size > 10_000
