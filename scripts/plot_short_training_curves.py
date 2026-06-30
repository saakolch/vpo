#!/usr/bin/env python3
"""Generate Stage 2 short-training curve PNGs from canonical tables."""

from __future__ import annotations

import argparse
import csv
import math
import struct
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.short_training_common import parse_bool, progress, require_stage2_path


WIDTH = 760
HEIGHT = 440
MARGIN = 48
BG = (255, 255, 255)
INK = (31, 41, 55)
COLORS = [
    (37, 99, 235),
    (5, 150, 105),
    (220, 38, 38),
    (147, 51, 234),
    (234, 88, 12),
]

CHECKPOINT_PLOTS = [
    ("best@1", "best_at_1_over_updates.png"),
    ("best@3", "best_at_3_over_updates.png"),
    ("best@10", "best_at_10_over_updates.png"),
    ("best@30", "best_at_30_over_updates.png"),
    ("winner_entropy", "winner_entropy_over_updates.png"),
    ("eum_gap", "eum_gap_over_updates.png"),
    ("target_regret", "target_regret_over_updates.png"),
    ("reward_collinearity", "reward_collinearity_over_updates.png"),
    ("effective_rank", "effective_rank_over_updates.png"),
]

HEALTH_PLOTS = [
    ("kl_to_reference", "kl_to_reference_over_updates.png"),
    ("grad_norm", "grad_norm_over_updates.png"),
    ("clip_fraction", "clip_fraction_over_updates.png"),
    ("parse_format_success_rate", "parse_format_success_rate_over_updates.png"),
    ("response_length_mean", "response_length_over_updates.png"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def new_canvas() -> bytearray:
    return bytearray(BG * (WIDTH * HEIGHT))


def set_pixel(canvas: bytearray, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        idx = (y * WIDTH + x) * 3
        canvas[idx:idx + 3] = bytes(color)


def draw_rect(canvas: bytearray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    for y in range(max(0, y0), min(HEIGHT, y1 + 1)):
        for x in range(max(0, x0), min(WIDTH, x1 + 1)):
            set_pixel(canvas, x, y, color)


def draw_line(canvas: bytearray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                set_pixel(canvas, x + ox, y + oy, color)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def write_png(path: Path, canvas: bytearray) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    raw = bytearray()
    stride = WIDTH * 3
    for y in range(HEIGHT):
        raw.append(0)
        raw.extend(canvas[y * stride:(y + 1) * stride])
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def grouped_points(rows: list[dict[str, str]], metric: str) -> dict[str, list[tuple[int, float]]]:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = _float(row.get(metric))
        update = _float(row.get("update"))
        method = row.get("method")
        if value is None or update is None or not method:
            continue
        grouped[method][int(update)].append(value)
    out: dict[str, list[tuple[int, float]]] = {}
    for method, by_update in grouped.items():
        out[method] = sorted((update, sum(vals) / len(vals)) for update, vals in by_update.items())
    return out


def curve_png(path: Path, rows: list[dict[str, str]], metric: str) -> None:
    series = grouped_points(rows, metric)
    canvas = new_canvas()
    draw_line(canvas, MARGIN, HEIGHT - MARGIN, WIDTH - MARGIN, HEIGHT - MARGIN, INK)
    draw_line(canvas, MARGIN, MARGIN, MARGIN, HEIGHT - MARGIN, INK)
    all_points = [point for points in series.values() for point in points]
    if not all_points:
        write_png(path, canvas)
        return
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmax = xmin + 1
    if abs(ymax - ymin) < 1e-12:
        ymin -= 0.5
        ymax += 0.5
    plot_w = WIDTH - 2 * MARGIN
    plot_h = HEIGHT - 2 * MARGIN
    for idx, (_method, points) in enumerate(sorted(series.items())):
        color = COLORS[idx % len(COLORS)]
        pixels = []
        for update, value in points:
            x = MARGIN + int(plot_w * (update - xmin) / (xmax - xmin))
            y = HEIGHT - MARGIN - int(plot_h * (value - ymin) / (ymax - ymin))
            pixels.append((x, y))
        for (x0, y0), (x1, y1) in zip(pixels, pixels[1:]):
            draw_line(canvas, x0, y0, x1, y1, color)
        for x, y in pixels:
            draw_rect(canvas, x - 3, y - 3, x + 3, y + 3, color)
    write_png(path, canvas)


def generate_plots(tables_dir: Path, output_dir: Path) -> list[Path]:
    out_dir = require_stage2_path(output_dir)
    checkpoint_rows = read_csv(tables_dir / "checkpoint_table.csv")
    health_rows = read_csv(tables_dir / "train_health_table.csv")
    written: list[Path] = []
    for metric, filename in progress(CHECKPOINT_PLOTS, desc="checkpoint plots"):
        path = out_dir / filename
        curve_png(path, checkpoint_rows, metric)
        written.append(path)
    for metric, filename in progress(HEALTH_PLOTS, desc="health plots"):
        path = out_dir / filename
        curve_png(path, health_rows, metric)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", default="pre_experiments/2_short_training/tables")
    parser.add_argument("--output-dir", default="pre_experiments/2_short_training/plots")
    parser.add_argument("--allow_main_experiments", nargs="?", const=True, default=False, type=parse_bool)
    args = parser.parse_args()
    tables_dir = require_stage2_path(Path(args.tables_dir), args.allow_main_experiments)
    output_dir = require_stage2_path(Path(args.output_dir), args.allow_main_experiments)
    written = generate_plots(tables_dir, output_dir)
    print(f"Wrote {len(written)} plots to {output_dir}")


if __name__ == "__main__":
    main()
