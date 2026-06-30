#!/usr/bin/env python3
"""Generate labeled PNG visualizations from frozen diagnostic TSV files."""

from __future__ import annotations

import argparse
import csv
import math
import struct
import zlib
from pathlib import Path

import numpy as np

from scripts.geometry_diagnostics import parse_bool, require_pre_experiment_path


WIDTH = 1600
HEIGHT = 1100
BG = (255, 255, 255)
INK = (31, 41, 55)
MUTED = (107, 114, 128)
GRID = (229, 231, 235)
PALETTE = [
    (37, 99, 235),
    (5, 150, 105),
    (217, 119, 6),
    (147, 51, 234),
    (220, 38, 38),
    (8, 145, 178),
    (79, 70, 229),
    (101, 163, 13),
]

FONT = {
    " ": ["000", "000", "000", "000", "000", "000", "000"],
    "!": ["1", "1", "1", "1", "1", "0", "1"],
    "%": ["10001", "00010", "00100", "01000", "10000", "10001", "00000"],
    "(": ["001", "010", "100", "100", "100", "010", "001"],
    ")": ["100", "010", "001", "001", "001", "010", "100"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    ",": ["00", "00", "00", "00", "00", "10", "10"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ".": ["0", "0", "0", "0", "0", "0", "1"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    ":": ["0", "1", "0", "0", "0", "1", "0"],
    "<": ["0001", "0010", "0100", "1000", "0100", "0010", "0001"],
    "=": ["00000", "11111", "00000", "11111", "00000", "00000", "00000"],
    ">": ["1000", "0100", "0010", "0001", "0010", "0100", "1000"],
    "?": ["01110", "10001", "00001", "00010", "00100", "00000", "00100"],
    "@": ["01110", "10001", "10111", "10101", "10111", "10000", "01111"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    "|": ["1", "1", "1", "1", "1", "1", "1"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


PLOT_SPECS = [
    ("reward_collinearity", "reward_collinearity_distribution.png", "Reward collinearity"),
    ("effective_rank", "effective_rank_distribution.png", "Effective rank"),
    ("pareto_fraction", "pareto_fraction_histogram.png", "Pareto fraction"),
    ("eum", "eum_distribution.png", "Expected utility max"),
    ("eum_gap", "eum_gap_distribution.png", "EUM gap"),
    ("winner_entropy_normalized", "winner_entropy_histogram.png", "Winner entropy"),
    ("dominant_candidate_mass", "dominant_candidate_mass_histogram.png", "Dominant candidate mass"),
    ("target_regret", "target_regret_distribution.png", "Target regret"),
]

ACCURACY_METRICS = ("mean", "best@1", "best@3", "best@10", "best@30")
ACCURACY_PLOT = "accuracy_metrics_latest.png"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _float(value: str | None) -> float | None:
    if value in ("", None):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def new_canvas() -> bytearray:
    return bytearray(bytes(BG) * (WIDTH * HEIGHT))


def set_pixel(canvas: bytearray, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        idx = (y * WIDTH + x) * 3
        canvas[idx:idx + 3] = bytes(color)


def draw_rect(canvas: bytearray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    rgb = bytes(color)
    for y in range(max(0, y0), min(HEIGHT, y1 + 1)):
        offset = y * WIDTH * 3
        for x in range(max(0, x0), min(WIDTH, x1 + 1)):
            idx = offset + x * 3
            canvas[idx:idx + 3] = rgb


def draw_line(canvas: bytearray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], width: int = 2) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    radius = max(0, width // 2)
    while True:
        draw_rect(canvas, x - radius, y - radius, x + radius, y + radius, color)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def glyph_width(ch: str) -> int:
    glyph = FONT.get(ch.upper(), FONT["?"])
    return max(len(row) for row in glyph)


def text_width(text: str, scale: int = 2) -> int:
    total = 0
    for ch in text.upper():
        total += (glyph_width(ch) + 1) * scale
    return max(0, total - scale)


def fit_text(text: str, max_width: int, scale: int = 2) -> str:
    text = text.upper()
    if text_width(text, scale) <= max_width:
        return text
    suffix = ".."
    while len(text) > len(suffix) and text_width(text + suffix, scale) > max_width:
        text = text[:-1]
    return text + suffix


def draw_text(
    canvas: bytearray,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int] = INK,
    *,
    scale: int = 2,
    max_width: int | None = None,
) -> int:
    if max_width is not None:
        text = fit_text(text, max_width, scale)
    cursor = x
    for ch in text.upper():
        glyph = FONT.get(ch, FONT["?"])
        for gy, row in enumerate(glyph):
            for gx, pixel in enumerate(row):
                if pixel == "1":
                    draw_rect(
                        canvas,
                        cursor + gx * scale,
                        y + gy * scale,
                        cursor + (gx + 1) * scale - 1,
                        y + (gy + 1) * scale - 1,
                        color,
                    )
        cursor += (glyph_width(ch) + 1) * scale
    return cursor


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


def prompt_groups(rows: list[dict[str, str]]) -> list[tuple[str, str, list[dict[str, str]]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    order: list[tuple[str, str]] = []
    for row in rows:
        if row.get("row_type") != "prompt":
            continue
        key = (row.get("benchmark", ""), row.get("model", ""))
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)
    return [(benchmark, model, grouped[(benchmark, model)]) for benchmark, model in order]


def values_for(rows: list[dict[str, str]], column: str) -> list[float]:
    values = []
    for row in rows:
        value = _float(row.get(column))
        if value is not None:
            values.append(value)
    return values


def summary_rows_for(rows: list[dict[str, str]], output_dir: Path) -> list[dict[str, str]]:
    summary_path = output_dir / "summary.tsv"
    if summary_path.exists():
        return [row for row in read_tsv(summary_path) if row.get("row_type") == "aggregate"]
    return [row for row in rows if row.get("row_type") == "aggregate"]


def column_values(rows: list[dict[str, str]], column: str) -> list[float]:
    values = []
    for _, _, group_rows in prompt_groups(rows):
        values.extend(values_for(group_rows, column))
    return values


def short_model(model: str) -> str:
    return model.split("/")[-1] if model else "UNKNOWN_MODEL"


def stage_label(rows: list[dict[str, str]], output_dir: Path) -> str:
    for row in rows:
        if row.get("stage_percent"):
            return f"RUN {row['stage_percent']}%"
    return output_dir.name.replace("_", " ").upper()


def split_label(rows: list[dict[str, str]]) -> str:
    for row in rows:
        if row.get("dataset_split"):
            return row["dataset_split"].upper()
    return "UNKNOWN SPLIT"


def fmt_num(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def draw_header(
    canvas: bytearray,
    title: str,
    rows: list[dict[str, str]],
    output_dir: Path,
    note: str,
) -> None:
    groups = prompt_groups(rows)
    prompts = sum(len(group_rows) for _, _, group_rows in groups)
    draw_text(canvas, 34, 28, title, INK, scale=3, max_width=WIDTH - 68)
    subtitle = f"{stage_label(rows, output_dir)} | {split_label(rows)} | PROMPT ROWS BY BENCHMARK / MODEL | GROUPS={len(groups)} | ROWS={prompts}"
    draw_text(canvas, 36, 78, subtitle, MUTED, scale=2, max_width=WIDTH - 72)
    draw_text(canvas, 36, 108, note, MUTED, scale=2, max_width=WIDTH - 72)
    draw_line(canvas, 32, 142, WIDTH - 32, 142, GRID, width=2)


def panel_layout(count: int) -> list[tuple[int, int, int, int]]:
    cols = 2 if count > 1 else 1
    rows = max(1, math.ceil(count / cols))
    left, right, top, bottom = 58, 42, 165, 48
    gap_x, gap_y = 42, 48
    panel_w = (WIDTH - left - right - gap_x * (cols - 1)) // cols
    panel_h = (HEIGHT - top - bottom - gap_y * (rows - 1)) // rows
    boxes = []
    for i in range(count):
        row = i // cols
        col = i % cols
        x0 = left + col * (panel_w + gap_x)
        y0 = top + row * (panel_h + gap_y)
        boxes.append((x0, y0, x0 + panel_w, y0 + panel_h))
    return boxes


def draw_panel_frame(canvas: bytearray, box: tuple[int, int, int, int], title: str, color: tuple[int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    draw_text(canvas, x0, y0, title, color, scale=2, max_width=x1 - x0)
    plot_x0 = x0 + 56
    plot_y0 = y0 + 34
    plot_x1 = x1 - 10
    plot_y1 = y1 - 34
    draw_line(canvas, plot_x0, plot_y1, plot_x1, plot_y1, INK, width=2)
    draw_line(canvas, plot_x0, plot_y0, plot_x0, plot_y1, INK, width=2)
    for frac in (0.25, 0.5, 0.75):
        y = plot_y1 - int((plot_y1 - plot_y0) * frac)
        draw_line(canvas, plot_x0, y, plot_x1, y, GRID, width=1)
    return plot_x0, plot_y0, plot_x1, plot_y1


def accuracy_value(row: dict[str, str], metric: str) -> float | None:
    if metric == "mean":
        value = _float(row.get("mean"))
        if value is not None:
            return value
        eum = _float(row.get("eum"))
        eum_gap = _float(row.get("eum_gap"))
        if eum is not None and eum_gap is not None:
            return eum - eum_gap
        return _float(row.get("best@1"))
    return _float(row.get(metric))


def accuracy_provenance(row: dict[str, str]) -> str:
    return row.get("metric_provenance") or "frozen_geometry_summary"


def accuracy_png(path: Path, rows: list[dict[str, str]], output_dir: Path) -> None:
    summary_rows = summary_rows_for(rows, output_dir)
    canvas = new_canvas()
    draw_text(canvas, 34, 28, "Frozen accuracy / score summary", INK, scale=3, max_width=WIDTH - 68)
    subtitle = f"{stage_label(rows, output_dir)} | {split_label(rows)} | SUMMARY ROWS BY BENCHMARK / MODEL | GROUPS={len(summary_rows)} | ROWS={len(summary_rows)}"
    draw_text(canvas, 36, 78, subtitle, MUTED, scale=2, max_width=WIDTH - 72)
    note = "X=MEAN AND BEST@K SUMMARY COLUMNS | Y=ACCURACY/SCORE | METRIC_PROVENANCE SHOWN IN EACH PANEL"
    draw_text(canvas, 36, 108, note, MUTED, scale=2, max_width=WIDTH - 72)
    draw_line(canvas, 32, 142, WIDTH - 32, 142, GRID, width=2)
    if not summary_rows:
        draw_text(canvas, 80, 180, "NO SUMMARY ROWS FOUND", INK, scale=3)
        write_png(path, canvas)
        return

    all_values = []
    for row in summary_rows:
        all_values.extend(value for metric in ACCURACY_METRICS if (value := accuracy_value(row, metric)) is not None)
    ymax = max(all_values) if all_values else 1.0
    ymax = 1.0 if ymax <= 1.0 else ymax * 1.05

    for i, (row, box) in enumerate(zip(summary_rows, panel_layout(len(summary_rows)))):
        color = PALETTE[i % len(PALETTE)]
        benchmark = row.get("benchmark", "unknown_benchmark")
        model = row.get("model", "unknown_model")
        title = f"{benchmark} | {short_model(model)} | {stage_label([row], output_dir)}"
        x0, y0, x1, y1 = box
        draw_text(canvas, x0, y0, title, color, scale=2, max_width=x1 - x0)
        draw_text(canvas, x0, y0 + 22, f"PROV={accuracy_provenance(row)}", MUTED, scale=1, max_width=x1 - x0)

        plot_x0 = x0 + 58
        plot_y0 = y0 + 54
        plot_x1 = x1 - 12
        plot_y1 = y1 - 42
        draw_line(canvas, plot_x0, plot_y1, plot_x1, plot_y1, INK, width=2)
        draw_line(canvas, plot_x0, plot_y0, plot_x0, plot_y1, INK, width=2)
        for frac in (0.25, 0.5, 0.75):
            y = plot_y1 - int((plot_y1 - plot_y0) * frac)
            draw_line(canvas, plot_x0, y, plot_x1, y, GRID, width=1)
        draw_text(canvas, plot_x0 - 52, plot_y0, fmt_num(ymax), MUTED, scale=1, max_width=50)
        draw_text(canvas, plot_x0 - 52, plot_y0 + 18, "Y=ACC", MUTED, scale=1, max_width=50)

        plot_w = max(1, plot_x1 - plot_x0)
        plot_h = max(1, plot_y1 - plot_y0)
        step = plot_w / len(ACCURACY_METRICS)
        bar_w = max(6, int(step * 0.45))
        for metric_idx, metric in enumerate(ACCURACY_METRICS):
            value = accuracy_value(row, metric)
            center = plot_x0 + int(step * (metric_idx + 0.5))
            if value is not None:
                bar_h = int(plot_h * max(0.0, value) / ymax)
                draw_rect(canvas, center - bar_w // 2, plot_y1 - bar_h, center + bar_w // 2, plot_y1 - 1, color)
            draw_text(canvas, center - 28, plot_y1 + 8, metric, MUTED, scale=1, max_width=58)
    write_png(path, canvas)


def histogram_png(path: Path, rows: list[dict[str, str]], column: str, label: str, output_dir: Path, bins: int = 24) -> None:
    groups = prompt_groups(rows)
    all_values = column_values(rows, column)
    canvas = new_canvas()
    draw_header(canvas, f"{label} distribution", rows, output_dir, f"Each panel: benchmark | model. X={label}; Y=prompt count per bin.")
    if not groups or not all_values:
        draw_text(canvas, 80, 180, "NO PROMPT DATA FOUND", INK, scale=3)
        write_png(path, canvas)
        return

    lo, hi = float(min(all_values)), float(max(all_values))
    if abs(hi - lo) < 1e-12:
        lo -= 0.5
        hi += 0.5

    for i, ((benchmark, model, group_rows), box) in enumerate(zip(groups, panel_layout(len(groups)))):
        color = PALETTE[i % len(PALETTE)]
        vals = np.asarray(values_for(group_rows, column), dtype=np.float64)
        mean = float(vals.mean()) if vals.size else 0.0
        panel_title = f"{benchmark} | {short_model(model)} | N={vals.size} | MEAN={fmt_num(mean)}"
        plot_x0, plot_y0, plot_x1, plot_y1 = draw_panel_frame(canvas, box, panel_title, color)
        if not vals.size:
            draw_text(canvas, plot_x0 + 8, plot_y0 + 12, "NO DATA", MUTED, scale=2)
            continue
        counts, edges = np.histogram(vals, bins=bins, range=(lo, hi))
        max_count = max(int(counts.max()), 1)
        plot_w = max(1, plot_x1 - plot_x0)
        plot_h = max(1, plot_y1 - plot_y0)
        for b, count in enumerate(counts):
            bar_x0 = plot_x0 + 2 + int(plot_w * b / bins)
            bar_x1 = plot_x0 + int(plot_w * (b + 1) / bins) - 2
            bar_h = int(plot_h * int(count) / max_count)
            draw_rect(canvas, bar_x0, plot_y1 - bar_h, max(bar_x0, bar_x1), plot_y1 - 1, color)
        draw_text(canvas, plot_x0, plot_y1 + 8, f"{fmt_num(edges[0])}", MUTED, scale=1)
        draw_text(canvas, plot_x1 - 80, plot_y1 + 8, f"{fmt_num(edges[-1])}", MUTED, scale=1)
        draw_text(canvas, plot_x0 - 50, plot_y0, f"{max_count}", MUTED, scale=1)
        draw_text(canvas, plot_x0 + 72, plot_y1 + 8, "X=VALUE", MUTED, scale=1)
        draw_text(canvas, plot_x0 - 50, plot_y0 + 18, "Y=COUNT", MUTED, scale=1, max_width=48)
    write_png(path, canvas)


def curve_png(path: Path, rows: list[dict[str, str]], output_dir: Path) -> None:
    groups = prompt_groups(rows)
    ks = [1, 3, 10, 30]
    canvas = new_canvas()
    draw_header(canvas, "Best-of-K mean curves (best@K)", rows, output_dir, "Each panel: benchmark | model. X=K candidates; Y=mean best@K over prompt rows.")
    if not groups:
        draw_text(canvas, 80, 180, "NO PROMPT DATA FOUND", INK, scale=3)
        write_png(path, canvas)
        return

    group_values = []
    ymax = 0.0
    for benchmark, model, group_rows in groups:
        values = []
        for k in ks:
            clean = values_for(group_rows, f"best@{k}")
            values.append(float(np.mean(clean)) if clean else 0.0)
        ymax = max(ymax, max(values))
        group_values.append((benchmark, model, values))
    ymax = max(ymax, 1e-9)

    for i, ((benchmark, model, values), box) in enumerate(zip(group_values, panel_layout(len(groups)))):
        color = PALETTE[i % len(PALETTE)]
        panel_title = f"{benchmark} | {short_model(model)}"
        plot_x0, plot_y0, plot_x1, plot_y1 = draw_panel_frame(canvas, box, panel_title, color)
        points = []
        for j, (k, value) in enumerate(zip(ks, values)):
            x = plot_x0 + int((plot_x1 - plot_x0) * j / max(1, len(ks) - 1))
            y = plot_y1 - int((plot_y1 - plot_y0) * value / ymax)
            points.append((x, y))
            draw_text(canvas, x - 10, plot_y1 + 8, f"{k}", MUTED, scale=1)
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            draw_line(canvas, x0, y0, x1, y1, color, width=3)
        for x, y in points:
            draw_rect(canvas, x - 5, y - 5, x + 5, y + 5, color)
        draw_text(canvas, plot_x0, plot_y0 - 14, f"YMAX={fmt_num(ymax)}", MUTED, scale=1)
        draw_text(canvas, plot_x0 + 120, plot_y1 + 8, "K", MUTED, scale=1)
    write_png(path, canvas)


def generate_plots(metrics_tsv: Path, output_dir: Path, *, allow_main_experiments: bool = False) -> list[Path]:
    rows = read_tsv(metrics_tsv)
    out_dir = require_pre_experiment_path(output_dir, allow_main_experiments=allow_main_experiments)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for column, filename, label in PLOT_SPECS:
        path = out_dir / filename
        histogram_png(path, rows, column, label, out_dir)
        written.append(path)
    curve_path = out_dir / "best_of_k_slope_curves.png"
    curve_png(curve_path, rows, out_dir)
    written.append(curve_path)
    accuracy_path = out_dir / ACCURACY_PLOT
    accuracy_png(accuracy_path, rows, out_dir)
    written.append(accuracy_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow_main_experiments", nargs="?", const=True, default=False, type=parse_bool)
    args = parser.parse_args()
    generate_plots(Path(args.metrics), Path(args.output_dir), allow_main_experiments=args.allow_main_experiments)


if __name__ == "__main__":
    main()
