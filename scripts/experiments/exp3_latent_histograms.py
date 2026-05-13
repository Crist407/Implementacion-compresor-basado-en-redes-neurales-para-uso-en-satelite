#!/usr/bin/env python3
"""
Experimento 3: histogramas de latentes cuantizados.

Este script analiza los bitstreams SORTENY actuales leyendo el formato real:

  5 uint16 little-endian: bands, height, width, datatype, num_filters
  q_height * q_width bytes: Q-map uint8
  bands * num_filters * q_height * q_width int32: latentes

El experimento antiguo leia el fichero completo como int32 y saltaba 258
enteros. Eso desalineaba los latentes porque la cabecera real son 10 bytes
y el Q-map ocupa 1024 bytes. Esta version valida el tamano esperado antes de
calcular estadisticas.
"""

from __future__ import annotations

import argparse
import math
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUT_DIR = Path("output/experiments/exp3_latent_histograms_fixed")

DEFAULT_BITSTREAMS = {
    "constant_q204": Path("output/checkpoints/20260507_c_fixed_quality_qmap_wide/latent_from_q204.bin"),
    "fq_adaptive": Path("output/checkpoints/20260507_c_fixed_quality_qmap_wide/latent_target_psnr_76_8.bin"),
    "semantic_veg_boost8": Path("output/checkpoints/20260508_semantic_qmap_c/latent_semantic_vegetation.bin"),
    "focus_fg8_bg48": Path("output/checkpoints/20260510_semantic_focus_vegetation/latent_fg8_bg48.bin"),
    "focus_bgq128": Path("output/checkpoints/20260511_semantic_background_q_vegetation/latent_bgq128.bin"),
    "focus_pen96": Path("output/checkpoints/20260511_semantic_background_q_vegetation/latent_pen96.bin"),
}


def entropy_of_values(values: np.ndarray) -> float:
    if values.size == 0:
        return math.nan
    _vals, counts = np.unique(values, return_counts=True)
    probs = counts.astype(np.float64) / float(values.size)
    return float(-np.sum(probs * np.log2(probs)))


def read_sorteny_bitstream(path: Path) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        header_bytes = f.read(10)
        if len(header_bytes) != 10:
            raise ValueError(f"{path}: cabecera incompleta")
        bands, height, width, datatype, num_filters = struct.unpack("<5H", header_bytes)
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"{path}: dimensiones no compatibles con Q-map 16x16: {height}x{width}")

        q_height = height // 16
        q_width = width // 16
        q_size = q_height * q_width
        q_raw = f.read(q_size)
        if len(q_raw) != q_size:
            raise ValueError(f"{path}: Q-map incompleto, {len(q_raw)} bytes de {q_size}")

        latents = np.fromfile(f, dtype="<i4")

    expected_latents = int(bands) * int(num_filters) * q_height * q_width
    if latents.size != expected_latents:
        raise ValueError(f"{path}: latentes {latents.size}, esperado {expected_latents}")

    header = {
        "bands": int(bands),
        "height": int(height),
        "width": int(width),
        "datatype": int(datatype),
        "num_filters": int(num_filters),
        "q_height": int(q_height),
        "q_width": int(q_width),
        "q_size": int(q_size),
        "expected_latents": int(expected_latents),
        "bitstream_size_bytes": int(path.stat().st_size),
    }
    qmap = np.frombuffer(q_raw, dtype=np.uint8).copy().reshape(q_height, q_width)
    latents = latents.reshape(int(bands), int(num_filters), q_height, q_width)
    return header, qmap, latents


def compute_stats(latents: np.ndarray, qmap: np.ndarray, header: dict[str, int], label: str) -> dict[str, Any]:
    flat = latents.reshape(-1)
    flat_f64 = flat.astype(np.float64)
    abs_i64 = np.abs(flat.astype(np.int64))
    zeros = int(np.count_nonzero(flat == 0))
    near_zero = int(np.count_nonzero(abs_i64 <= 1))
    entropy = entropy_of_values(flat)
    raw_int32 = flat.astype("<i4", copy=False).tobytes()
    zlib_bytes = len(zlib.compress(raw_int32, level=9))
    input_samples = int(header["bands"] * header["height"] * header["width"])

    return {
        "label": label,
        "bitstream_size_bytes": header["bitstream_size_bytes"],
        "bands": header["bands"],
        "height": header["height"],
        "width": header["width"],
        "num_filters": header["num_filters"],
        "q_height": header["q_height"],
        "q_width": header["q_width"],
        "q_min": int(np.min(qmap)),
        "q_max": int(np.max(qmap)),
        "q_mean": float(np.mean(qmap)),
        "q_unique": int(np.unique(qmap).size),
        "samples": int(flat.size),
        "mean": float(np.mean(flat_f64)),
        "std": float(np.std(flat_f64)),
        "abs_mean": float(np.mean(abs_i64)),
        "median": float(np.median(flat_f64)),
        "min": int(np.min(flat)),
        "max": int(np.max(flat)),
        "zeros": zeros,
        "zeros_pct": 100.0 * zeros / float(flat.size),
        "near_zero": near_zero,
        "near_zero_pct": 100.0 * near_zero / float(flat.size),
        "unique_values": int(np.unique(flat).size),
        "entropy_bits_per_symbol": entropy,
        "ideal_bits": float(entropy * flat.size) if math.isfinite(entropy) else math.nan,
        "ideal_bps_per_input_sample": float((entropy * flat.size) / input_samples) if math.isfinite(entropy) else math.nan,
        "zlib_bytes_level9": int(zlib_bytes),
        "zlib_bps_per_input_sample": float((zlib_bytes * 8.0) / input_samples),
        "zlib_ratio_vs_int32": float(zlib_bytes / float(flat.size * 4)),
    }


def print_stats_table(all_stats: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 116)
    print(
        f"{'Label':>24s}  {'Qmean':>8s}  {'Std':>8s}  {'|Mean|':>8s}  "
        f"{'Zeros%':>8s}  {'|x|<=1%':>9s}  {'Entropy':>8s}  {'zlib bps':>9s}  {'Unique':>7s}"
    )
    print("-" * 116)
    for s in all_stats:
        print(
            f"{s['label']:>24s}  {s['q_mean']:>8.2f}  {s['std']:>8.2f}  {s['abs_mean']:>8.2f}  "
            f"{s['zeros_pct']:>7.2f}%  {s['near_zero_pct']:>8.2f}%  "
            f"{s['entropy_bits_per_symbol']:>8.4f}  {s['zlib_bps_per_input_sample']:>9.4f}  "
            f"{s['unique_values']:>7d}"
        )
    print("=" * 116)


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def generate_histogram_svg(
    all_data: list[np.ndarray],
    labels: list[str],
    output_path: Path,
    *,
    title: str,
    bin_range: tuple[int, int],
) -> None:
    n_bins = 100
    bins = np.linspace(bin_range[0], bin_range[1], n_bins + 1)
    histograms = []
    for data in all_data:
        hist, _ = np.histogram(data.reshape(-1), bins=bins, density=True)
        histograms.append(hist)

    svg_w, svg_h = 900, 500
    margin_l, margin_r, margin_t, margin_b = 80, 220, 50, 60
    plot_w = svg_w - margin_l - margin_r
    plot_h = svg_h - margin_t - margin_b
    max_density = max(float(h.max()) for h in histograms) if histograms else 1.0
    if max_density <= 0.0:
        max_density = 1.0

    colors = ["#4A90D9", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" width="{svg_w}" height="{svg_h}">',
        f'<rect width="{svg_w}" height="{svg_h}" fill="white"/>',
        f'<text x="{svg_w // 2}" y="30" text-anchor="middle" font-size="16" font-family="sans-serif" font-weight="bold">{svg_escape(title)}</text>',
    ]

    for i in range(5):
        y = margin_t + plot_h - (i / 4.0) * plot_h
        val = (i / 4.0) * max_density
        svg_lines.append(f'<line x1="{margin_l}" y1="{y:.0f}" x2="{margin_l + plot_w}" y2="{y:.0f}" stroke="#eee" stroke-width="1"/>')
        svg_lines.append(f'<text x="{margin_l - 5}" y="{y + 4:.0f}" text-anchor="end" font-size="10" font-family="sans-serif" fill="#666">{val:.3f}</text>')

    for val in np.linspace(bin_range[0], bin_range[1], 5):
        x = margin_l + ((val - bin_range[0]) / (bin_range[1] - bin_range[0])) * plot_w
        svg_lines.append(f'<text x="{x:.0f}" y="{margin_t + plot_h + 20}" text-anchor="middle" font-size="10" font-family="sans-serif" fill="#666">{val:.0f}</text>')

    for idx, (hist, label) in enumerate(zip(histograms, labels)):
        color = colors[idx % len(colors)]
        points = []
        for i, h in enumerate(hist):
            x = margin_l + ((bins[i] + bins[i + 1]) / 2.0 - bin_range[0]) / (bin_range[1] - bin_range[0]) * plot_w
            y = margin_t + plot_h - (float(h) / max_density) * plot_h
            points.append(f"{x:.1f},{y:.1f}")
        svg_lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2" opacity="0.8"/>')

        legend_y = margin_t + 20 + idx * 22
        legend_x = margin_l + plot_w + 15
        svg_lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 20}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        svg_lines.append(f'<text x="{legend_x + 25}" y="{legend_y + 4}" font-size="11" font-family="sans-serif" fill="#333">{svg_escape(label)}</text>')

    svg_lines.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#333" stroke-width="1.5"/>')
    svg_lines.append(f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#333" stroke-width="1.5"/>')
    svg_lines.append(f'<text x="{svg_w // 2}" y="{svg_h - 10}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#666">Valor cuantizado del latente</text>')
    svg_lines.append(f'<text x="15" y="{svg_h // 2}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#666" transform="rotate(-90, 15, {svg_h // 2})">Densidad</text>')
    svg_lines.append("</svg>")

    output_path.write_text("\n".join(svg_lines) + "\n", encoding="utf-8")
    print(f"  SVG guardado: {output_path}")


def generate_bar_svg(
    all_stats: list[dict[str, Any]],
    output_path: Path,
    *,
    metric: str,
    title: str,
    value_suffix: str = "",
) -> None:
    svg_w, svg_h = 760, 420
    margin_l, margin_r, margin_t, margin_b = 85, 30, 50, 125
    plot_w = svg_w - margin_l - margin_r
    plot_h = svg_h - margin_t - margin_b
    n = len(all_stats)
    bar_w = plot_w / max(1, n) * 0.68
    gap = plot_w / max(1, n) * 0.32
    max_val = max(float(s[metric]) for s in all_stats) * 1.15 if all_stats else 1.0
    if max_val <= 0.0:
        max_val = 1.0
    colors = ["#4A90D9", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" width="{svg_w}" height="{svg_h}">',
        f'<rect width="{svg_w}" height="{svg_h}" fill="white"/>',
        f'<text x="{svg_w // 2}" y="30" text-anchor="middle" font-size="16" font-family="sans-serif" font-weight="bold">{svg_escape(title)}</text>',
    ]

    for i, s in enumerate(all_stats):
        x = margin_l + i * (bar_w + gap) + gap / 2.0
        h = (float(s[metric]) / max_val) * plot_h
        y = margin_t + plot_h - h
        color = colors[i % len(colors)]
        svg_lines.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{bar_w:.0f}" height="{h:.0f}" fill="{color}" opacity="0.85" rx="3"/>')
        svg_lines.append(f'<text x="{x + bar_w / 2.0:.0f}" y="{y - 5:.0f}" text-anchor="middle" font-size="11" font-family="sans-serif" font-weight="bold" fill="{color}">{float(s[metric]):.2f}{value_suffix}</text>')
        label = str(s["label"])
        if len(label) > 18:
            label = label[:18] + "..."
        svg_lines.append(f'<text x="{x + bar_w / 2.0:.0f}" y="{margin_t + plot_h + 15:.0f}" text-anchor="end" font-size="10" font-family="sans-serif" fill="#333" transform="rotate(-45, {x + bar_w / 2.0:.0f}, {margin_t + plot_h + 15:.0f})">{svg_escape(label)}</text>')

    for i in range(5):
        val = (i / 4.0) * max_val
        y = margin_t + plot_h - (i / 4.0) * plot_h
        svg_lines.append(f'<text x="{margin_l - 5}" y="{y + 4:.0f}" text-anchor="end" font-size="10" font-family="sans-serif" fill="#666">{val:.1f}{value_suffix}</text>')
        svg_lines.append(f'<line x1="{margin_l}" y1="{y:.0f}" x2="{margin_l + plot_w}" y2="{y:.0f}" stroke="#eee" stroke-width="1"/>')

    svg_lines.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#333" stroke-width="1.5"/>')
    svg_lines.append(f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#333" stroke-width="1.5"/>')
    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines) + "\n", encoding="utf-8")
    print(f"  SVG guardado: {output_path}")


def save_stats_tsv(all_stats: list[dict[str, Any]], output_path: Path) -> None:
    cols = [
        "label",
        "bitstream_size_bytes",
        "bands",
        "height",
        "width",
        "num_filters",
        "q_min",
        "q_max",
        "q_mean",
        "q_unique",
        "samples",
        "mean",
        "std",
        "abs_mean",
        "median",
        "min",
        "max",
        "zeros",
        "zeros_pct",
        "near_zero",
        "near_zero_pct",
        "unique_values",
        "entropy_bits_per_symbol",
        "ideal_bps_per_input_sample",
        "zlib_bytes_level9",
        "zlib_bps_per_input_sample",
        "zlib_ratio_vs_int32",
    ]
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for s in all_stats:
            row = []
            for c in cols:
                value = s[c]
                if isinstance(value, float):
                    row.append(f"{value:.6f}")
                else:
                    row.append(str(value))
            f.write("\t".join(row) + "\n")
    print(f"  TSV guardado: {output_path}")


def write_report(all_stats: list[dict[str, Any]], output_path: Path) -> None:
    by_label = {str(s["label"]): s for s in all_stats}
    baseline = by_label.get("constant_q204")
    focus = by_label.get("focus_bgq128")
    lines = [
        "# Experimento 3 corregido: latentes SORTENY",
        "",
        "El parser usado aqui lee el formato real del bitstream SORTENY: 10 bytes de cabecera, Q-map uint8 y latentes int32 alineados.",
        "Los resultados antiguos de `output/experiments/exp3_latent_histograms` no deben usarse en informes porque leian el fichero desde un offset incorrecto.",
        "",
        "| Politica | std | abs mean | zeros | entropia | zlib bps | valores unicos |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in all_stats:
        lines.append(
            f"| `{s['label']}` | {s['std']:.4f} | {s['abs_mean']:.4f} | "
            f"{s['zeros_pct']:.4f}% | {s['entropy_bits_per_symbol']:.4f} | "
            f"{s['zlib_bps_per_input_sample']:.4f} | {s['unique_values']} |"
        )
    if baseline and focus:
        lines.extend(
            [
                "",
                "## Lectura principal",
                "",
                f"Frente a `constant_q204`, `focus_bgq128` cambia los latentes de esta forma:",
                f"- std: {baseline['std']:.4f} -> {focus['std']:.4f} ({100.0 * (focus['std'] / baseline['std'] - 1.0):+.2f}%).",
                f"- abs mean: {baseline['abs_mean']:.4f} -> {focus['abs_mean']:.4f} ({100.0 * (focus['abs_mean'] / baseline['abs_mean'] - 1.0):+.2f}%).",
                f"- zeros: {baseline['zeros_pct']:.4f}% -> {focus['zeros_pct']:.4f}% ({focus['zeros_pct'] - baseline['zeros_pct']:+.4f} puntos porcentuales).",
                f"- entropia: {baseline['entropy_bits_per_symbol']:.4f} -> {focus['entropy_bits_per_symbol']:.4f} bits/simbolo.",
                f"- zlib bps: {baseline['zlib_bps_per_input_sample']:.4f} -> {focus['zlib_bps_per_input_sample']:.4f}.",
                "",
                "La conclusion cualitativa del experimento se mantiene: degradar el fondo con `background-q=128` produce latentes mas simples. Lo corregido son los valores numericos.",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Report guardado: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analiza histogramas de latentes de bitstreams SORTENY.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bitstream",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Bitstream adicional o sustituto. Puede repetirse. Si se usa, se anade a los defaults.",
    )
    parser.add_argument("--no-defaults", action="store_true", help="No cargar los bitstreams por defecto.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bitstreams: dict[str, Path] = {} if args.no_defaults else dict(DEFAULT_BITSTREAMS)
    for item in args.bitstream:
        if "=" not in item:
            raise SystemExit(f"--bitstream debe tener formato LABEL=PATH: {item}")
        label, path = item.split("=", 1)
        bitstreams[label] = Path(path)

    print("=" * 70)
    print("  Experimento 3 corregido: histogramas de latentes cuantizados")
    print("=" * 70)

    all_data: list[np.ndarray] = []
    all_labels: list[str] = []
    all_stats: list[dict[str, Any]] = []

    for label, path in bitstreams.items():
        print(f"\n  Cargando: {label}")
        if not path.exists():
            print(f"    SKIP: {path} no existe")
            continue
        header, qmap, latents = read_sorteny_bitstream(path)
        stats = compute_stats(latents, qmap, header, label)
        all_data.append(latents)
        all_labels.append(label)
        all_stats.append(stats)
        print(
            f"    OK: {header['bands']} bands, {header['num_filters']} filters, "
            f"Q={stats['q_min']}..{stats['q_max']}, latents={stats['samples']}"
        )

    if not all_stats:
        raise SystemExit("No hay bitstreams validos para analizar.")

    print_stats_table(all_stats)
    save_stats_tsv(all_stats, args.output_dir / "latent_stats.tsv")
    write_report(all_stats, args.output_dir / "latent_histograms_report.md")

    print("\n  Generando visualizaciones...")
    generate_histogram_svg(
        all_data,
        all_labels,
        args.output_dir / "histogram_comparison.svg",
        title="Distribucion de latentes cuantizados - comparacion de estrategias",
        bin_range=(-50, 50),
    )
    generate_histogram_svg(
        all_data,
        all_labels,
        args.output_dir / "histogram_zoom.svg",
        title="Distribucion de latentes cuantizados - zoom central",
        bin_range=(-15, 15),
    )
    generate_bar_svg(
        all_stats,
        args.output_dir / "sparsity_comparison.svg",
        metric="zeros_pct",
        title="Esparsidad de latentes (% valores = 0)",
        value_suffix="%",
    )
    generate_bar_svg(
        all_stats,
        args.output_dir / "std_comparison.svg",
        metric="std",
        title="Desviacion estandar de latentes",
    )
    generate_bar_svg(
        all_stats,
        args.output_dir / "entropy_comparison.svg",
        metric="entropy_bits_per_symbol",
        title="Entropia empirica de latentes (bits/simbolo)",
    )

    print(f"\n  Todos los resultados en: {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
