#!/usr/bin/env python3
"""Compare Raspberry lambda005 optimized C benchmark against the previous baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


THREADS = 4
FLOAT_TOL = 1e-9
QUALITY_TOL = 1e-6
Q_TOL = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--optimized-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{k: json_ready(v) for k, v in row.items()} for row in rows])


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def to_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 3) -> str:
    number = to_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def pct_delta(new: float, old: float) -> float:
    if not math.isfinite(new) or not math.isfinite(old) or old == 0.0:
        return math.nan
    return 100.0 * (new - old) / old


def speedup(old: float, new: float) -> float:
    if not math.isfinite(old) or not math.isfinite(new) or new == 0.0:
        return math.nan
    return old / new


def load_summary(report_dir: Path) -> list[dict[str, str]]:
    path = report_dir / "raspberry_benchmark_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_csv(path)


def index_threads4(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if to_int(row.get("threads")) != THREADS:
            continue
        out[(row.get("input_id", ""), row.get("case", ""))] = row
    return out


def diff_metric(opt: dict[str, str], base: dict[str, str], key: str) -> tuple[float, float, float]:
    base_v = to_float(base.get(key))
    opt_v = to_float(opt.get(key))
    return base_v, opt_v, opt_v - base_v if math.isfinite(base_v) and math.isfinite(opt_v) else math.nan


def compare_rows(
    baseline: dict[tuple[str, str], dict[str, str]],
    optimized: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    speed_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    thermal_rows: list[dict[str, Any]] = []
    for key in sorted(set(baseline) | set(optimized)):
        input_id, case = key
        base = baseline.get(key)
        opt = optimized.get(key)
        if base is None or opt is None:
            speed_rows.append(
                {
                    "input_id": input_id,
                    "case": case,
                    "status": "missing_baseline" if base is None else "missing_optimized",
                }
            )
            continue

        base_comp, opt_comp, delta_comp = diff_metric(opt, base, "compress_elapsed_s")
        base_dec, opt_dec, delta_dec = diff_metric(opt, base, "decompress_elapsed_s")
        base_total = base_comp + base_dec if math.isfinite(base_comp) and math.isfinite(base_dec) else math.nan
        opt_total = opt_comp + opt_dec if math.isfinite(opt_comp) and math.isfinite(opt_dec) else math.nan
        base_rss, opt_rss, delta_rss = diff_metric(opt, base, "compress_max_rss_kb")
        base_cpu, opt_cpu, delta_cpu = diff_metric(opt, base, "compress_cpu_pct")
        base_temp, opt_temp, delta_temp = diff_metric(opt, base, "temp_max_c")

        speed_rows.append(
            {
                "input_id": input_id,
                "case": case,
                "status": "ok",
                "baseline_compress_s": base_comp,
                "optimized_compress_s": opt_comp,
                "delta_compress_s": delta_comp,
                "compress_speedup": speedup(base_comp, opt_comp),
                "compress_pct_delta": pct_delta(opt_comp, base_comp),
                "baseline_decompress_s": base_dec,
                "optimized_decompress_s": opt_dec,
                "delta_decompress_s": delta_dec,
                "decompress_speedup": speedup(base_dec, opt_dec),
                "decompress_pct_delta": pct_delta(opt_dec, base_dec),
                "baseline_total_s": base_total,
                "optimized_total_s": opt_total,
                "delta_total_s": opt_total - base_total if math.isfinite(base_total) and math.isfinite(opt_total) else math.nan,
                "total_speedup": speedup(base_total, opt_total),
                "total_pct_delta": pct_delta(opt_total, base_total),
                "baseline_compress_cpu_pct": base_cpu,
                "optimized_compress_cpu_pct": opt_cpu,
                "delta_compress_cpu_pct": delta_cpu,
                "baseline_compress_max_rss_kb": base_rss,
                "optimized_compress_max_rss_kb": opt_rss,
                "delta_compress_max_rss_kb": delta_rss,
            }
        )

        q_checks = ["q_min", "q_max", "q_mean", "q_unique"]
        quality_checks = [
            "global_psnr_db",
            "roi_psnr_db",
            "background_psnr_db",
            "bitstream_gzip_bps_per_input_sample",
            "bitstream_bps_per_input_sample",
        ]
        q_ok = True
        quality_ok = True
        q_diffs: dict[str, float] = {}
        quality_diffs: dict[str, float] = {}
        for metric in q_checks:
            _, _, diff = diff_metric(opt, base, metric)
            q_diffs[f"delta_{metric}"] = diff
            if math.isfinite(diff) and abs(diff) > Q_TOL:
                q_ok = False
        for metric in quality_checks:
            _, _, diff = diff_metric(opt, base, metric)
            quality_diffs[f"delta_{metric}"] = diff
            if math.isfinite(diff) and abs(diff) > QUALITY_TOL:
                quality_ok = False

        quality_rows.append(
            {
                "input_id": input_id,
                "case": case,
                "q_parity_ok": q_ok,
                "quality_parity_ok": quality_ok,
                **q_diffs,
                **quality_diffs,
            }
        )

        thermal_rows.append(
            {
                "input_id": input_id,
                "case": case,
                "baseline_temp_max_c": base_temp,
                "optimized_temp_max_c": opt_temp,
                "delta_temp_max_c": delta_temp,
                "baseline_throttled_values": base.get("throttled_values", ""),
                "optimized_throttled_values": opt.get("throttled_values", ""),
                "baseline_compress_max_rss_kb": base_rss,
                "optimized_compress_max_rss_kb": opt_rss,
                "delta_compress_max_rss_kb": delta_rss,
            }
        )
    return speed_rows, quality_rows, thermal_rows


def average(rows: list[dict[str, Any]], key: str) -> float:
    vals = [to_float(row.get(key)) for row in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def group_by_case(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = sorted({str(row.get("case", "")) for row in rows if row.get("status") == "ok"})
    out: list[dict[str, Any]] = []
    for case in cases:
        items = [row for row in rows if row.get("case") == case and row.get("status") == "ok"]
        out.append(
            {
                "case": case,
                "runs": len(items),
                "baseline_compress_s_mean": average(items, "baseline_compress_s"),
                "optimized_compress_s_mean": average(items, "optimized_compress_s"),
                "compress_speedup_mean": average(items, "compress_speedup"),
                "baseline_decompress_s_mean": average(items, "baseline_decompress_s"),
                "optimized_decompress_s_mean": average(items, "optimized_decompress_s"),
                "decompress_speedup_mean": average(items, "decompress_speedup"),
                "baseline_total_s_mean": average(items, "baseline_total_s"),
                "optimized_total_s_mean": average(items, "optimized_total_s"),
                "total_speedup_mean": average(items, "total_speedup"),
                "total_pct_delta_mean": average(items, "total_pct_delta"),
                "rss_delta_kb_mean": average(items, "delta_compress_max_rss_kb"),
            }
        )
    return out


def all_throttled_values(rows: list[dict[str, Any]], key: str) -> str:
    values: set[str] = set()
    for row in rows:
        for item in str(row.get(key, "")).split(";"):
            if item:
                values.add(item)
    return ", ".join(sorted(values)) if values else "sin dato"


def write_report(
    path: Path,
    speed_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    thermal_rows: list[dict[str, Any]],
    grouped: list[dict[str, Any]],
) -> None:
    ok_rows = [row for row in speed_rows if row.get("status") == "ok"]
    quality_failures = [row for row in quality_rows if not row.get("q_parity_ok") or not row.get("quality_parity_ok")]
    avg_total_speedup = average(ok_rows, "total_speedup")
    avg_compress_speedup = average(ok_rows, "compress_speedup")
    avg_decompress_speedup = average(ok_rows, "decompress_speedup")
    lines = [
        "# Raspberry lambda005 optimization comparison",
        "",
        "Comparacion entre el benchmark Raspberry previo y el bundle C optimizado. Solo se usan runs con `threads=4`.",
        "",
        "## Resumen",
        "",
        f"- Pares comparados: `{len(ok_rows)}`.",
        f"- Speedup total medio: `{fmt(avg_total_speedup)}x`.",
        f"- Speedup compresion medio: `{fmt(avg_compress_speedup)}x`.",
        f"- Speedup descompresion medio: `{fmt(avg_decompress_speedup)}x`.",
        f"- Diferencia media de tiempo total: `{fmt(average(ok_rows, 'delta_total_s'))} s`.",
        f"- Diferencia media RSS compresion: `{fmt(average(ok_rows, 'delta_compress_max_rss_kb'), 0)} KB`.",
        f"- Fallos de paridad Q/calidad: `{len(quality_failures)}`.",
        f"- Throttling baseline: `{all_throttled_values(thermal_rows, 'baseline_throttled_values')}`.",
        f"- Throttling optimizado: `{all_throttled_values(thermal_rows, 'optimized_throttled_values')}`.",
        "",
        "## Speedup por modo",
        "",
        "| Modo | runs | comp base | comp opt | speedup comp | decomp base | decomp opt | speedup decomp | total base | total opt | speedup total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grouped:
        lines.append(
            f"| {row['case']} | {row['runs']} | {fmt(row['baseline_compress_s_mean'])} | "
            f"{fmt(row['optimized_compress_s_mean'])} | {fmt(row['compress_speedup_mean'])} | "
            f"{fmt(row['baseline_decompress_s_mean'])} | {fmt(row['optimized_decompress_s_mean'])} | "
            f"{fmt(row['decompress_speedup_mean'])} | {fmt(row['baseline_total_s_mean'])} | "
            f"{fmt(row['optimized_total_s_mean'])} | {fmt(row['total_speedup_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretacion",
            "",
            "- `speedup > 1` significa que el bundle optimizado fue mas rapido.",
            "- La paridad esperada es igualdad de Q-map, PSNR y proxies; el bitstream C sigue teniendo tamano practicamente fijo.",
            "- Si aparece throttling distinto de `0x0`, los tiempos son reales de Raspberry pero deben reportarse como afectados por alimentacion/temperatura.",
        ]
    )
    if quality_failures:
        lines.extend(
            [
                "",
                "## Advertencias de paridad",
                "",
                "| Imagen | Modo | Q parity | Quality parity |",
                "|---|---|---:|---:|",
            ]
        )
        for row in quality_failures:
            lines.append(f"| {row['input_id']} | {row['case']} | {row['q_parity_ok']} | {row['quality_parity_ok']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recommendation(path: Path, speed_rows: list[dict[str, Any]], quality_rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in speed_rows if row.get("status") == "ok"]
    quality_failures = [row for row in quality_rows if not row.get("q_parity_ok") or not row.get("quality_parity_ok")]
    avg_total_speedup = average(ok_rows, "total_speedup")
    decision = "promote_optimized_c" if avg_total_speedup >= 1.02 and not quality_failures else "review_before_promote"
    text = [
        "# Raspberry optimization recommendation",
        "",
        f"- Decision: `{decision}`.",
        f"- Speedup total medio: `{fmt(avg_total_speedup)}x`.",
        f"- Fallos de paridad: `{len(quality_failures)}`.",
        "",
        "La version optimizada debe promocionarse si mantiene paridad funcional y mejora tiempos en Raspberry con `threads=4`.",
        "La recomendacion debe citar throttling si aparece en el benchmark optimizado.",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def svg_speedup(path: Path, grouped: list[dict[str, Any]], key: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1100
    height = 520
    left, right, top, bottom = 80, 30, 55, 165
    vals = [(row["case"], to_float(row.get(key))) for row in grouped]
    vals = [(label, value) for label, value in vals if math.isfinite(value)]
    if not vals:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>\n", encoding="utf-8")
        return
    plot_w = width - left - right
    plot_h = height - top - bottom
    vmax = max(1.1, max(value for _, value in vals) * 1.05)
    vmin = 0.0
    step = plot_w / len(vals)
    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2}' y='30' text-anchor='middle' font-family='sans-serif' font-size='18'>{title}</text>",
        f"<line x1='{left}' y1='{height-bottom}' x2='{width-right}' y2='{height-bottom}' stroke='#333'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{height-bottom}' stroke='#333'/>",
    ]
    y_one = top + (vmax - 1.0) / (vmax - vmin) * plot_h
    lines.append(f"<line x1='{left}' y1='{y_one:.1f}' x2='{width-right}' y2='{y_one:.1f}' stroke='#999' stroke-dasharray='4 4'/>")
    lines.append(f"<text x='{left-8}' y='{y_one+4:.1f}' text-anchor='end' font-family='sans-serif' font-size='11'>1x</text>")
    for idx, (label, value) in enumerate(vals):
        bar_w = max(10.0, step * 0.58)
        x = left + idx * step + (step - bar_w) / 2
        y = top + (vmax - value) / (vmax - vmin) * plot_h
        h = height - bottom - y
        color = "#15803d" if value >= 1.0 else "#b91c1c"
        lines.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{h:.1f}' fill='{color}'/>")
        lines.append(f"<text x='{x+bar_w/2:.1f}' y='{y-5:.1f}' text-anchor='middle' font-family='sans-serif' font-size='10'>{value:.2f}x</text>")
        lines.append(
            f"<text x='{x+bar_w/2:.1f}' y='{height-bottom+12}' transform='rotate(55 {x+bar_w/2:.1f} {height-bottom+12})' "
            f"text-anchor='start' font-family='sans-serif' font-size='9'>{label}</text>"
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = index_threads4(load_summary(args.baseline_report))
    optimized = index_threads4(load_summary(args.optimized_report))
    speed_rows, quality_rows, thermal_rows = compare_rows(baseline, optimized)
    grouped = group_by_case(speed_rows)

    write_csv(args.output_dir / "raspberry_optimization_speedup.csv", speed_rows)
    write_json(args.output_dir / "raspberry_optimization_speedup.json", {"rows": speed_rows, "grouped_by_case": grouped})
    write_csv(args.output_dir / "raspberry_optimization_quality_parity.csv", quality_rows)
    write_csv(args.output_dir / "raspberry_optimization_thermal_summary.csv", thermal_rows)
    write_report(args.output_dir / "raspberry_optimization_comparison_report.md", speed_rows, quality_rows, thermal_rows, grouped)
    write_recommendation(args.output_dir / "raspberry_optimization_recommendation.md", speed_rows, quality_rows)
    svg_speedup(args.output_dir / "figures" / "total_speedup_by_mode.svg", grouped, "total_speedup_mean", "Raspberry optimized C total speedup")
    svg_speedup(args.output_dir / "figures" / "compress_speedup_by_mode.svg", grouped, "compress_speedup_mean", "Raspberry optimized C compress speedup")
    svg_speedup(args.output_dir / "figures" / "decompress_speedup_by_mode.svg", grouped, "decompress_speedup_mean", "Raspberry optimized C decompress speedup")
    print(f"Wrote Raspberry optimization comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
