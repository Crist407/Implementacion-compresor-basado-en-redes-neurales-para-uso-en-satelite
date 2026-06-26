#!/usr/bin/env python3
"""Build a local report from a Raspberry C-only lambda005 benchmark checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


BANDS = 8
HEIGHT = 512
WIDTH = 512
BLOCK = 16
Q_SIDE = 32
MAX_U16 = 65535.0
SAMPLES = BANDS * HEIGHT * WIDTH
RAW_BYTES = SAMPLES * 2
QMAP_BYTES = Q_SIDE * Q_SIDE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_meta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key] = value
    out["meta_path"] = str(path)
    # Expected layout: <checkpoint>/<input_id>/runs/<case>/threads_<N>/run_meta.tsv
    # Single-input checkpoints also follow the same suffix below their root.
    try:
        out["image_dir"] = str(path.parents[3])
        out["input_id"] = path.parents[3].name
    except IndexError:
        out["image_dir"] = ""
        out["input_id"] = ""
    return out


def resolve_path(benchmark_dir: Path, value: str, image_dir: Path | None = None) -> Path:
    if not value:
        return Path("")
    raw = Path(value)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([raw, Path.cwd() / raw, benchmark_dir / raw])
        if image_dir is not None:
            candidates.append(image_dir / raw)

    parts = raw.parts
    if benchmark_dir.name in parts:
        idx = parts.index(benchmark_dir.name)
        suffix = Path(*parts[idx + 1 :]) if idx + 1 < len(parts) else Path()
        candidates.append(benchmark_dir / suffix)
    if image_dir is not None and image_dir.name in parts:
        idx = parts.index(image_dir.name)
        suffix = Path(*parts[idx + 1 :]) if idx + 1 < len(parts) else Path()
        candidates.append(image_dir / suffix)

    if raw.name:
        candidates.append(benchmark_dir / "input" / raw.name)
        if image_dir is not None:
            candidates.append(image_dir / "input" / raw.name)

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return raw


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(json_ready(rows))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_ready(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return json_ready(float(obj))
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def psnr(mse: float) -> float:
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((MAX_U16 * MAX_U16) / mse)


def load_raw(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint16)
    if data.size != SAMPLES:
        raise ValueError(f"{path}: {data.size} samples, expected {SAMPLES}")
    return data.reshape(BANDS, HEIGHT, WIDTH)


def load_qmap(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size != QMAP_BYTES:
        raise ValueError(f"{path}: {data.size} bytes, expected {QMAP_BYTES}")
    return data.reshape(Q_SIDE, Q_SIDE)


def metrics_for_arrays(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff * diff))
    return {
        "mse": mse,
        "psnr_db": psnr(mse),
        "mae": float(np.mean(abs_diff)),
        "max_abs": float(np.max(abs_diff)),
        "exact_pct": float(np.mean(abs_diff == 0.0) * 100.0),
    }


def block_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, np.ndarray]:
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    abs_diff = np.abs(diff)
    shaped = diff.reshape(BANDS, Q_SIDE, BLOCK, Q_SIDE, BLOCK)
    abs_shaped = abs_diff.reshape(BANDS, Q_SIDE, BLOCK, Q_SIDE, BLOCK)
    return {
        "mse": np.mean(shaped * shaped, axis=(0, 2, 4)),
        "mae": np.mean(abs_shaped, axis=(0, 2, 4)),
        "max_abs": np.max(abs_shaped, axis=(0, 2, 4)),
    }


def group_metrics(blocks: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    if mask is None or not np.any(mask):
        return {"blocks": 0, "mse": math.nan, "psnr_db": math.nan, "mae": math.nan, "max_abs": math.nan}
    mse = float(np.mean(blocks["mse"][mask]))
    return {
        "blocks": int(np.count_nonzero(mask)),
        "mse": mse,
        "psnr_db": psnr(mse),
        "mae": float(np.mean(blocks["mae"][mask])),
        "max_abs": float(np.max(blocks["max_abs"][mask])),
    }


def load_roi(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if not reader.fieldnames or "semantic_match" not in reader.fieldnames:
            return None
        roi = np.zeros((Q_SIDE, Q_SIDE), dtype=bool)
        for row in reader:
            try:
                by = int(row["block_y"])
                bx = int(row["block_x"])
                roi[by, bx] = int(float(row["semantic_match"])) != 0
            except (KeyError, ValueError, IndexError):
                continue
    return roi


def parse_elapsed(value: str) -> float:
    value = value.strip()
    if not value:
        return math.nan
    parts = value.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(value)
    except ValueError:
        return math.nan


def parse_time_log(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        fields = {
            "User time (seconds):": "user_s",
            "System time (seconds):": "system_s",
            "Percent of CPU this job got:": "cpu_pct",
            "Elapsed (wall clock) time (h:mm:ss or m:ss):": "elapsed_s",
            "Maximum resident set size (kbytes):": "max_rss_kb",
        }
        for prefix, key in fields.items():
            if not stripped.startswith(prefix):
                continue
            value = stripped[len(prefix) :].strip()
            if key == "elapsed_s":
                out[key] = parse_elapsed(value)
            elif key == "cpu_pct":
                out[key] = to_float(value.rstrip("%"))
            else:
                out[key] = to_float(value)
            break
    return out


def parse_thermal(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": path}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="replace")
    temp = re.search(r"temp=([0-9.]+)'C", text)
    throttled = re.search(r"throttled=(0x[0-9a-fA-F]+)", text)
    freq = re.search(r"\n([0-9]{6,})\n", text)
    if temp:
        out["temp_c"] = float(temp.group(1))
    if throttled:
        out["throttled"] = throttled.group(1)
    if freq:
        out["freq_hz"] = int(freq.group(1))
    return out


def to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def stat_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def analyze_run(benchmark_dir: Path, meta: dict[str, str]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    image_dir = Path(meta.get("image_dir", "")) if meta.get("image_dir") else None
    input_path = resolve_path(benchmark_dir, meta.get("input", ""), image_dir)
    qmap_path = resolve_path(benchmark_dir, meta.get("qmap", ""), image_dir)
    summary_tsv = resolve_path(benchmark_dir, meta.get("summary_tsv", ""), image_dir)
    bitstream = resolve_path(benchmark_dir, meta.get("bitstream", ""), image_dir)
    bitstream_gzip = resolve_path(benchmark_dir, meta.get("bitstream_gzip", ""), image_dir)
    recon = resolve_path(benchmark_dir, meta.get("reconstruction", ""), image_dir)
    compress_time = resolve_path(benchmark_dir, meta.get("compress_time_log", ""), image_dir)
    decompress_time = resolve_path(benchmark_dir, meta.get("decompress_time_log", ""), image_dir)

    original = load_raw(input_path)
    reconstructed = load_raw(recon)
    qmap = load_qmap(qmap_path)
    blocks = block_metrics(original, reconstructed)
    global_m = metrics_for_arrays(original, reconstructed)
    roi = load_roi(summary_tsv)
    background = None if roi is None else ~roi
    roi_m = group_metrics(blocks, roi) if roi is not None else group_metrics(blocks, np.zeros((Q_SIDE, Q_SIDE), dtype=bool))
    bg_m = group_metrics(blocks, background) if background is not None else group_metrics(blocks, np.zeros((Q_SIDE, Q_SIDE), dtype=bool))
    ctime = parse_time_log(compress_time)
    dtime = parse_time_log(decompress_time)

    thermal_labels = [
        "thermal_compress_before",
        "thermal_compress_after",
        "thermal_decompress_before",
        "thermal_decompress_after",
    ]
    thermal = {label: parse_thermal(resolve_path(benchmark_dir, meta.get(label, ""), image_dir)) for label in thermal_labels}
    temps = [to_float(v.get("temp_c")) for v in thermal.values()]
    temps = [t for t in temps if math.isfinite(t)]
    throttled_values = [str(v.get("throttled", "")) for v in thermal.values() if v.get("throttled")]

    bitstream_bytes = stat_size(bitstream)
    gzip_bytes = stat_size(bitstream_gzip)
    row = {
        "input_id": meta.get("input_id", ""),
        "image_dir": image_dir or "",
        "case": meta.get("case", ""),
        "threads": int(meta.get("threads", "0") or 0),
        "input_path": input_path,
        "qmap_path": qmap_path,
        "summary_tsv": summary_tsv if summary_tsv.exists() else "",
        "bitstream_path": bitstream,
        "reconstruction_path": recon,
        "input_bytes": stat_size(input_path),
        "qmap_bytes": stat_size(qmap_path),
        "bitstream_bytes": bitstream_bytes,
        "bitstream_bps_per_input_sample": (bitstream_bytes * 8.0) / SAMPLES if bitstream_bytes else math.nan,
        "bitstream_gzip_bytes": gzip_bytes,
        "bitstream_gzip_bps_per_input_sample": (gzip_bytes * 8.0) / SAMPLES if gzip_bytes else math.nan,
        "reconstruction_bytes": stat_size(recon),
        "q_min": int(np.min(qmap)),
        "q_max": int(np.max(qmap)),
        "q_mean": float(np.mean(qmap)),
        "q_unique": int(np.unique(qmap).size),
        "roi_blocks": roi_m["blocks"],
        "roi_pct": float(100.0 * roi_m["blocks"] / QMAP_BYTES) if roi is not None else math.nan,
        "global_psnr_db": global_m["psnr_db"],
        "global_mse": global_m["mse"],
        "global_mae": global_m["mae"],
        "roi_psnr_db": roi_m["psnr_db"],
        "roi_mse": roi_m["mse"],
        "roi_mae": roi_m["mae"],
        "background_psnr_db": bg_m["psnr_db"],
        "background_mse": bg_m["mse"],
        "background_mae": bg_m["mae"],
        "compress_elapsed_s": ctime.get("elapsed_s", math.nan),
        "compress_user_s": ctime.get("user_s", math.nan),
        "compress_system_s": ctime.get("system_s", math.nan),
        "compress_cpu_pct": ctime.get("cpu_pct", math.nan),
        "compress_max_rss_kb": ctime.get("max_rss_kb", math.nan),
        "decompress_elapsed_s": dtime.get("elapsed_s", math.nan),
        "decompress_user_s": dtime.get("user_s", math.nan),
        "decompress_system_s": dtime.get("system_s", math.nan),
        "decompress_cpu_pct": dtime.get("cpu_pct", math.nan),
        "decompress_max_rss_kb": dtime.get("max_rss_kb", math.nan),
        "temp_max_c": max(temps) if temps else math.nan,
        "temp_min_c": min(temps) if temps else math.nan,
        "throttled_values": ";".join(sorted(set(throttled_values))),
        "lambda_value": meta.get("lambda_value", ""),
        "max_lambda": meta.get("max_lambda", ""),
        "meta_path": meta.get("meta_path", ""),
    }
    return row, blocks


def build_comparisons(rows: list[dict[str, Any]], block_cache: dict[tuple[str, str, int], dict[str, np.ndarray]], benchmark_dir: Path) -> list[dict[str, Any]]:
    by_case_thread = {(row["input_id"], row["case"], row["threads"]): row for row in rows}
    out: list[dict[str, Any]] = []
    for row in rows:
        case = str(row["case"])
        if not case.endswith("_focus_bgq128"):
            continue
        summary = Path(str(row.get("summary_tsv", "")))
        roi = load_roi(summary) if summary.exists() else None
        if roi is None or not np.any(roi):
            continue
        bg = ~roi
        input_id = str(row["input_id"])
        focus_blocks = block_cache[(input_id, case, int(row["threads"]))]
        focus_roi = group_metrics(focus_blocks, roi)
        focus_bg = group_metrics(focus_blocks, bg)
        for baseline_name in ["adaptive_s8", "q204"]:
            baseline = by_case_thread.get((input_id, baseline_name, row["threads"]))
            if baseline is None:
                continue
            base_blocks = block_cache[(input_id, baseline_name, int(row["threads"]))]
            base_roi = group_metrics(base_blocks, roi)
            base_bg = group_metrics(base_blocks, bg)
            out.append(
                {
                    "case": case,
                    "input_id": input_id,
                    "baseline": baseline_name,
                    "threads": row["threads"],
                    "roi_pct": row["roi_pct"],
                    "focus_bitstream_bps": row["bitstream_bps_per_input_sample"],
                    "baseline_bitstream_bps": baseline["bitstream_bps_per_input_sample"],
                    "delta_bitstream_bps": row["bitstream_bps_per_input_sample"] - baseline["bitstream_bps_per_input_sample"],
                    "focus_gzip_bps": row["bitstream_gzip_bps_per_input_sample"],
                    "baseline_gzip_bps": baseline["bitstream_gzip_bps_per_input_sample"],
                    "delta_gzip_bps": row["bitstream_gzip_bps_per_input_sample"] - baseline["bitstream_gzip_bps_per_input_sample"],
                    "focus_roi_psnr_db": focus_roi["psnr_db"],
                    "baseline_roi_psnr_db": base_roi["psnr_db"],
                    "delta_roi_psnr_db": focus_roi["psnr_db"] - base_roi["psnr_db"],
                    "focus_background_psnr_db": focus_bg["psnr_db"],
                    "baseline_background_psnr_db": base_bg["psnr_db"],
                    "delta_background_psnr_db": focus_bg["psnr_db"] - base_bg["psnr_db"],
                    "focus_q_mean": row["q_mean"],
                    "baseline_q_mean": baseline["q_mean"],
                    "delta_q_mean": row["q_mean"] - baseline["q_mean"],
                }
            )
    return out


def avg(rows: list[dict[str, Any]], key: str) -> float:
    vals = [to_float(row.get(key)) for row in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else math.nan


def write_report(path: Path, rows: list[dict[str, Any]], comparisons: list[dict[str, Any]], benchmark_dir: Path) -> None:
    throttled = sorted({v for row in rows for v in str(row.get("throttled_values", "")).split(";") if v})
    lines = [
        "# Raspberry lambda005 auto benchmark report",
        "",
        f"Input checkpoint: `{benchmark_dir}`.",
        "",
        "Este informe se calcula en PC/WSL. La Raspberry solo ejecuto binarios C y comandos del sistema.",
        "",
        "## Resumen",
        "",
        f"- Runs analizados: `{len(rows)}`.",
        f"- Imagenes/crops analizados: `{len({row.get('input_id') for row in rows})}`.",
        f"- Comparaciones focus vs baseline: `{len(comparisons)}`.",
        f"- Throttling observado: `{', '.join(throttled) if throttled else 'sin dato'}`.",
        f"- Temp maxima observada: `{fmt(avg(rows, 'temp_max_c'))} C` media de maximos por run.",
        "",
        "## Promedios por caso",
        "",
        "| Caso | runs | bps bitstream | comp s | decomp s | RSS comp KB | PSNR global | ROI % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in sorted({row["case"] for row in rows}):
        items = [row for row in rows if row["case"] == case]
        lines.append(
            f"| {case} | {len(items)} | {fmt(avg(items, 'bitstream_bps_per_input_sample'), 5)} | "
            f"{fmt(avg(items, 'compress_elapsed_s'))} | {fmt(avg(items, 'decompress_elapsed_s'))} | "
            f"{fmt(avg(items, 'compress_max_rss_kb'), 0)} | {fmt(avg(items, 'global_psnr_db'))} | {fmt(avg(items, 'roi_pct'))} |"
        )
    lines.extend(
        [
            "",
            "## Focus vs adaptive_s8",
            "",
            "| Imagen | Caso | threads | delta bitstream bps | delta gzip bps | delta ROI PSNR | delta fondo PSNR | delta Q medio |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        if row["baseline"] != "adaptive_s8":
            continue
        lines.append(
            f"| {row['input_id']} | {row['case']} | {row['threads']} | {fmt(row['delta_bitstream_bps'], 5)} | "
            f"{fmt(row['delta_gzip_bps'], 5)} | "
            f"{fmt(row['delta_roi_psnr_db'])} | {fmt(row['delta_background_psnr_db'])} | {fmt(row['delta_q_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretacion",
            "",
            "- `delta bps < 0` indica bitstream SORTENY C menor que el baseline.",
            "- `delta ROI PSNR >= 0` indica que la ROI se conserva o mejora frente al baseline.",
            "- Si `throttled_values` contiene valores distintos de `0x0`, los tiempos deben citarse con advertencia de alimentacion/temperatura.",
            "- Esta prueba no usa CSMR ni TensorFlow; mide la ruta C embarcable.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    number = to_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def svg_bar(path: Path, rows: list[dict[str, Any]], key: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = [(row, to_float(row.get(key))) for row in rows]
    vals = [(row, val) for row, val in vals if math.isfinite(val)]
    width = max(900, 70 * max(1, len(vals)))
    height = 520
    if not vals:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>\n", encoding="utf-8")
        return
    left, bottom, top, right = 72, 150, 56, 24
    plot_w = width - left - right
    plot_h = height - top - bottom
    vmin = min(0.0, min(v for _, v in vals))
    vmax = max(v for _, v in vals)
    if vmax <= vmin:
        vmax = vmin + 1.0
    step = plot_w / len(vals)
    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2}' y='28' text-anchor='middle' font-family='sans-serif' font-size='18'>{title}</text>",
        f"<line x1='{left}' y1='{height-bottom}' x2='{width-right}' y2='{height-bottom}' stroke='#333'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{height-bottom}' stroke='#333'/>",
    ]
    for i, (row, val) in enumerate(vals):
        bar_w = max(8, step * 0.62)
        x = left + i * step + (step - bar_w) / 2
        y = top + (vmax - val) / (vmax - vmin) * plot_h
        h = height - bottom - y
        color = "#2563eb" if row.get("case") in {"q204", "adaptive_s8"} else "#dc2626"
        lines.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{h:.1f}' fill='{color}'/>")
        label = f"{row.get('case')} t{row.get('threads')}"
        lines.append(
            f"<text x='{x+bar_w/2:.1f}' y='{height-bottom+10}' transform='rotate(60 {x+bar_w/2:.1f} {height-bottom+10})' "
            f"text-anchor='start' font-family='sans-serif' font-size='9'>{label}</text>"
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    benchmark_dir = args.benchmark_dir
    output_dir = args.output_dir or benchmark_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metas = [read_meta(path) for path in sorted((benchmark_dir / "runs").glob("*/*/run_meta.tsv"))]
    if not metas:
        metas = [read_meta(path) for path in sorted(benchmark_dir.glob("*/runs/*/*/run_meta.tsv"))]
    if not metas:
        raise RuntimeError(f"No run_meta.tsv files found under {benchmark_dir}")

    rows: list[dict[str, Any]] = []
    block_cache: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    for meta in metas:
        row, blocks = analyze_run(benchmark_dir, meta)
        rows.append(row)
        block_cache[(str(row["input_id"]), str(row["case"]), int(row["threads"]))] = blocks
    comparisons = build_comparisons(rows, block_cache, benchmark_dir)

    write_csv(output_dir / "raspberry_benchmark_summary.csv", rows)
    write_json(output_dir / "raspberry_benchmark_summary.json", {"rows": rows})
    write_csv(output_dir / "raspberry_focus_comparison.csv", comparisons)
    write_report(output_dir / "raspberry_benchmark_report.md", rows, comparisons, benchmark_dir)
    svg_bar(output_dir / "figures" / "compress_time_by_case.svg", rows, "compress_elapsed_s", "Raspberry compress elapsed seconds")
    svg_bar(output_dir / "figures" / "bitstream_bps_by_case.svg", rows, "bitstream_bps_per_input_sample", "Raspberry SORTENY bitstream bps")
    svg_bar(output_dir / "figures" / "compress_rss_by_case.svg", rows, "compress_max_rss_kb", "Raspberry compress max RSS KB")
    print(f"Wrote Raspberry benchmark report to {output_dir}")


if __name__ == "__main__":
    main()
