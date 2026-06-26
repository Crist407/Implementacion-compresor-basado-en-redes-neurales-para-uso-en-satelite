#!/usr/bin/env python3
"""Build a local report for the Raspberry lambda005 Q-map cost benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_RPI_REPORT = Path("output/checkpoints/20260614_raspberry_lambda005_optimized_benchmark_report")
DEFAULT_OUT = Path("output/checkpoints/20260701_raspberry_lambda005_qmap_cost_report")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qmap-checkpoint", type=Path, required=True)
    parser.add_argument("--raspberry-report", type=Path, default=DEFAULT_RPI_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_meta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key] = value
    out["meta_path"] = str(path)
    try:
        out["image_dir"] = str(path.parents[3])
        out.setdefault("input_id", path.parents[3].name)
    except IndexError:
        out.setdefault("image_dir", "")
    return out


def resolve_path(checkpoint: Path, value: str, image_dir: Path | None = None) -> Path:
    if not value:
        return Path("")
    raw = Path(value)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([raw, Path.cwd() / raw, checkpoint / raw])
        if image_dir is not None:
            candidates.append(image_dir / raw)

    parts = raw.parts
    if checkpoint.name in parts:
        idx = parts.index(checkpoint.name)
        suffix = Path(*parts[idx + 1 :]) if idx + 1 < len(parts) else Path()
        candidates.append(checkpoint / suffix)
    if image_dir is not None and image_dir.name in parts:
        idx = parts.index(image_dir.name)
        suffix = Path(*parts[idx + 1 :]) if idx + 1 < len(parts) else Path()
        candidates.append(image_dir / suffix)

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return raw


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
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


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
    fields = {
        "User time (seconds):": "user_s",
        "System time (seconds):": "system_s",
        "Percent of CPU this job got:": "cpu_pct",
        "Elapsed (wall clock) time (h:mm:ss or m:ss):": "elapsed_s",
        "Maximum resident set size (kbytes):": "max_rss_kb",
    }
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
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
    out: dict[str, Any] = {"path": str(path)}
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


def stat_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def median(values: list[float]) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return math.nan
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def percentile(values: list[float], pct: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[int(pos)]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def fmt(value: Any, digits: int = 3) -> str:
    number = to_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def cost_bucket(pct_of_compress: float) -> str:
    if not math.isfinite(pct_of_compress):
        return "unknown"
    if pct_of_compress < 5.0:
        return "negligible"
    if pct_of_compress < 15.0:
        return "small_measurable"
    return "needs_attention"


def load_pipeline_rows(report_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    summary = report_dir / "raspberry_benchmark_summary.csv"
    if not summary.exists():
        raise FileNotFoundError(f"Missing optimized Raspberry summary: {summary}")
    rows = read_csv(summary)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if to_int(row.get("threads")) != 4:
            continue
        input_id = row.get("input_id", "")
        case = row.get("case", "")
        compress = to_float(row.get("compress_elapsed_s"))
        decompress = to_float(row.get("decompress_elapsed_s"))
        row["compress_elapsed_s"] = compress
        row["decompress_elapsed_s"] = decompress
        row["total_elapsed_s"] = compress + decompress if math.isfinite(compress) and math.isfinite(decompress) else math.nan
        out[(input_id, case)] = row
    if not out:
        raise RuntimeError(f"No threads=4 rows found in {summary}")
    return out


def analyze_qmap_run(checkpoint: Path, meta: dict[str, str], pipeline: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    image_dir = Path(meta.get("image_dir", "")) if meta.get("image_dir") else None
    qmap_path = resolve_path(checkpoint, meta.get("qmap", ""), image_dir)
    summary_path = resolve_path(checkpoint, meta.get("summary_tsv", ""), image_dir)
    time_log = resolve_path(checkpoint, meta.get("qmap_time_log", ""), image_dir)
    thermal_before = parse_thermal(resolve_path(checkpoint, meta.get("thermal_qmap_before", ""), image_dir))
    thermal_after = parse_thermal(resolve_path(checkpoint, meta.get("thermal_qmap_after", ""), image_dir))
    timing = parse_time_log(time_log)
    temps = [to_float(thermal_before.get("temp_c")), to_float(thermal_after.get("temp_c"))]
    temps = [t for t in temps if math.isfinite(t)]
    throttled_values = sorted(
        {
            str(v)
            for v in [thermal_before.get("throttled"), thermal_after.get("throttled")]
            if v
        }
    )

    input_id = meta.get("input_id", "")
    case = meta.get("case", "")
    pipe = pipeline.get((input_id, case), {})
    q_elapsed = to_float(timing.get("elapsed_s"))
    compress = to_float(pipe.get("compress_elapsed_s"))
    decompress = to_float(pipe.get("decompress_elapsed_s"))
    total = to_float(pipe.get("total_elapsed_s"))
    pct_compress = (100.0 * q_elapsed / compress) if math.isfinite(q_elapsed) and math.isfinite(compress) and compress > 0 else math.nan
    pct_total = (100.0 * q_elapsed / total) if math.isfinite(q_elapsed) and math.isfinite(total) and total > 0 else math.nan

    return {
        "input_id": input_id,
        "case": case,
        "repeat": to_int(meta.get("repeat")),
        "threads": to_int(meta.get("threads")),
        "command_type": meta.get("command_type", ""),
        "preset": meta.get("preset", ""),
        "threshold": meta.get("threshold", ""),
        "tier": meta.get("tier", ""),
        "qmap_elapsed_s": q_elapsed,
        "qmap_user_s": to_float(timing.get("user_s")),
        "qmap_system_s": to_float(timing.get("system_s")),
        "qmap_cpu_pct": to_float(timing.get("cpu_pct")),
        "qmap_max_rss_kb": to_float(timing.get("max_rss_kb")),
        "qmap_bytes": stat_size(qmap_path),
        "summary_tsv_bytes": stat_size(summary_path),
        "temp_min_c": min(temps) if temps else math.nan,
        "temp_max_c": max(temps) if temps else math.nan,
        "throttled_values": ";".join(throttled_values),
        "compress_elapsed_s": compress,
        "decompress_elapsed_s": decompress,
        "total_elapsed_s": total,
        "qmap_pct_of_compress": pct_compress,
        "qmap_pct_of_total": pct_total,
        "cost_bucket": cost_bucket(pct_compress),
        "pipeline_match": bool(pipe),
        "qmap_path": qmap_path,
        "summary_tsv": summary_path,
        "qmap_time_log": time_log,
        "meta_path": meta.get("meta_path", ""),
    }


def aggregate(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)

    out: list[dict[str, Any]] = []
    for group_key, items in sorted(groups.items()):
        elapsed = [to_float(row.get("qmap_elapsed_s")) for row in items]
        pct_comp = [to_float(row.get("qmap_pct_of_compress")) for row in items]
        pct_total = [to_float(row.get("qmap_pct_of_total")) for row in items]
        rss = [to_float(row.get("qmap_max_rss_kb")) for row in items]
        cpu = [to_float(row.get("qmap_cpu_pct")) for row in items]
        row = {key: group_key[i] for i, key in enumerate(keys)}
        row.update(
            {
                "runs": len(items),
                "matched_pipeline_runs": sum(1 for item in items if item.get("pipeline_match")),
                "qmap_elapsed_mean_s": mean(elapsed),
                "qmap_elapsed_median_s": median(elapsed),
                "qmap_elapsed_p95_s": percentile(elapsed, 95.0),
                "qmap_pct_of_compress_mean": mean(pct_comp),
                "qmap_pct_of_total_mean": mean(pct_total),
                "qmap_max_rss_mean_kb": mean(rss),
                "qmap_cpu_pct_mean": mean(cpu),
                "cost_bucket": cost_bucket(mean(pct_comp)),
            }
        )
        out.append(row)
    return out


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    by_case: list[dict[str, Any]],
    overall: dict[str, Any],
    qmap_checkpoint: Path,
    raspberry_report: Path,
) -> None:
    throttled = sorted({v for row in rows for v in str(row.get("throttled_values", "")).split(";") if v})
    lines = [
        "# Raspberry lambda005 Q-map cost report",
        "",
        f"Q-map checkpoint: `{qmap_checkpoint}`.",
        f"Pipeline Raspberry report: `{raspberry_report}`.",
        "",
        "Aquest informe quantifica el cost de generar els Q-map respecte al cost de compressio i descompressio ja mesurat amb la versio C optimitzada.",
        "",
        "## Resum",
        "",
        f"- Execucions Q-map analitzades: `{len(rows)}`.",
        f"- Casos amb comparacio contra pipeline de 4 threads: `{sum(1 for row in rows if row.get('pipeline_match'))}`.",
        f"- Temps mitja Q-map: `{fmt(overall.get('qmap_elapsed_mean_s'))} s`.",
        f"- Temps mitja Q-map / compressio: `{fmt(overall.get('qmap_pct_of_compress_mean'))}%`.",
        f"- Temps mitja Q-map / compressio+descompressio: `{fmt(overall.get('qmap_pct_of_total_mean'))}%`.",
        f"- Interpretacio global: `{overall.get('cost_bucket', 'unknown')}`.",
        f"- Throttling observat: `{', '.join(throttled) if throttled else 'sense dada'}`.",
        "",
        "Regla de lectura: per sota del 5% del temps de compressio, el cost del preset es considera despreciable; entre 5% i 15%, petit pero mesurable; per sobre del 15%, caldria optimitzar o justificar.",
        "",
        "## Cost per cas",
        "",
        "| Cas | tipus | runs | Q-map s mitja | Q-map p95 s | % compressio | % total | RSS KB | lectura |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in by_case:
        lines.append(
            f"| {row.get('case')} | {row.get('command_type')} | {row.get('runs')} | "
            f"{fmt(row.get('qmap_elapsed_mean_s'))} | {fmt(row.get('qmap_elapsed_p95_s'))} | "
            f"{fmt(row.get('qmap_pct_of_compress_mean'))} | {fmt(row.get('qmap_pct_of_total_mean'))} | "
            f"{fmt(row.get('qmap_max_rss_mean_kb'), 0)} | {row.get('cost_bucket')} |"
        )

    lines.extend(
        [
            "",
            "## Interpretacio",
            "",
            "- Aquesta prova nomes executa `sorteny_fq_qmap` i `sorteny_semantic_qmap`; no executa compressor ni descompressor.",
            "- `q204` i `adaptive_s8` mesuren el cost de generar Q-map global/adaptatiu a partir de calibracio.",
            "- Els presets semantics mesuren el cost addicional de llegir la imatge, calcular la mascara ROI i escriure el Q-map.",
            "- La columna `% compressio` compara el temps del Q-map amb el temps de `sorteny_compressor` del mateix `input_id + case` amb 4 threads.",
            "- La columna `% total` compara amb `sorteny_compressor + sorteny_decompressor`; aquesta lectura es rellevant per la ruta fixed-quality quan cal validar qualitat objectiu.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def svg_bar(path: Path, rows: list[dict[str, Any]], key: str, title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = [(str(row.get("case", "")), to_float(row.get(key))) for row in rows]
    vals = [(name, val) for name, val in vals if math.isfinite(val)]
    width = max(900, 72 * max(1, len(vals)))
    height = 520
    if not vals:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>\n", encoding="utf-8")
        return
    left, right, top, bottom = 72, 24, 56, 170
    plot_w = width - left - right
    plot_h = height - top - bottom
    vmax = max(val for _, val in vals)
    if vmax <= 0.0:
        vmax = 1.0
    step = plot_w / len(vals)
    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2}' y='28' text-anchor='middle' font-family='sans-serif' font-size='18'>{title}</text>",
        f"<text x='18' y='{top+plot_h/2}' transform='rotate(-90 18 {top+plot_h/2})' text-anchor='middle' font-family='sans-serif' font-size='12'>{ylabel}</text>",
        f"<line x1='{left}' y1='{height-bottom}' x2='{width-right}' y2='{height-bottom}' stroke='#333'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{height-bottom}' stroke='#333'/>",
    ]
    for i, (name, val) in enumerate(vals):
        bar_w = max(8.0, step * 0.62)
        x = left + i * step + (step - bar_w) / 2
        h = val / vmax * plot_h
        y = height - bottom - h
        color = "#2563eb" if name in {"q204", "adaptive_s8"} else "#16a34a"
        lines.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{h:.1f}' fill='{color}'/>")
        lines.append(f"<text x='{x+bar_w/2:.1f}' y='{y-4:.1f}' text-anchor='middle' font-family='sans-serif' font-size='10'>{val:.2f}</text>")
        lines.append(
            f"<text x='{x+bar_w/2:.1f}' y='{height-bottom+10}' transform='rotate(60 {x+bar_w/2:.1f} {height-bottom+10})' "
            f"text-anchor='start' font-family='sans-serif' font-size='9'>{name}</text>"
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    qmap_checkpoint = args.qmap_checkpoint
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = load_pipeline_rows(args.raspberry_report)
    meta_paths = sorted(qmap_checkpoint.glob("*/runs/*/repeat_*/run_meta.tsv"))
    if not meta_paths:
        meta_paths = sorted((qmap_checkpoint / "runs").glob("*/repeat_*/run_meta.tsv"))
    if not meta_paths:
        raise RuntimeError(f"No Q-map run_meta.tsv files found under {qmap_checkpoint}")

    rows = [analyze_qmap_run(qmap_checkpoint, read_meta(path), pipeline) for path in meta_paths]
    by_case = aggregate(rows, ["case", "command_type", "preset", "threshold", "tier"])
    by_case_type = aggregate(rows, ["command_type"])
    overall_rows = aggregate(rows, [])
    overall = overall_rows[0] if overall_rows else {}

    write_csv(output_dir / "raspberry_qmap_cost_rows.csv", rows)
    write_json(output_dir / "raspberry_qmap_cost_rows.json", {"rows": rows})
    write_csv(output_dir / "raspberry_qmap_cost_by_case.csv", by_case)
    write_csv(output_dir / "raspberry_qmap_cost_by_type.csv", by_case_type)
    write_json(
        output_dir / "raspberry_qmap_cost_summary.json",
        {
            "qmap_checkpoint": qmap_checkpoint,
            "raspberry_report": args.raspberry_report,
            "overall": overall,
            "by_case": by_case,
            "by_type": by_case_type,
        },
    )
    write_report(
        output_dir / "raspberry_qmap_cost_report.md",
        rows,
        by_case,
        overall,
        qmap_checkpoint,
        args.raspberry_report,
    )
    svg_bar(
        output_dir / "figures" / "qmap_elapsed_by_case.svg",
        by_case,
        "qmap_elapsed_mean_s",
        "Q-map generation elapsed time by case",
        "seconds",
    )
    svg_bar(
        output_dir / "figures" / "qmap_pct_of_compress_by_case.svg",
        by_case,
        "qmap_pct_of_compress_mean",
        "Q-map cost as percent of compression",
        "% of compressor",
    )
    print(f"Wrote Raspberry Q-map cost report to {output_dir}")


if __name__ == "__main__":
    main()
