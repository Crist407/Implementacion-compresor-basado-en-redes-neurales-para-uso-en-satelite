#!/usr/bin/env python3
"""
Valida target PSNR -> target MSE -> Q-map -> reconstruccion real en dataset.

El script es auxiliar de evidencia: orquesta binarios C y scripts de apoyo,
pero la generacion de Q-map final sigue ocurriendo en `sorteny_fq_qmap`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MAX_U16 = 65535.0
DEFAULT_TARGETS = [74.5, 75.0, 75.5, 76.0, 76.5, 76.8, 77.0, 77.5]
WIDE_Q_VALUES = [128, 144, 160, 176, 192, 204, 216, 232, 240, 248, 255]


@dataclass(frozen=True)
class CropInfo:
    path: Path
    name: str
    semantic_matches: int | None = None


def psnr_to_mse(psnr_db: float) -> float:
    return (MAX_U16 * MAX_U16) / (10.0 ** (psnr_db / 10.0))


def psnr_from_mse(mse: float) -> float:
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((MAX_U16 * MAX_U16) / mse)


def finite_float(value: float) -> float | str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    return float(value)


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return finite_float(float(obj))
    if isinstance(obj, float):
        return finite_float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def executable(path: Path) -> str:
    if path.is_absolute():
        return str(path)
    if path.parent == Path(".") or str(path.parent) == ".":
        return f"./{path.name}"
    return str(path)


def run_cmd(cmd: list[str], cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    elapsed = time.perf_counter() - t0
    log_path.write_text(
        "\n".join(
            [
                "$ " + " ".join(cmd),
                f"exit_code={proc.returncode}",
                f"elapsed_s={elapsed:.6f}",
                "",
                "== STDOUT ==",
                proc.stdout,
                "== STDERR ==",
                proc.stderr,
            ]
        ),
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}; see {log_path}")
    return elapsed


def load_raw_u16(path: Path, bands: int, height: int, width: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint16)
    expected = bands * height * width
    if data.size != expected:
        raise ValueError(f"{path}: raw samples {data.size}, expected {expected}")
    return data.reshape(bands, height, width)


def load_qmap(path: Path, q_height: int, q_width: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    expected = q_height * q_width
    if data.size != expected:
        raise ValueError(f"{path}: qmap bytes {data.size}, expected {expected}")
    return data.reshape(q_height, q_width)


def metrics_for_arrays(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff * diff))
    return {
        "mse": mse,
        "psnr_db": psnr_from_mse(mse),
        "mae": float(np.mean(abs_diff)),
        "max_abs": float(np.max(abs_diff)),
        "exact_pct": float(np.mean(abs_diff == 0.0) * 100.0),
    }


def block_mse_array(original: np.ndarray, reconstructed: np.ndarray, block_size: int) -> np.ndarray:
    _, height, width = original.shape
    bh = height // block_size
    bw = width // block_size
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    out = np.zeros((bh, bw), dtype=np.float64)
    for by in range(bh):
        y0 = by * block_size
        y1 = y0 + block_size
        for bx in range(bw):
            x0 = bx * block_size
            x1 = x0 + block_size
            d = diff[:, y0:y1, x0:x1]
            out[by, bx] = float(np.mean(d * d))
    return out


def qmap_stats(qmap: np.ndarray) -> dict[str, Any]:
    return {
        "q_min": int(np.min(qmap)),
        "q_max": int(np.max(qmap)),
        "q_mean": float(np.mean(qmap)),
        "q_unique": int(np.unique(qmap).size),
        "q_128_blocks": int(np.sum(qmap == 128)),
        "q_255_blocks": int(np.sum(qmap == 255)),
        "qmap_bytes": int(qmap.size),
    }


def parse_fq_summary(path: Path) -> dict[str, Any]:
    counts = {
        "reachable_blocks": 0,
        "too_relaxed_blocks": 0,
        "too_strict_blocks": 0,
        "invalid_blocks": 0,
        "adaptive_budget_blocks": 0,
        "clamped_low_blocks": 0,
        "clamped_high_blocks": 0,
        "fallback_blocks": 0,
    }
    predicted: list[float] = []
    best: list[float] = []
    worst: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t")
        for row in reader:
            viability = row.get("viability", "")
            reason = row.get("reason", "")
            if viability == "reachable":
                counts["reachable_blocks"] += 1
            elif viability == "too_relaxed":
                counts["too_relaxed_blocks"] += 1
            elif viability == "too_strict":
                counts["too_strict_blocks"] += 1
            elif viability == "invalid":
                counts["invalid_blocks"] += 1
            elif viability == "adaptive_budget":
                counts["adaptive_budget_blocks"] += 1
            if reason == "clamped_low_q":
                counts["clamped_low_blocks"] += 1
            elif reason == "clamped_high_q":
                counts["clamped_high_blocks"] += 1
            elif reason == "fallback_baseline":
                counts["fallback_blocks"] += 1
            for key, bucket in [
                ("predicted_mse", predicted),
                ("best_mse_qmax", best),
                ("worst_mse_qmin", worst),
            ]:
                value = row.get(key)
                if value not in (None, ""):
                    bucket.append(float(value))
    total = sum(counts[k] for k in ["reachable_blocks", "too_relaxed_blocks", "too_strict_blocks", "invalid_blocks", "adaptive_budget_blocks"])
    return {
        **counts,
        "blocks": total,
        "reachable_pct": (counts["reachable_blocks"] / total * 100.0) if total else math.nan,
        "too_relaxed_pct": (counts["too_relaxed_blocks"] / total * 100.0) if total else math.nan,
        "too_strict_pct": (counts["too_strict_blocks"] / total * 100.0) if total else math.nan,
        "predicted_mse_mean": float(np.mean(predicted)) if predicted else math.nan,
        "predicted_psnr_from_mean_mse": psnr_from_mse(float(np.mean(predicted))) if predicted else math.nan,
        "best_mse_qmax_mean": float(np.mean(best)) if best else math.nan,
        "worst_mse_qmin_mean": float(np.mean(worst)) if worst else math.nan,
    }


def validate_raw_dataset(data_dir: Path, bands: int, height: int, width: int, max_crops: int | None) -> list[CropInfo]:
    raw_files = sorted(p for p in data_dir.glob("*.raw") if not p.name.endswith(":Zone.Identifier"))
    expected_bytes = bands * height * width * 2
    if max_crops is not None:
        raw_files = raw_files[:max_crops]
    crops: list[CropInfo] = []
    for p in raw_files:
        if p.stat().st_size != expected_bytes:
            raise ValueError(f"{p}: size {p.stat().st_size}, expected {expected_bytes}")
        crops.append(CropInfo(path=p, name=p.stem))
    return crops


def load_semantic_matches(summary_csv: Path) -> dict[str, int]:
    if not summary_csv.exists():
        return {}
    out: dict[str, int] = {}
    for row in read_csv(summary_csv):
        stem = Path(row["file"]).stem
        out[stem] = int(float(row["semantic_matches"]))
    return out


def attach_matches(crops: list[CropInfo], matches: dict[str, int]) -> list[CropInfo]:
    return [CropInfo(path=c.path, name=c.name, semantic_matches=matches.get(c.name)) for c in crops]


def select_recalibration_crops(crops: list[CropInfo], count: int) -> list[CropInfo]:
    if count <= 0 or not crops:
        return []
    ranked = sorted(crops, key=lambda c: ((c.semantic_matches if c.semantic_matches is not None else -1), c.name))
    if count >= len(ranked):
        return ranked
    positions = np.linspace(0, len(ranked) - 1, count)
    selected: list[CropInfo] = []
    seen: set[str] = set()
    for pos in positions:
        idx = int(round(float(pos)))
        crop = ranked[idx]
        if crop.name not in seen:
            selected.append(crop)
            seen.add(crop.name)
    for crop in ranked:
        if len(selected) >= count:
            break
        if crop.name not in seen:
            selected.append(crop)
            seen.add(crop.name)
    return selected


RESULT_FIELDS = [
    "mode",
    "crop",
    "file",
    "target_psnr",
    "target_mse",
    "achieved_psnr",
    "achieved_mse",
    "psnr_error_db",
    "mse_error",
    "abs_psnr_error_db",
    "q_mean",
    "q_min",
    "q_max",
    "q_unique",
    "q_128_blocks",
    "q_255_blocks",
    "qmap_bytes",
    "reachable_blocks",
    "too_relaxed_blocks",
    "too_strict_blocks",
    "invalid_blocks",
    "reachable_pct",
    "too_relaxed_pct",
    "too_strict_pct",
    "predicted_mse_mean",
    "predicted_psnr_from_mean_mse",
    "compress_s",
    "decompress_s",
    "qmap_s",
    "analysis_s",
    "monotonic_rank",
]


def existing_keys(results_csv: Path) -> set[tuple[str, str, str]]:
    if not results_csv.exists():
        return set()
    out: set[tuple[str, str, str]] = set()
    for row in read_csv(results_csv):
        out.add((row["mode"], row["crop"], row["target_psnr"]))
    return out


def run_target_case(
    *,
    root: Path,
    mode: str,
    crop: CropInfo,
    target_psnr: float,
    calibration: Path,
    outdir: Path,
    args: argparse.Namespace,
    env: dict[str, str],
    keep_artifacts: str,
) -> dict[str, Any]:
    case_name = f"{mode}_{crop.name}_psnr_{target_psnr:.1f}".replace(".", "p")
    case_dir = outdir / "cases" / mode / crop.name / f"psnr_{target_psnr:.1f}".replace(".", "p")
    logs_dir = outdir / "logs" / mode / crop.name
    qmap = case_dir / "qmap.bin"
    q_summary = case_dir / "qmap_summary.tsv"
    latent = case_dir / "latent.bin"
    recon = case_dir / "reconstructed.raw"
    case_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    cmd = [
        executable(args.fq_bin),
        "--calibration",
        str(calibration),
        "--target-psnr",
        f"{target_psnr:.8g}",
        "--output-qmap",
        str(qmap),
        "--summary-tsv",
        str(q_summary),
    ]
    qmap_s = run_cmd(cmd, root, logs_dir / f"{case_name}_qmap.log", env=env)

    cmd = [
        executable(args.compressor),
        str(crop.path),
        f"{args.lambda_value:.8g}",
        str(latent),
        str(args.encoder_weights),
        f"{args.max_lambda:.8g}",
        str(qmap),
    ]
    compress_s = run_cmd(cmd, root, logs_dir / f"{case_name}_compress.log", env=env)

    cmd = [
        executable(args.decompressor),
        str(latent),
        str(recon),
        str(args.decoder_weights),
        f"{args.max_lambda:.8g}",
    ]
    decompress_s = run_cmd(cmd, root, logs_dir / f"{case_name}_decompress.log", env=env)

    analysis_t0 = time.perf_counter()
    original = load_raw_u16(crop.path, args.bands, args.height, args.width)
    reconstructed = load_raw_u16(recon, args.bands, args.height, args.width)
    metrics = metrics_for_arrays(original, reconstructed)
    _ = block_mse_array(original, reconstructed, args.block_size)
    analysis_s = time.perf_counter() - analysis_t0

    q_stats = qmap_stats(load_qmap(qmap, args.height // args.block_size, args.width // args.block_size))
    fq_stats = parse_fq_summary(q_summary)
    target_mse = psnr_to_mse(target_psnr)

    row = {
        "mode": mode,
        "crop": crop.name,
        "file": str(crop.path),
        "target_psnr": f"{target_psnr:.1f}",
        "target_mse": target_mse,
        "achieved_psnr": metrics["psnr_db"],
        "achieved_mse": metrics["mse"],
        "psnr_error_db": metrics["psnr_db"] - target_psnr,
        "mse_error": metrics["mse"] - target_mse,
        "abs_psnr_error_db": abs(metrics["psnr_db"] - target_psnr),
        **q_stats,
        **fq_stats,
        "compress_s": compress_s,
        "decompress_s": decompress_s,
        "qmap_s": qmap_s,
        "analysis_s": analysis_s,
        "monotonic_rank": math.nan,
        "elapsed_total_s": time.perf_counter() - t0,
    }

    if keep_artifacts == "none":
        for p in [latent, recon]:
            if p.exists():
                p.unlink()
    elif keep_artifacts == "sample" and mode == "fixed":
        # Keep all qmaps/summaries, but only keep heavy artifacts for a few representative cases.
        keep = crop.semantic_matches in (None, 0) or target_psnr in (min(args.targets), max(args.targets), 76.8)
        if not keep:
            for p in [latent, recon]:
                if p.exists():
                    p.unlink()
    return row


def summarize_group(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        base = {k: v for k, v in zip(keys, key)}
        psnr_errors = np.array([float(r["psnr_error_db"]) for r in items], dtype=np.float64)
        abs_errors = np.abs(psnr_errors)
        achieved = np.array([float(r["achieved_psnr"]) for r in items], dtype=np.float64)
        reachable = np.array([float(r["reachable_pct"]) for r in items], dtype=np.float64)
        row = {
            **base,
            "cases": len(items),
            "achieved_psnr_mean": float(np.mean(achieved)),
            "achieved_psnr_min": float(np.min(achieved)),
            "achieved_psnr_max": float(np.max(achieved)),
            "psnr_error_mean_db": float(np.mean(psnr_errors)),
            "abs_psnr_error_mean_db": float(np.mean(abs_errors)),
            "abs_psnr_error_max_db": float(np.max(abs_errors)),
            "reachable_pct_mean": float(np.mean(reachable)),
            "too_relaxed_blocks_mean": float(np.mean([float(r["too_relaxed_blocks"]) for r in items])),
            "too_strict_blocks_mean": float(np.mean([float(r["too_strict_blocks"]) for r in items])),
            "q_mean_mean": float(np.mean([float(r["q_mean"]) for r in items])),
            "q_128_blocks_mean": float(np.mean([float(r["q_128_blocks"]) for r in items])),
            "q_255_blocks_mean": float(np.mean([float(r["q_255_blocks"]) for r in items])),
        }
        out.append(row)
    return out


def monotonicity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["mode"]), str(row["crop"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (mode, crop), items in sorted(groups.items()):
        ordered = sorted(items, key=lambda r: float(r["target_psnr"]))
        psnrs = [float(r["achieved_psnr"]) for r in ordered]
        targets = [float(r["target_psnr"]) for r in ordered]
        violations = 0
        for a, b in zip(psnrs, psnrs[1:]):
            if b + 1e-9 < a:
                violations += 1
        out.append(
            {
                "mode": mode,
                "crop": crop,
                "targets": " ".join(f"{t:.1f}" for t in targets),
                "achieved_psnrs": " ".join(f"{p:.6f}" for p in psnrs),
                "monotonic": int(violations == 0),
                "violations": violations,
            }
        )
    return out


def compare_fixed_recalibrated(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fixed: dict[tuple[str, str], dict[str, Any]] = {}
    recal: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["crop"]), str(row["target_psnr"]))
        if row["mode"] == "fixed":
            fixed[key] = row
        elif row["mode"] == "recalibrated":
            recal[key] = row
    out: list[dict[str, Any]] = []
    for key in sorted(set(fixed) & set(recal)):
        f = fixed[key]
        r = recal[key]
        out.append(
            {
                "crop": key[0],
                "target_psnr": key[1],
                "fixed_achieved_psnr": f["achieved_psnr"],
                "recalibrated_achieved_psnr": r["achieved_psnr"],
                "fixed_abs_error_db": f["abs_psnr_error_db"],
                "recalibrated_abs_error_db": r["abs_psnr_error_db"],
                "abs_error_delta_recal_minus_fixed_db": float(r["abs_psnr_error_db"]) - float(f["abs_psnr_error_db"]),
                "fixed_reachable_pct": f["reachable_pct"],
                "recalibrated_reachable_pct": r["reachable_pct"],
                "fixed_q_mean": f["q_mean"],
                "recalibrated_q_mean": r["q_mean"],
            }
        )
    return out


def write_markdown(path: Path, rows: list[dict[str, Any]], per_target: list[dict[str, Any]], monotonic: list[dict[str, Any]], comparison: list[dict[str, Any]], args: argparse.Namespace) -> None:
    fixed_rows = [r for r in rows if r["mode"] == "fixed"]
    recal_rows = [r for r in rows if r["mode"] == "recalibrated"]
    fixed_abs = np.array([float(r["abs_psnr_error_db"]) for r in fixed_rows], dtype=np.float64) if fixed_rows else np.array([])
    recal_abs = np.array([float(r["abs_psnr_error_db"]) for r in recal_rows], dtype=np.float64) if recal_rows else np.array([])
    lines = [
        "# Target PSNR Dataset Validation",
        "",
        "Validacion matematica de `target PSNR -> target MSE -> Q-map -> reconstruccion real`.",
        "",
        "## Resumen",
        "",
        f"- Crops fijos evaluados: {len(set(r['crop'] for r in fixed_rows))}.",
        f"- Targets: {', '.join(f'{t:.1f}' for t in args.targets)} dB.",
        f"- Casos fijos: {len(fixed_rows)}.",
        f"- Casos recalibrados: {len(recal_rows)}.",
    ]
    if fixed_abs.size:
        lines.append(f"- Error absoluto PSNR medio con calibracion fija: {float(np.mean(fixed_abs)):.4f} dB.")
        lines.append(f"- Error absoluto PSNR maximo con calibracion fija: {float(np.max(fixed_abs)):.4f} dB.")
    if recal_abs.size:
        lines.append(f"- Error absoluto PSNR medio con recalibracion por crop: {float(np.mean(recal_abs)):.4f} dB.")
    mono_viol = sum(int(r["violations"]) for r in monotonic)
    lines.append(f"- Violaciones de monotonia por crop/modo: {mono_viol}.")
    lines.extend(
        [
            "",
            "## Por target, calibracion fija",
            "",
            "| Target | Cases | Achieved mean | Error mean | Abs error mean | Reachable | Q mean | Q128 | Q255 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_target:
        if row.get("mode") != "fixed":
            continue
        lines.append(
            f"| {float(row['target_psnr']):.1f} | {row['cases']} | {row['achieved_psnr_mean']:.4f} | "
            f"{row['psnr_error_mean_db']:.4f} | {row['abs_psnr_error_mean_db']:.4f} | "
            f"{row['reachable_pct_mean']:.2f}% | {row['q_mean_mean']:.2f} | "
            f"{row['q_128_blocks_mean']:.1f} | {row['q_255_blocks_mean']:.1f} |"
        )
    if comparison:
        deltas = np.array([float(r["abs_error_delta_recal_minus_fixed_db"]) for r in comparison], dtype=np.float64)
        lines.extend(
            [
                "",
                "## Calibracion fija vs recalibracion por crop",
                "",
                f"- Delta medio de error absoluto recalibrado-fijo: {float(np.mean(deltas)):.4f} dB.",
                f"- Casos donde recalibrar mejora: {int(np.sum(deltas < 0.0))}/{len(deltas)}.",
            ]
        )
    lines.extend(
        [
            "",
            "Los objetivos extremos deben interpretarse junto con la saturacion a Q=128/Q=255. "
            "Cuando muchos bloques aparecen como `too_relaxed` o `too_strict`, el resultado no invalida "
            "la formula: indica que el objetivo esta fuera del rango Q calibrado para esos bloques.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida target PSNR/MSE sobre el dataset Sentinel-2A.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path, default=Path("data/Sentinel2A_crop_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/checkpoints/20260518_target_psnr_dataset_validation"))
    parser.add_argument("--calibration", type=Path, default=Path("output/checkpoints/20260507_c_fixed_quality_qmap_wide/fq_calibration.tsv"))
    parser.add_argument("--semantic-summary", type=Path, default=Path("output/checkpoints/20260513_sentinel2a_8band_dataset_validation/semantic_dataset_summary.csv"))
    parser.add_argument("--targets", type=float, nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--max-crops", type=int, default=None, help="Limita crops para smoke tests.")
    parser.add_argument("--recalibrate-count", type=int, default=5)
    parser.add_argument("--q-values", type=int, nargs="+", default=WIDE_Q_VALUES)
    parser.add_argument("--keep-artifacts", choices=["all", "sample", "none"], default="all")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--lambda-value", type=float, default=0.1)
    parser.add_argument("--max-lambda", type=float, default=0.125)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--encoder-weights", type=Path, default=Path("weights/encoder"))
    parser.add_argument("--decoder-weights", type=Path, default=Path("weights/decoder"))
    parser.add_argument("--fq-bin", type=Path, default=Path("./sorteny_fq_qmap"))
    parser.add_argument("--compressor", type=Path, default=Path("./sorteny_compressor"))
    parser.add_argument("--decompressor", type=Path, default=Path("./sorteny_decompressor"))
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    args.data_dir = resolve(root, args.data_dir)
    args.output_dir = resolve(root, args.output_dir)
    args.calibration = resolve(root, args.calibration)
    args.semantic_summary = resolve(root, args.semantic_summary)
    args.encoder_weights = resolve(root, args.encoder_weights)
    args.decoder_weights = resolve(root, args.decoder_weights)

    for path in [args.data_dir, args.calibration, args.encoder_weights, args.decoder_weights, resolve(root, args.fq_bin), resolve(root, args.compressor), resolve(root, args.decompressor)]:
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = args.output_dir / "target_psnr_dataset_results.csv"
    crops = validate_raw_dataset(args.data_dir, args.bands, args.height, args.width, args.max_crops)
    crops = attach_matches(crops, load_semantic_matches(args.semantic_summary))
    recal_crops = select_recalibration_crops(crops, args.recalibrate_count)
    recal_names = {c.name for c in recal_crops}

    manifest_rows = [
        {
            "crop": c.name,
            "file": str(c.path),
            "bytes": c.path.stat().st_size,
            "semantic_matches": c.semantic_matches if c.semantic_matches is not None else "",
            "selected_for_recalibration": int(c.name in recal_names),
        }
        for c in crops
    ]
    write_csv(args.output_dir / "dataset_manifest.csv", manifest_rows, ["crop", "file", "bytes", "semantic_matches", "selected_for_recalibration"])

    existing = existing_keys(results_csv) if args.resume else set()
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)

    all_rows: list[dict[str, Any]] = []
    if results_csv.exists() and args.resume:
        for row in read_csv(results_csv):
            converted: dict[str, Any] = dict(row)
            for key in RESULT_FIELDS:
                if key in converted and key not in {"mode", "crop", "file", "target_psnr"}:
                    try:
                        converted[key] = float(converted[key])
                    except ValueError:
                        pass
            all_rows.append(converted)

    print(f"Crops: {len(crops)}; targets: {args.targets}; recalibration crops: {len(recal_crops)}")
    for crop in crops:
        for target in args.targets:
            key = ("fixed", crop.name, f"{target:.1f}")
            if key in existing:
                continue
            row = run_target_case(
                root=root,
                mode="fixed",
                crop=crop,
                target_psnr=target,
                calibration=args.calibration,
                outdir=args.output_dir,
                args=args,
                env=env,
                keep_artifacts=args.keep_artifacts,
            )
            all_rows.append(row)
            append_csv(results_csv, [row], RESULT_FIELDS)
            existing.add(key)
            print(f"[fixed] {crop.name} target={target:.1f} achieved={row['achieved_psnr']:.4f} error={row['psnr_error_db']:.4f}")

    recal_root = args.output_dir / "recalibration"
    sweep_script = root / "src/python/analysis/sweep_q_quality.py"
    calib_script = root / "src/python/analysis/build_fq_calibration.py"
    for crop in recal_crops:
        crop_recal_dir = recal_root / crop.name
        sweep_dir = crop_recal_dir / "q_sweep"
        calibration_tsv = crop_recal_dir / "fq_calibration.tsv"
        if not calibration_tsv.exists():
            q_args = [str(q) for q in args.q_values]
            cmd = [
                sys.executable,
                str(sweep_script),
                "--input",
                str(crop.path),
                "--output-dir",
                str(sweep_dir),
                "--q-values",
                *q_args,
                "--threads",
                str(args.threads),
                "--encoder",
                str(resolve(root, args.compressor)),
                "--decoder",
                str(resolve(root, args.decompressor)),
                "--encoder-weights",
                str(args.encoder_weights),
                "--decoder-weights",
                str(args.decoder_weights),
                "--max-lambda",
                f"{args.max_lambda:.8g}",
            ]
            run_cmd(cmd, root, args.output_dir / "logs" / "recalibration" / crop.name / "sweep.log", env=env)
            cmd = [
                sys.executable,
                str(calib_script),
                "--sweep",
                str(sweep_dir / "sweep_results.json"),
                "--weights",
                str(args.encoder_weights),
                "--output-tsv",
                str(calibration_tsv),
                "--summary-json",
                str(crop_recal_dir / "fq_calibration_summary.json"),
            ]
            run_cmd(cmd, root, args.output_dir / "logs" / "recalibration" / crop.name / "build_calibration.log", env=env)
        for target in args.targets:
            key = ("recalibrated", crop.name, f"{target:.1f}")
            if key in existing:
                continue
            row = run_target_case(
                root=root,
                mode="recalibrated",
                crop=crop,
                target_psnr=target,
                calibration=calibration_tsv,
                outdir=args.output_dir,
                args=args,
                env=env,
                keep_artifacts=args.keep_artifacts,
            )
            all_rows.append(row)
            append_csv(results_csv, [row], RESULT_FIELDS)
            existing.add(key)
            print(f"[recal] {crop.name} target={target:.1f} achieved={row['achieved_psnr']:.4f} error={row['psnr_error_db']:.4f}")

    per_crop = summarize_group(all_rows, ["mode", "crop"])
    per_target = summarize_group(all_rows, ["mode", "target_psnr"])
    block_feas = summarize_group(all_rows, ["mode", "target_psnr"])
    monotonic = monotonicity_rows(all_rows)
    comparison = compare_fixed_recalibrated(all_rows)

    write_csv(args.output_dir / "per_crop_summary.csv", per_crop, list(per_crop[0].keys()) if per_crop else [])
    write_csv(args.output_dir / "per_target_summary.csv", per_target, list(per_target[0].keys()) if per_target else [])
    write_csv(args.output_dir / "block_feasibility_summary.csv", block_feas, list(block_feas[0].keys()) if block_feas else [])
    write_csv(args.output_dir / "monotonicity_summary.csv", monotonic, ["mode", "crop", "targets", "achieved_psnrs", "monotonic", "violations"])
    write_csv(args.output_dir / "fixed_vs_recalibrated_summary.csv", comparison, list(comparison[0].keys()) if comparison else [])

    summary = {
        "checkpoint": str(args.output_dir),
        "config": {
            "targets": args.targets,
            "crops": len(crops),
            "recalibration_crops": [c.name for c in recal_crops],
            "q_values_recalibration": args.q_values,
            "keep_artifacts": args.keep_artifacts,
            "threads": args.threads,
        },
        "results": all_rows,
        "per_crop_summary": per_crop,
        "per_target_summary": per_target,
        "monotonicity": monotonic,
        "fixed_vs_recalibrated": comparison,
    }
    write_json(args.output_dir / "target_psnr_dataset_results.json", summary)
    write_markdown(args.output_dir / "target_psnr_dataset_summary.md", all_rows, per_target, monotonic, comparison, args)

    fixed_rows = [r for r in all_rows if r["mode"] == "fixed"]
    fixed_abs = np.array([float(r["abs_psnr_error_db"]) for r in fixed_rows], dtype=np.float64) if fixed_rows else np.array([])
    print(f"[OK] Target PSNR dataset validation: {args.output_dir}")
    if fixed_abs.size:
        print(f"fixed cases={len(fixed_rows)} mean_abs_error={float(np.mean(fixed_abs)):.4f} dB max_abs_error={float(np.max(fixed_abs)):.4f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
