#!/usr/bin/env python3
"""
Demuestra tres interpretaciones distintas de un objetivo PSNR/MSE.

El script es auxiliar: orquesta binarios C y calcula metricas. La generacion
de Q-map local y semantico sigue ocurriendo en `sorteny_fq_qmap` y
`sorteny_semantic_qmap`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import subprocess
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MAX_U16 = 65535.0
DEFAULT_TARGETS = [74.5, 76.8, 77.5]


@dataclass(frozen=True)
class CropInfo:
    path: Path
    name: str
    semantic_matches: int | None = None


@dataclass
class PipelineResult:
    qmap: Path
    latent: Path
    recon: Path
    q_summary: Path
    qmap_s: float
    compress_s: float
    decompress_s: float
    analysis_s: float
    qmap_arr: np.ndarray
    original: np.ndarray
    reconstructed: np.ndarray
    block: dict[str, np.ndarray]
    global_metrics: dict[str, float]
    latent_stats: dict[str, Any]
    latent_zero_map: np.ndarray
    latent_entropy_map: np.ndarray


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


def executable(path: Path) -> str:
    if path.is_absolute():
        return str(path)
    if path.parent == Path(".") or str(path.parent) == ".":
        return f"./{path.name}"
    return str(path)


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


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


def block_metrics(original: np.ndarray, reconstructed: np.ndarray, block_size: int) -> dict[str, np.ndarray]:
    _, height, width = original.shape
    bh = height // block_size
    bw = width // block_size
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    abs_diff = np.abs(diff)
    mse = np.zeros((bh, bw), dtype=np.float64)
    mae = np.zeros((bh, bw), dtype=np.float64)
    max_abs = np.zeros((bh, bw), dtype=np.float64)
    exact_pct = np.zeros((bh, bw), dtype=np.float64)
    for by in range(bh):
        y0 = by * block_size
        y1 = y0 + block_size
        for bx in range(bw):
            x0 = bx * block_size
            x1 = x0 + block_size
            d = diff[:, y0:y1, x0:x1]
            ad = abs_diff[:, y0:y1, x0:x1]
            mse[by, bx] = float(np.mean(d * d))
            mae[by, bx] = float(np.mean(ad))
            max_abs[by, bx] = float(np.max(ad))
            exact_pct[by, bx] = float(np.mean(ad == 0.0) * 100.0)
    psnr = np.vectorize(psnr_from_mse, otypes=[np.float64])(mse)
    return {"mse": mse, "psnr_db": psnr, "mae": mae, "max_abs": max_abs, "exact_pct": exact_pct}


def q_stats(qmap: np.ndarray, roi: np.ndarray | None = None) -> dict[str, Any]:
    stats = {
        "q_min": int(np.min(qmap)),
        "q_max": int(np.max(qmap)),
        "q_mean": float(np.mean(qmap)),
        "q_unique": int(np.unique(qmap).size),
        "q_128_blocks": int(np.sum(qmap == 128)),
        "q_255_blocks": int(np.sum(qmap == 255)),
        "qmap_bytes": int(qmap.size),
    }
    if roi is not None:
        bg = ~roi
        stats["q_roi_mean"] = float(np.mean(qmap[roi])) if np.any(roi) else math.nan
        stats["q_background_mean"] = float(np.mean(qmap[bg])) if np.any(bg) else math.nan
    return stats


def parse_fq_summary(path: Path) -> dict[str, Any]:
    counts = {
        "reachable_blocks": 0,
        "too_relaxed_blocks": 0,
        "too_strict_blocks": 0,
        "invalid_blocks": 0,
        "adaptive_budget_blocks": 0,
    }
    predicted: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t")
        for row in reader:
            viability = row.get("viability", "")
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
            value = row.get("predicted_mse")
            if value not in (None, ""):
                predicted.append(float(value))
    total = sum(counts.values())
    return {
        **counts,
        "blocks": total,
        "reachable_pct": (counts["reachable_blocks"] / total * 100.0) if total else math.nan,
        "too_relaxed_pct": (counts["too_relaxed_blocks"] / total * 100.0) if total else math.nan,
        "too_strict_pct": (counts["too_strict_blocks"] / total * 100.0) if total else math.nan,
        "predicted_mse_mean": float(np.mean(predicted)) if predicted else math.nan,
    }


def read_semantic_tsv(path: Path, q_height: int, q_width: int) -> tuple[np.ndarray, np.ndarray]:
    ndvi = np.full((q_height, q_width), np.nan, dtype=np.float64)
    roi = np.zeros((q_height, q_width), dtype=bool)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            by = int(row["block_y"])
            bx = int(row["block_x"])
            ndvi[by, bx] = float(row["index_mean"])
            roi[by, bx] = int(row["semantic_match"]) != 0
    if np.isnan(ndvi).any():
        raise ValueError(f"{path}: missing semantic blocks")
    return ndvi, roi


def group_summary(block: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    if not np.any(mask):
        return {"blocks": 0, "mse": math.nan, "psnr_db": math.nan, "mae": math.nan, "max_abs": math.nan, "exact_pct": math.nan}
    mse = float(np.mean(block["mse"][mask]))
    return {
        "blocks": int(np.sum(mask)),
        "mse": mse,
        "psnr_db": psnr_from_mse(mse),
        "mae": float(np.mean(block["mae"][mask])),
        "max_abs": float(np.max(block["max_abs"][mask])),
        "exact_pct": float(np.mean(block["exact_pct"][mask])),
    }


def entropy_of_values(values: np.ndarray) -> float:
    flat = values.reshape(-1)
    _, counts = np.unique(flat, return_counts=True)
    probs = counts.astype(np.float64) / float(flat.size)
    return float(-np.sum(probs * np.log2(probs)))


def read_bitstream(path: Path) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        header_bytes = f.read(10)
        if len(header_bytes) != 10:
            raise ValueError(f"{path}: incomplete header")
        bands, height, width, datatype, num_filters = struct.unpack("<5H", header_bytes)
        q_height = height // 16
        q_width = width // 16
        q_size = q_height * q_width
        q_raw = f.read(q_size)
        if len(q_raw) != q_size:
            raise ValueError(f"{path}: incomplete qmap")
        latents = np.fromfile(f, dtype=np.int32)
    expected_latents = bands * num_filters * q_height * q_width
    if latents.size != expected_latents:
        raise ValueError(f"{path}: latents {latents.size}, expected {expected_latents}")
    header = {
        "bands": int(bands),
        "height": int(height),
        "width": int(width),
        "datatype": int(datatype),
        "num_filters": int(num_filters),
        "q_height": int(q_height),
        "q_width": int(q_width),
    }
    qmap = np.frombuffer(q_raw, dtype=np.uint8).copy().reshape(q_height, q_width)
    latents = latents.reshape(bands, num_filters, q_height, q_width)
    return header, qmap, latents


def latent_stats(latents: np.ndarray, input_samples: int) -> dict[str, Any]:
    flat = latents.reshape(-1)
    abs_flat = np.abs(flat.astype(np.int64))
    entropy = entropy_of_values(flat)
    ideal_bits = entropy * float(flat.size)
    zbytes = len(zlib.compress(flat.astype(np.int32, copy=False).tobytes(), level=9))
    raw_bytes = int(flat.size * 4)
    return {
        "latent_samples": int(flat.size),
        "zero_pct": float(np.mean(flat == 0) * 100.0),
        "entropy_bits_per_symbol": entropy,
        "ideal_bits": ideal_bits,
        "ideal_bps_per_input_sample": ideal_bits / float(input_samples),
        "zlib_bytes_level9": int(zbytes),
        "zlib_bps_per_input_sample": float((zbytes * 8) / input_samples),
        "zlib_ratio_vs_int32": float(zbytes / raw_bytes),
        "mean_abs": float(np.mean(abs_flat)),
        "max_abs": int(np.max(abs_flat)),
        "unique_values": int(np.unique(flat).size),
    }


def latent_zero_map(latents: np.ndarray) -> np.ndarray:
    return np.mean(latents == 0, axis=(0, 1)) * 100.0


def latent_entropy_map(latents: np.ndarray) -> np.ndarray:
    out = np.zeros(latents.shape[2:], dtype=np.float64)
    for by in range(latents.shape[2]):
        for bx in range(latents.shape[3]):
            out[by, bx] = entropy_of_values(latents[:, :, by, bx])
    return out


def normalize_to_u8(data: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    arr = data.astype(np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.min(finite)) if vmin is None else float(vmin)
    hi = float(np.max(finite)) if vmax is None else float(vmax)
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.rint(np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


def upsample(image: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def diverging_rgb(data: np.ndarray, max_abs: float | None = None) -> np.ndarray:
    arr = data.astype(np.float64)
    if max_abs is None:
        finite = arr[np.isfinite(arr)]
        max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
    if max_abs <= 0.0:
        max_abs = 1.0
    n = np.clip(arr / max_abs, -1.0, 1.0)
    rgb = np.full(arr.shape + (3,), 255.0, dtype=np.float64)
    pos = n > 0.0
    neg = n < 0.0
    rgb[pos, 1] = 255.0 * (1.0 - n[pos])
    rgb[pos, 2] = 255.0 * (1.0 - n[pos])
    rgb[neg, 0] = 255.0 * (1.0 + n[neg])
    rgb[neg, 1] = 255.0 * (1.0 + n[neg])
    return np.rint(np.clip(rgb, 0.0, 255.0)).astype(np.uint8)


def write_pgm(path: Path, image: np.ndarray) -> None:
    img = image.astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode("ascii"))
        f.write(img.tobytes())


def write_ppm(path: Path, image: np.ndarray) -> None:
    img = image.astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P6\n{img.shape[1]} {img.shape[0]}\n255\n".encode("ascii"))
        f.write(img.tobytes())


def write_map_pair(base_dir: Path, name: str, image: np.ndarray, rgb: bool = False, factor: int = 16) -> None:
    if rgb:
        write_ppm(base_dir / "32x32" / name, image)
        write_ppm(base_dir / "512x512" / name, upsample(image, factor))
    else:
        write_pgm(base_dir / "32x32" / name, image)
        write_pgm(base_dir / "512x512" / name, upsample(image, factor))


def load_semantic_matches(summary_csv: Path) -> dict[str, int]:
    if not summary_csv.exists():
        return {}
    out: dict[str, int] = {}
    for row in read_csv(summary_csv):
        out[Path(row["file"]).stem] = int(float(row["semantic_matches"]))
    return out


def validate_raw_dataset(data_dir: Path, bands: int, height: int, width: int) -> list[CropInfo]:
    raw_files = sorted(p for p in data_dir.glob("*.raw") if not p.name.endswith(":Zone.Identifier"))
    expected_bytes = bands * height * width * 2
    crops: list[CropInfo] = []
    for p in raw_files:
        if p.stat().st_size != expected_bytes:
            raise ValueError(f"{p}: size {p.stat().st_size}, expected {expected_bytes}")
        crops.append(CropInfo(path=p, name=p.stem))
    return crops


def select_representative_crops(crops: list[CropInfo], count: int) -> list[CropInfo]:
    if count <= 0 or count >= len(crops):
        return crops
    ranked = sorted(crops, key=lambda c: ((c.semantic_matches if c.semantic_matches is not None else -1), c.name))
    positions = np.linspace(0, len(ranked) - 1, count)
    selected: list[CropInfo] = []
    seen: set[str] = set()
    for pos in positions:
        crop = ranked[int(round(float(pos)))]
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


def analyze_pipeline(
    *,
    root: Path,
    crop: CropInfo,
    qmap: Path,
    q_summary: Path,
    latent: Path,
    recon: Path,
    logs_dir: Path,
    args: argparse.Namespace,
    env: dict[str, str],
    command_prefix: str,
    qmap_s: float,
) -> PipelineResult:
    t_compress = run_cmd(
        [
            executable(args.compressor),
            str(crop.path),
            f"{args.lambda_value:.8g}",
            str(latent),
            str(args.encoder_weights),
            f"{args.max_lambda:.8g}",
            str(qmap),
        ],
        root,
        logs_dir / f"{command_prefix}_compress.log",
        env=env,
    )
    t_decompress = run_cmd(
        [
            executable(args.decompressor),
            str(latent),
            str(recon),
            str(args.decoder_weights),
            f"{args.max_lambda:.8g}",
        ],
        root,
        logs_dir / f"{command_prefix}_decompress.log",
        env=env,
    )
    t0 = time.perf_counter()
    original = load_raw_u16(crop.path, args.bands, args.height, args.width)
    reconstructed = load_raw_u16(recon, args.bands, args.height, args.width)
    block = block_metrics(original, reconstructed, args.block_size)
    global_metrics = metrics_for_arrays(original, reconstructed)
    _, _, latents = read_bitstream(latent)
    lstats = latent_stats(latents, args.bands * args.height * args.width)
    zmap = latent_zero_map(latents)
    emap = latent_entropy_map(latents)
    qmap_arr = load_qmap(qmap, args.height // args.block_size, args.width // args.block_size)
    analysis_s = time.perf_counter() - t0
    return PipelineResult(
        qmap=qmap,
        latent=latent,
        recon=recon,
        q_summary=q_summary,
        qmap_s=qmap_s,
        compress_s=t_compress,
        decompress_s=t_decompress,
        analysis_s=analysis_s,
        qmap_arr=qmap_arr,
        original=original,
        reconstructed=reconstructed,
        block=block,
        global_metrics=global_metrics,
        latent_stats=lstats,
        latent_zero_map=zmap,
        latent_entropy_map=emap,
    )


def run_fq_case(
    *,
    root: Path,
    crop: CropInfo,
    case_dir: Path,
    logs_dir: Path,
    args: argparse.Namespace,
    env: dict[str, str],
    name: str,
    fq_args: list[str],
) -> PipelineResult:
    qmap = case_dir / "qmap.bin"
    q_summary = case_dir / "qmap_summary.tsv"
    latent = case_dir / "latent.bin"
    recon = case_dir / "reconstructed.raw"
    case_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        executable(args.fq_bin),
        "--calibration",
        str(args.calibration),
        *fq_args,
        "--output-qmap",
        str(qmap),
        "--summary-tsv",
        str(q_summary),
    ]
    t_qmap = run_cmd(cmd, root, logs_dir / f"{name}_qmap.log", env=env)
    return analyze_pipeline(
        root=root,
        crop=crop,
        qmap=qmap,
        q_summary=q_summary,
        latent=latent,
        recon=recon,
        logs_dir=logs_dir,
        args=args,
        env=env,
        command_prefix=name,
        qmap_s=t_qmap,
    )


def run_semantic_case(
    *,
    root: Path,
    crop: CropInfo,
    case_dir: Path,
    logs_dir: Path,
    args: argparse.Namespace,
    env: dict[str, str],
    name: str,
    policy_args: list[str],
) -> PipelineResult:
    qmap = case_dir / "qmap.bin"
    q_summary = case_dir / "semantic_summary.tsv"
    latent = case_dir / "latent.bin"
    recon = case_dir / "reconstructed.raw"
    case_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        executable(args.semantic_bin),
        "--input",
        str(crop.path),
        "--calibration",
        str(args.calibration),
        "--preset",
        "vegetation",
        "--bands",
        str(args.bands),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--band-layout",
        "sentinel2-8",
        "--threshold",
        f"{args.threshold:.8g}",
        "--q-mean",
        "204",
        "--adaptive-strength",
        "8",
        *policy_args,
        "--output-qmap",
        str(qmap),
        "--summary-tsv",
        str(q_summary),
    ]
    t_qmap = run_cmd(cmd, root, logs_dir / f"{name}_qmap.log", env=env)
    return analyze_pipeline(
        root=root,
        crop=crop,
        qmap=qmap,
        q_summary=q_summary,
        latent=latent,
        recon=recon,
        logs_dir=logs_dir,
        args=args,
        env=env,
        command_prefix=name,
        qmap_s=t_qmap,
    )


def summarize_local(result: PipelineResult, crop: CropInfo, target: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_mse = psnr_to_mse(target)
    fq_stats = parse_fq_summary(result.q_summary)
    q = q_stats(result.qmap_arr)
    mse = result.block["mse"]
    psnr = result.block["psnr_db"]
    abs_mse_error = np.abs(mse - target_mse)
    finite_psnr = psnr[np.isfinite(psnr)]
    psnr_errors = finite_psnr - target if finite_psnr.size else np.array([])
    row = {
        "objective_mode": "local_block",
        "crop": crop.name,
        "target_psnr": target,
        "target_mse": target_mse,
        "global_psnr": result.global_metrics["psnr_db"],
        "global_mse": result.global_metrics["mse"],
        "block_mse_error_mean": float(np.mean(mse - target_mse)),
        "block_mse_abs_error_mean": float(np.mean(abs_mse_error)),
        "block_mse_abs_error_median": float(np.median(abs_mse_error)),
        "block_psnr_error_mean": float(np.mean(psnr_errors)) if psnr_errors.size else math.nan,
        "block_psnr_abs_error_mean": float(np.mean(np.abs(psnr_errors))) if psnr_errors.size else math.nan,
        **q,
        **fq_stats,
        "qmap_s": result.qmap_s,
        "compress_s": result.compress_s,
        "decompress_s": result.decompress_s,
        "analysis_s": result.analysis_s,
    }
    block_rows: list[dict[str, Any]] = []
    for by in range(mse.shape[0]):
        for bx in range(mse.shape[1]):
            block_rows.append(
                {
                    "objective_mode": "local_block",
                    "crop": crop.name,
                    "target_psnr": target,
                    "policy": "target_psnr",
                    "block_y": by,
                    "block_x": bx,
                    "q": int(result.qmap_arr[by, bx]),
                    "mse": float(mse[by, bx]),
                    "psnr_db": float(psnr[by, bx]),
                    "target_mse": target_mse,
                    "mse_error": float(mse[by, bx] - target_mse),
                    "psnr_error_db": float(psnr[by, bx] - target) if math.isfinite(float(psnr[by, bx])) else math.inf,
                    "roi": "",
                }
            )
    return row, block_rows


def summarize_global(result: PipelineResult, crop: CropInfo, target: float, q: int, status: str, iterations: int) -> dict[str, Any]:
    return {
        "objective_mode": "global_image",
        "crop": crop.name,
        "target_psnr": target,
        "target_mse": psnr_to_mse(target),
        "achieved_psnr": result.global_metrics["psnr_db"],
        "achieved_mse": result.global_metrics["mse"],
        "psnr_error_db": result.global_metrics["psnr_db"] - target,
        "abs_psnr_error_db": abs(result.global_metrics["psnr_db"] - target),
        "selected_q": q,
        "status": status,
        "iterations": iterations,
        **q_stats(result.qmap_arr),
        "qmap_s": result.qmap_s,
        "compress_s": result.compress_s,
        "decompress_s": result.decompress_s,
        "analysis_s": result.analysis_s,
    }


def summarize_semantic(result: PipelineResult, crop: CropInfo, policy: str, roi: np.ndarray) -> dict[str, Any]:
    bg = ~roi
    roi_metrics = group_summary(result.block, roi)
    bg_metrics = group_summary(result.block, bg)
    q = q_stats(result.qmap_arr, roi)
    return {
        "objective_mode": "semantic_roi_background",
        "crop": crop.name,
        "policy": policy,
        "roi_blocks": roi_metrics["blocks"],
        "background_blocks": bg_metrics["blocks"],
        "global_psnr": result.global_metrics["psnr_db"],
        "global_mse": result.global_metrics["mse"],
        "roi_psnr": roi_metrics["psnr_db"],
        "roi_mse": roi_metrics["mse"],
        "background_psnr": bg_metrics["psnr_db"],
        "background_mse": bg_metrics["mse"],
        **q,
        "latent_zero_pct": result.latent_stats["zero_pct"],
        "latent_entropy_bits_per_symbol": result.latent_stats["entropy_bits_per_symbol"],
        "latent_ideal_bps": result.latent_stats["ideal_bps_per_input_sample"],
        "latent_zlib_bps": result.latent_stats["zlib_bps_per_input_sample"],
        "latent_mean_abs": result.latent_stats["mean_abs"],
        "qmap_s": result.qmap_s,
        "compress_s": result.compress_s,
        "decompress_s": result.decompress_s,
        "analysis_s": result.analysis_s,
    }


def find_global_q(
    *,
    root: Path,
    crop: CropInfo,
    target: float,
    base_dir: Path,
    logs_dir: Path,
    args: argparse.Namespace,
    env: dict[str, str],
) -> tuple[PipelineResult, int, str, int, dict[int, PipelineResult]]:
    evaluated: dict[int, PipelineResult] = {}

    def eval_q(q: int) -> PipelineResult:
        if q not in evaluated:
            evaluated[q] = run_fq_case(
                root=root,
                crop=crop,
                case_dir=base_dir / f"q_{q:03d}",
                logs_dir=logs_dir,
                args=args,
                env=env,
                name=f"global_target_{target:.1f}_q_{q:03d}".replace(".", "p"),
                fq_args=["--target-from-q", str(q)],
            )
        return evaluated[q]

    low_q = args.global_q_min
    high_q = args.global_q_max
    low = eval_q(low_q)
    high = eval_q(high_q)
    low_psnr = low.global_metrics["psnr_db"]
    high_psnr = high.global_metrics["psnr_db"]
    if target <= low_psnr:
        return low, low_q, "too_relaxed_at_q_min", 2, evaluated
    if target >= high_psnr:
        return high, high_q, "too_strict_at_q_max", 2, evaluated

    best_q = low_q
    best = low
    best_err = abs(low_psnr - target)
    lo = low_q
    hi = high_q
    iterations = 2
    while lo <= hi and iterations < args.global_max_iterations + 2:
        mid = (lo + hi) // 2
        result = eval_q(mid)
        iterations += 1
        err = abs(result.global_metrics["psnr_db"] - target)
        if err < best_err:
            best_err = err
            best_q = mid
            best = result
        if err <= args.global_tolerance_db:
            return result, mid, "reached_tolerance", iterations, evaluated
        if result.global_metrics["psnr_db"] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return best, best_q, "best_effort", iterations, evaluated


def write_maps(outdir: Path, crop: CropInfo, local_result: PipelineResult | None, semantic_results: dict[str, PipelineResult], roi: np.ndarray | None) -> None:
    maps_dir = outdir / "maps" / crop.name
    if local_result is not None:
        target_mse = psnr_to_mse(76.8)
        write_map_pair(maps_dir, "local_target_76p8_qmap.pgm", normalize_to_u8(local_result.qmap_arr, 128.0, 255.0))
        write_map_pair(maps_dir, "local_target_76p8_abs_mse_error.pgm", normalize_to_u8(np.abs(local_result.block["mse"] - target_mse)))
        write_map_pair(maps_dir, "local_target_76p8_psnr.pgm", normalize_to_u8(local_result.block["psnr_db"]))
    if semantic_results and roi is not None:
        write_map_pair(maps_dir, "roi_vegetation_ndvi_ge_040.pgm", roi.astype(np.uint8) * 255)
        adaptive = semantic_results.get("adaptive_s8")
        focus = semantic_results.get("vegetation_focus_bgq128")
        for name, result in semantic_results.items():
            write_map_pair(maps_dir, f"semantic_qmap_{name}.pgm", normalize_to_u8(result.qmap_arr, 128.0, 255.0))
            write_map_pair(maps_dir, f"semantic_psnr_{name}.pgm", normalize_to_u8(result.block["psnr_db"]))
            write_map_pair(maps_dir, f"semantic_latent_zero_{name}.pgm", normalize_to_u8(result.latent_zero_map, 0.0, 100.0))
            write_map_pair(maps_dir, f"semantic_latent_entropy_{name}.pgm", normalize_to_u8(result.latent_entropy_map))
        if adaptive is not None and focus is not None:
            write_map_pair(
                maps_dir,
                "delta_psnr_focus_bgq128_minus_adaptive.ppm",
                diverging_rgb(focus.block["psnr_db"] - adaptive.block["psnr_db"]),
                rgb=True,
            )
            write_map_pair(
                maps_dir,
                "delta_q_focus_bgq128_minus_adaptive.ppm",
                diverging_rgb(focus.qmap_arr.astype(np.float64) - adaptive.qmap_arr.astype(np.float64)),
                rgb=True,
            )


def write_report(path: Path, local_rows: list[dict[str, Any]], global_rows: list[dict[str, Any]], semantic_rows: list[dict[str, Any]]) -> None:
    def mean(values: list[Any]) -> float:
        arr = np.array([float(v) for v in values], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else math.nan

    lines = [
        "# PSNR Objective Modes Demo",
        "",
        "Esta demo separa tres significados de un objetivo PSNR/MSE.",
        "",
        "## 1. Objetivo local por bloque",
        "",
        "| Target | Cases | Global PSNR mean | Block abs PSNR error mean | Reachable mean | Q128 mean | Q255 mean |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in sorted({float(r["target_psnr"]) for r in local_rows}):
        rows = [r for r in local_rows if float(r["target_psnr"]) == target]
        lines.append(
            f"| {target:.1f} | {len(rows)} | {mean([r['global_psnr'] for r in rows]):.4f} | "
            f"{mean([r['block_psnr_abs_error_mean'] for r in rows]):.4f} | "
            f"{mean([r['reachable_pct'] for r in rows]):.2f}% | "
            f"{mean([r['q_128_blocks'] for r in rows]):.1f} | "
            f"{mean([r['q_255_blocks'] for r in rows]):.1f} |"
        )
    lines.extend(
        [
            "",
            "## 2. Objetivo global de imagen",
            "",
            "| Target | Cases | Achieved mean | Abs error mean | Reached tolerance | Saturated |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for target in sorted({float(r["target_psnr"]) for r in global_rows}):
        rows = [r for r in global_rows if float(r["target_psnr"]) == target]
        reached = sum(1 for r in rows if r["status"] == "reached_tolerance")
        saturated = sum(1 for r in rows if str(r["status"]).startswith("too_"))
        lines.append(
            f"| {target:.1f} | {len(rows)} | {mean([r['achieved_psnr'] for r in rows]):.4f} | "
            f"{mean([r['abs_psnr_error_db'] for r in rows]):.4f} | {reached} | {saturated} |"
        )
    lines.extend(
        [
            "",
            "## 3. Objetivo semantico ROI/fondo",
            "",
            "| Policy | Cases | Global PSNR | ROI PSNR | Background PSNR | Q mean | Q ROI | Q background | Zeros | Entropy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for policy in sorted({str(r["policy"]) for r in semantic_rows}):
        rows = [r for r in semantic_rows if r["policy"] == policy]
        lines.append(
            f"| {policy} | {len(rows)} | {mean([r['global_psnr'] for r in rows]):.4f} | "
            f"{mean([r['roi_psnr'] for r in rows]):.4f} | {mean([r['background_psnr'] for r in rows]):.4f} | "
            f"{mean([r['q_mean'] for r in rows]):.4f} | {mean([r['q_roi_mean'] for r in rows]):.4f} | "
            f"{mean([r['q_background_mean'] for r in rows]):.4f} | "
            f"{mean([r['latent_zero_pct'] for r in rows]):.2f}% | "
            f"{mean([r['latent_entropy_bits_per_symbol'] for r in rows]):.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretacion: el caso local demuestra ajuste por bloque, el caso global busca un promedio de imagen, "
            "y el caso semantico evalua ROI/fondo. El exito semantico no requiere maximizar el PSNR global.",
            "",
            "La reduccion real de ancho de banda queda pendiente del codificador entropico; los latentes siguen escritos como int32.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demuestra tres modos de objetivo PSNR/MSE.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path, default=Path("data/Sentinel2A_crop_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/checkpoints/20260519_psnr_objective_modes_demo"))
    parser.add_argument("--calibration", type=Path, default=Path("output/checkpoints/20260507_c_fixed_quality_qmap_wide/fq_calibration.tsv"))
    parser.add_argument("--semantic-summary", type=Path, default=Path("output/checkpoints/20260513_sentinel2a_8band_dataset_validation/semantic_dataset_summary.csv"))
    parser.add_argument("--targets", type=float, nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--max-crops", type=int, default=3)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--lambda-value", type=float, default=0.1)
    parser.add_argument("--max-lambda", type=float, default=0.125)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--global-tolerance-db", type=float, default=0.05)
    parser.add_argument("--global-max-iterations", type=int, default=7)
    parser.add_argument("--global-q-min", type=int, default=128)
    parser.add_argument("--global-q-max", type=int, default=255)
    parser.add_argument("--encoder-weights", type=Path, default=Path("weights/encoder"))
    parser.add_argument("--decoder-weights", type=Path, default=Path("weights/decoder"))
    parser.add_argument("--fq-bin", type=Path, default=Path("./sorteny_fq_qmap"))
    parser.add_argument("--semantic-bin", type=Path, default=Path("./sorteny_semantic_qmap"))
    parser.add_argument("--compressor", type=Path, default=Path("./sorteny_compressor"))
    parser.add_argument("--decompressor", type=Path, default=Path("./sorteny_decompressor"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    args.data_dir = resolve(root, args.data_dir)
    args.output_dir = resolve(root, args.output_dir)
    args.calibration = resolve(root, args.calibration)
    args.semantic_summary = resolve(root, args.semantic_summary)
    args.encoder_weights = resolve(root, args.encoder_weights)
    args.decoder_weights = resolve(root, args.decoder_weights)
    for path in [
        args.data_dir,
        args.calibration,
        args.encoder_weights,
        args.decoder_weights,
        resolve(root, args.fq_bin),
        resolve(root, args.semantic_bin),
        resolve(root, args.compressor),
        resolve(root, args.decompressor),
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    matches = load_semantic_matches(args.semantic_summary)
    crops_all = validate_raw_dataset(args.data_dir, args.bands, args.height, args.width)
    crops_all = [CropInfo(c.path, c.name, matches.get(c.name)) for c in crops_all]
    crops = select_representative_crops(crops_all, args.max_crops)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    local_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    manifest_rows = [
        {
            "crop": c.name,
            "file": str(c.path),
            "semantic_matches": c.semantic_matches if c.semantic_matches is not None else "",
            "selected": int(c.name in {x.name for x in crops}),
        }
        for c in crops_all
    ]
    write_csv(args.output_dir / "dataset_manifest.csv", manifest_rows, ["crop", "file", "semantic_matches", "selected"])

    print(f"Crops selected: {len(crops)}; targets: {args.targets}")
    first_local_768: PipelineResult | None = None
    first_semantic_results: dict[str, PipelineResult] = {}
    first_roi: np.ndarray | None = None

    for crop_i, crop in enumerate(crops):
        crop_root = args.output_dir / "cases" / crop.name
        logs_root = args.output_dir / "logs" / crop.name

        for target in args.targets:
            result = run_fq_case(
                root=root,
                crop=crop,
                case_dir=crop_root / "local_block" / f"target_{target:.1f}".replace(".", "p"),
                logs_dir=logs_root / "local_block",
                args=args,
                env=env,
                name=f"local_target_{target:.1f}".replace(".", "p"),
                fq_args=["--target-psnr", f"{target:.8g}"],
            )
            row, blocks = summarize_local(result, crop, target)
            local_rows.append(row)
            block_rows.extend(blocks)
            if crop_i == 0 and abs(target - 76.8) < 1e-9:
                first_local_768 = result
            print(f"[local] {crop.name} target={target:.1f} global={row['global_psnr']:.4f} reachable={row['reachable_pct']:.2f}%")

        for target in args.targets:
            result, q, status, iterations, _ = find_global_q(
                root=root,
                crop=crop,
                target=target,
                base_dir=crop_root / "global_image" / f"target_{target:.1f}".replace(".", "p"),
                logs_dir=logs_root / "global_image",
                args=args,
                env=env,
            )
            row = summarize_global(result, crop, target, q, status, iterations)
            global_rows.append(row)
            print(f"[global] {crop.name} target={target:.1f} q={q} achieved={row['achieved_psnr']:.4f} status={status}")

        semantic_defs = [
            ("q204", "Q constante 204", "fq", ["--target-from-q", "204"]),
            ("adaptive_s8", "Adaptativo s8", "fq", ["--adaptive-difficulty", "--q-mean", "204", "--adaptive-strength", "8"]),
            (
                "vegetation_focus_bgq128",
                "Vegetation focus bgQ128",
                "semantic",
                ["--semantic-policy", "focus", "--foreground-boost", "16", "--background-q", "128"],
            ),
            (
                "vegetation_focus_bgpen24",
                "Vegetation focus bgPenalty24",
                "semantic",
                ["--semantic-policy", "focus", "--foreground-boost", "16", "--background-penalty", "24"],
            ),
        ]
        semantic_results: dict[str, PipelineResult] = {}
        roi: np.ndarray | None = None
        for key, _, kind, extra in semantic_defs:
            if kind == "fq":
                result = run_fq_case(
                    root=root,
                    crop=crop,
                    case_dir=crop_root / "semantic_roi_background" / key,
                    logs_dir=logs_root / "semantic_roi_background",
                    args=args,
                    env=env,
                    name=f"semantic_{key}",
                    fq_args=extra,
                )
            else:
                result = run_semantic_case(
                    root=root,
                    crop=crop,
                    case_dir=crop_root / "semantic_roi_background" / key,
                    logs_dir=logs_root / "semantic_roi_background",
                    args=args,
                    env=env,
                    name=f"semantic_{key}",
                    policy_args=extra,
                )
                if key == "vegetation_focus_bgq128":
                    _, roi = read_semantic_tsv(result.q_summary, args.height // args.block_size, args.width // args.block_size)
            semantic_results[key] = result
        if roi is None:
            _, roi = read_semantic_tsv(semantic_results["vegetation_focus_bgq128"].q_summary, args.height // args.block_size, args.width // args.block_size)
        for key, result in semantic_results.items():
            row = summarize_semantic(result, crop, key, roi)
            semantic_rows.append(row)
            latent_rows.append(
                {
                    "crop": crop.name,
                    "policy": key,
                    **result.latent_stats,
                }
            )
            for by in range(roi.shape[0]):
                for bx in range(roi.shape[1]):
                    block_rows.append(
                        {
                            "objective_mode": "semantic_roi_background",
                            "crop": crop.name,
                            "target_psnr": "",
                            "policy": key,
                            "block_y": by,
                            "block_x": bx,
                            "q": int(result.qmap_arr[by, bx]),
                            "mse": float(result.block["mse"][by, bx]),
                            "psnr_db": float(result.block["psnr_db"][by, bx]),
                            "target_mse": "",
                            "mse_error": "",
                            "psnr_error_db": "",
                            "roi": int(roi[by, bx]),
                        }
                    )
            print(f"[semantic] {crop.name} {key} global={row['global_psnr']:.4f} roi={row['roi_psnr']:.4f} bg={row['background_psnr']:.4f}")
        if crop_i == 0:
            first_semantic_results = semantic_results
            first_roi = roi

    write_csv(args.output_dir / "local_block_target_results.csv", local_rows, list(local_rows[0].keys()) if local_rows else [])
    write_csv(args.output_dir / "global_image_target_results.csv", global_rows, list(global_rows[0].keys()) if global_rows else [])
    write_csv(args.output_dir / "semantic_roi_target_results.csv", semantic_rows, list(semantic_rows[0].keys()) if semantic_rows else [])
    write_csv(args.output_dir / "latent_proxy_summary.csv", latent_rows, list(latent_rows[0].keys()) if latent_rows else [])
    write_csv(
        args.output_dir / "block_level_errors.csv",
        block_rows,
        ["objective_mode", "crop", "target_psnr", "policy", "block_y", "block_x", "q", "mse", "psnr_db", "target_mse", "mse_error", "psnr_error_db", "roi"],
    )

    summary_rows: list[dict[str, Any]] = []
    for row in local_rows:
        summary_rows.append(
            {
                "objective_mode": "local_block",
                "crop": row["crop"],
                "target_psnr": row["target_psnr"],
                "global_psnr": row["global_psnr"],
                "main_error_db": row["block_psnr_abs_error_mean"],
                "q_mean": row["q_mean"],
                "status": f"reachable={row['reachable_pct']:.2f}%",
            }
        )
    for row in global_rows:
        summary_rows.append(
            {
                "objective_mode": "global_image",
                "crop": row["crop"],
                "target_psnr": row["target_psnr"],
                "global_psnr": row["achieved_psnr"],
                "main_error_db": row["abs_psnr_error_db"],
                "q_mean": row["q_mean"],
                "status": row["status"],
            }
        )
    for row in semantic_rows:
        summary_rows.append(
            {
                "objective_mode": "semantic_roi_background",
                "crop": row["crop"],
                "target_psnr": "",
                "global_psnr": row["global_psnr"],
                "main_error_db": "",
                "q_mean": row["q_mean"],
                "status": row["policy"],
            }
        )
    write_csv(args.output_dir / "psnr_objective_modes_summary.csv", summary_rows, ["objective_mode", "crop", "target_psnr", "global_psnr", "main_error_db", "q_mean", "status"])
    write_json(
        args.output_dir / "psnr_objective_modes_summary.json",
        {
            "config": {
                "crops": [c.name for c in crops],
                "targets": args.targets,
                "global_tolerance_db": args.global_tolerance_db,
                "global_max_iterations": args.global_max_iterations,
            },
            "local_block": local_rows,
            "global_image": global_rows,
            "semantic_roi_background": semantic_rows,
            "latent_proxy": latent_rows,
        },
    )
    write_report(args.output_dir / "psnr_objective_modes_summary.md", local_rows, global_rows, semantic_rows)
    if crops:
        write_maps(args.output_dir, crops[0], first_local_768, first_semantic_results, first_roi)

    qmap_files = list((args.output_dir / "cases").glob("**/qmap.bin"))
    bad_qmaps = [str(p) for p in qmap_files if p.stat().st_size != 1024]
    if bad_qmaps:
        raise RuntimeError(f"Q-maps with invalid size: {bad_qmaps[:5]}")
    print(f"[OK] PSNR objective modes demo: {args.output_dir}")
    print(f"local cases={len(local_rows)} global cases={len(global_rows)} semantic cases={len(semantic_rows)} qmaps={len(qmap_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
