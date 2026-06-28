#!/usr/bin/env python3
"""
Validacion robusta de presets semanticos sobre el dataset Sentinel-2A.

Este script es auxiliar: orquesta binarios C, calcula metricas y genera
evidencia. La decision real del Q-map por preset sigue ocurriendo en
sorteny_semantic_qmap.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import struct
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


MAX_U16 = 65535.0
QMAP_SIDE = 32
QMAP_BYTES = QMAP_SIDE * QMAP_SIDE

USEFUL_PRESETS: list[dict[str, Any]] = [
    {"preset": "vegetation", "index": "NDVI", "bands": "B08,B04", "threshold": 0.40},
    {"preset": "vegetation_green", "index": "GNDVI", "bands": "B08,B03", "threshold": 0.50},
    {"preset": "chlorophyll", "index": "NDCI", "bands": "B05,B04", "threshold": 0.10},
    {"preset": "water_body", "index": "NDWI", "bands": "B03,B08", "threshold": 0.10},
    {"preset": "clouds", "index": "CBY", "bands": "B03,B04[,B11]", "threshold": 0.50},
    {"preset": "dark_regions", "index": "VIS_MEAN", "bands": "B02,B03,B04", "threshold": 0.26},
    {"preset": "local_contrast", "index": "VIS_STD", "bands": "B02,B03,B04", "threshold": 0.035},
    {"preset": "low_ndvi", "index": "NDVI", "bands": "B08,B04", "threshold": 0.15},
    {"preset": "high_ndvi", "index": "NDVI", "bands": "B08,B04", "threshold": 0.50},
    {"preset": "cloud_avoid", "index": "CBY_CLEAR", "bands": "B03,B04[,B11]", "threshold": 0.50},
]

MISSING_BAND_PRESETS: list[dict[str, Any]] = [
    {"preset": "water", "index": "NDMI", "bands": "B08,B11", "missing": "B11"},
    {"preset": "burned", "index": "NBR", "bands": "B08,B12", "missing": "B12"},
    {"preset": "snow", "index": "NDSI", "bands": "B03,B11", "missing": "B11"},
    {"preset": "barren_soil", "index": "BSI", "bands": "B02,B04,B08,B11", "missing": "B11"},
    {"preset": "burned_area", "index": "BAIS2", "bands": "B04,B06,B07,B8A,B12", "missing": "B12"},
]

MANUAL_ROI_PRESET_BASE: dict[str, Any] = {
    "index": "manual_roi",
    "bands": "operator_blocks",
    "threshold": 1.0,
}

SEMANTIC_POLICIES: list[dict[str, Any]] = [
    {
        "policy": "focus_bgpen24",
        "label": "Focus conservador bg penalty 24",
        "args": ["--semantic-policy", "focus", "--foreground-boost", "16", "--background-penalty", "24"],
    },
    {
        "policy": "focus_bgq128",
        "label": "Focus agresivo background Q 128",
        "args": ["--semantic-policy", "focus", "--foreground-boost", "16", "--background-q", "128"],
    },
]

CONTROL_POLICIES = ["q204", "adaptive_s8"]

PROGRESS_FIELDS = [
    "crop",
    "preset",
    "policy",
    "status",
    "started_at",
    "finished_at",
    "elapsed_s",
    "log_path",
    "case_result_path",
]


@dataclass(frozen=True)
class Crop:
    path: Path
    stem: str
    size_bytes: int


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
    return obj


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_csv_row(path: Path, row: dict[str, Any], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(json_ready(row))


def load_done_case(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("case_status") != "done":
        return None
    return data


def control_case_path(outdir: Path, crop: Crop, policy: str) -> Path:
    return outdir / "cases" / crop.stem / "_controls" / policy / "case_result.json"


def semantic_case_path(outdir: Path, crop: Crop, preset: str, policy: str) -> Path:
    return outdir / "cases" / crop.stem / preset / policy / "case_result.json"


def missing_case_path(outdir: Path, crop: Crop, preset: str) -> Path:
    return outdir / "cases" / crop.stem / preset / "missing_bands_audit" / "case_result.json"


def blocks_to_json(blocks: dict[str, np.ndarray]) -> dict[str, Any]:
    return {k: v.tolist() for k, v in blocks.items()}


def blocks_from_json(blocks: dict[str, Any]) -> dict[str, np.ndarray]:
    return {k: np.array(v, dtype=np.float64) for k, v in blocks.items()}


def metrics_to_record(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "global": metrics["global"],
        "blocks": blocks_to_json(metrics["blocks"]),
        "header": metrics["header"],
        "latent": metrics["latent"],
    }


def metrics_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "global": record["global"],
        "blocks": blocks_from_json(record["blocks"]),
        "header": record["header"],
        "latent": record["latent"],
    }


def write_case_result(path: Path, data: dict[str, Any]) -> None:
    data = {"schema": "sorteny_semantic_dataset_case_v1", **data}
    write_json(path, data)


def write_progress(
    progress_path: Path,
    *,
    crop: str,
    preset: str,
    policy: str,
    status: str,
    started_at: str,
    finished_at: str = "",
    elapsed_s: float | str = "",
    log_path: Path | str = "",
    case_result_path: Path | str = "",
) -> None:
    append_csv_row(
        progress_path,
        {
            "crop": crop,
            "preset": preset,
            "policy": policy,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_s": elapsed_s,
            "log_path": str(log_path),
            "case_result_path": str(case_result_path),
        },
        PROGRESS_FIELDS,
    )


def executable(path: Path) -> str:
    if path.is_absolute():
        return str(path)
    if path.parent == Path(".") or str(path.parent) == ".":
        return f"./{path.name}"
    return str(path)


def run_cmd(cmd: list[str], cwd: Path, log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
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
        raise ValueError(f"{path}: {data.size} muestras, esperado {expected}")
    return data.reshape(bands, height, width)


def load_qmap(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size != QMAP_BYTES:
        raise ValueError(f"{path}: Q-map tiene {data.size} bytes, esperado {QMAP_BYTES}")
    return data.reshape(QMAP_SIDE, QMAP_SIDE)


def read_bitstream(path: Path) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        header_bytes = f.read(10)
        if len(header_bytes) != 10:
            raise ValueError(f"{path}: cabecera incompleta")
        bands, height, width, datatype, num_filters = struct.unpack("<5H", header_bytes)
        q_height = height // 16
        q_width = width // 16
        q_size = q_height * q_width
        q_raw = f.read(q_size)
        if len(q_raw) != q_size:
            raise ValueError(f"{path}: Q-map incompleto")
        latents = np.fromfile(f, dtype=np.int32)
    expected_latents = bands * num_filters * q_height * q_width
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
    }
    qmap = np.frombuffer(q_raw, dtype=np.uint8).copy().reshape(q_height, q_width)
    latents = latents.reshape(bands, num_filters, q_height, q_width)
    return header, qmap, latents


def entropy_of_values(values: np.ndarray) -> float:
    flat = values.reshape(-1)
    _, counts = np.unique(flat, return_counts=True)
    probs = counts.astype(np.float64) / float(flat.size)
    return float(-np.sum(probs * np.log2(probs)))


def latent_stats(latents: np.ndarray, input_samples: int) -> dict[str, Any]:
    flat = latents.reshape(-1)
    raw = flat.astype(np.int32, copy=False).tobytes()
    entropy = entropy_of_values(flat)
    zbytes = len(zlib.compress(raw, level=9))
    return {
        "latent_samples": int(flat.size),
        "zero_pct": float(np.mean(flat == 0) * 100.0),
        "entropy_bits_per_symbol": entropy,
        "ideal_bps_per_input_sample": float((entropy * flat.size) / input_samples),
        "zlib_bytes_level9": int(zbytes),
        "zlib_bps_per_input_sample": float((zbytes * 8) / input_samples),
        "mean_abs": float(np.mean(np.abs(flat.astype(np.int64)))),
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
    bands, height, width = original.shape
    block_h = height // block_size
    block_w = width // block_size
    mse = np.zeros((block_h, block_w), dtype=np.float64)
    mae = np.zeros((block_h, block_w), dtype=np.float64)
    max_abs = np.zeros((block_h, block_w), dtype=np.float64)
    exact_pct = np.zeros((block_h, block_w), dtype=np.float64)
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    abs_diff = np.abs(diff)
    for by in range(block_h):
        y0 = by * block_size
        y1 = y0 + block_size
        for bx in range(block_w):
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


def group_summary(metrics: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    if not np.any(mask):
        return {"blocks": 0, "mse": math.nan, "psnr_db": math.nan, "mae": math.nan, "max_abs": math.nan}
    mse = float(np.mean(metrics["mse"][mask]))
    return {
        "blocks": int(np.sum(mask)),
        "mse": mse,
        "psnr_db": psnr_from_mse(mse),
        "mae": float(np.mean(metrics["mae"][mask])),
        "max_abs": float(np.max(metrics["max_abs"][mask])),
    }


def q_summary(qmap: np.ndarray, roi: np.ndarray) -> dict[str, Any]:
    background = ~roi
    return {
        "q_min": int(np.min(qmap)),
        "q_max": int(np.max(qmap)),
        "q_mean": float(np.mean(qmap)),
        "q_roi_mean": float(np.mean(qmap[roi])) if np.any(roi) else math.nan,
        "q_background_mean": float(np.mean(qmap[background])) if np.any(background) else math.nan,
        "q_unique": int(np.unique(qmap).size),
        "qmap_bytes": int(qmap.size),
    }


def read_semantic_tsv(path: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    if len(rows) != QMAP_BYTES:
        raise ValueError(f"{path}: {len(rows)} bloques, esperado {QMAP_BYTES}")

    final_q = np.array([int(r["final_q"]) for r in rows], dtype=np.int32).reshape(QMAP_SIDE, QMAP_SIDE)
    base_q = np.array([int(r["base_q"]) for r in rows], dtype=np.int32).reshape(QMAP_SIDE, QMAP_SIDE)
    matches = np.array([int(r["semantic_match"]) for r in rows], dtype=np.int32).reshape(QMAP_SIDE, QMAP_SIDE)
    missing = sum(1 for r in rows if r["reason"] == "missing_bands")
    reasons: dict[str, int] = {}
    for r in rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    valid_index = [float(r["index_mean"]) for r in rows if r["index_mean"] != "nan"]
    return {
        "rows": rows,
        "final_q": final_q,
        "base_q": base_q,
        "roi": matches.astype(bool),
        "semantic_matches": int(np.sum(matches)),
        "missing_bands": int(missing),
        "reason_counts": reasons,
        "index_mean": float(np.mean(valid_index)) if valid_index else math.nan,
        "index_min": float(np.min(valid_index)) if valid_index else math.nan,
        "index_max": float(np.max(valid_index)) if valid_index else math.nan,
    }


def roi_group(roi_blocks: int) -> str:
    if roi_blocks == 0:
        return "no_roi"
    if roi_blocks == QMAP_BYTES:
        return "all_roi"
    pct = 100.0 * roi_blocks / QMAP_BYTES
    if pct < 10.0:
        return "low_roi"
    if pct <= 70.0:
        return "mid_roi"
    return "high_roi"


def status_from_semantic(roi_blocks: int, missing_bands: int) -> str:
    if missing_bands > 0:
        return "missing_bands"
    if roi_blocks == 0:
        return "no_roi"
    if roi_blocks == QMAP_BYTES:
        return "all_roi"
    return "valid"


def normalize_to_u8(data: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    arr = data.astype(np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.min(finite)) if vmin is None else float(vmin)
    hi = float(np.max(finite)) if vmax is None else float(vmax)
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    x = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(x * 255.0).astype(np.uint8)


def diverging_rgb(data: np.ndarray, max_abs: float | None = None) -> np.ndarray:
    arr = data.astype(np.float64)
    finite = arr[np.isfinite(arr)]
    if max_abs is None:
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


def upsample(image: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


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


def write_representative_maps(
    maps_dir: Path,
    preset: str,
    crop_stem: str,
    roi: np.ndarray,
    adaptive_qmap: np.ndarray,
    semantic_qmap: np.ndarray,
    adaptive_metrics: dict[str, np.ndarray],
    semantic_metrics: dict[str, np.ndarray],
    semantic_latents: np.ndarray,
) -> None:
    out = maps_dir / preset / crop_stem
    write_map_pair(out, "roi_mask.pgm", roi.astype(np.uint8) * 255)
    write_map_pair(out, "qmap_adaptive_s8.pgm", normalize_to_u8(adaptive_qmap, 128, 255))
    write_map_pair(out, "qmap_semantic_focus.pgm", normalize_to_u8(semantic_qmap, 128, 255))
    write_map_pair(out, "delta_q_semantic_minus_adaptive.ppm", diverging_rgb(semantic_qmap.astype(float) - adaptive_qmap.astype(float)), rgb=True)
    write_map_pair(out, "delta_mse_semantic_minus_adaptive.ppm", diverging_rgb(semantic_metrics["mse"] - adaptive_metrics["mse"]), rgb=True)
    write_map_pair(out, "delta_psnr_semantic_minus_adaptive.ppm", diverging_rgb(semantic_metrics["psnr_db"] - adaptive_metrics["psnr_db"]), rgb=True)
    write_map_pair(out, "latent_zero_pct_semantic.pgm", normalize_to_u8(latent_zero_map(semantic_latents), 0, 100))
    write_map_pair(out, "latent_entropy_semantic.pgm", normalize_to_u8(latent_entropy_map(semantic_latents)))


def make_band_map(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "band_name\tindex",
                "B02\t0",
                "B03\t1",
                "B04\t2",
                "B05\t3",
                "B06\t4",
                "B07\t5",
                "B08\t6",
                "B8A\t7",
                "",
            ]
        ),
        encoding="utf-8",
    )


def manual_roi_name(args: argparse.Namespace) -> str:
    if args.manual_roi_rect:
        y0, x0, y1, x1 = args.manual_roi_rect
        return f"manual_rect_y{y0}_{y1}_x{x0}_{x1}"
    return f"manual_{args.manual_roi_pattern}"


def manual_roi_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.include_manual_roi:
        return []
    name = manual_roi_name(args)
    return [
        {
            "preset": name,
            **MANUAL_ROI_PRESET_BASE,
            "roi_source": "manual",
            "roi_pattern": args.manual_roi_pattern,
            "roi_rect": args.manual_roi_rect,
        }
    ]


def build_manual_roi(args: argparse.Namespace) -> np.ndarray:
    side = args.height // args.block_size
    width = args.width // args.block_size
    roi = np.zeros((side, width), dtype=np.uint8)
    if args.manual_roi_rect:
        y0, x0, y1, x1 = args.manual_roi_rect
        if not (0 <= y0 < y1 <= side and 0 <= x0 < x1 <= width):
            raise ValueError(f"--manual-roi-rect fuera de rango: {args.manual_roi_rect}, grid={side}x{width}")
        roi[y0:y1, x0:x1] = 1
    elif args.manual_roi_pattern == "center":
        y0 = side // 4
        y1 = side - y0
        x0 = width // 4
        x1 = width - x0
        roi[y0:y1, x0:x1] = 1
    elif args.manual_roi_pattern == "top_left":
        roi[: side // 2, : width // 2] = 1
    elif args.manual_roi_pattern == "bottom_right":
        roi[side // 2 :, width // 2 :] = 1
    else:
        raise ValueError(f"Patron manual no soportado: {args.manual_roi_pattern}")
    return roi


def write_roi_tsv(path: Path, roi: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["block_y", "block_x"], delimiter="\t")
        writer.writeheader()
        for by, bx in np.argwhere(roi != 0):
            writer.writerow({"block_y": int(by), "block_x": int(bx)})


def write_manual_roi_artifacts(args: argparse.Namespace, outdir: Path) -> dict[str, Any] | None:
    specs = manual_roi_specs(args)
    if not specs:
        return None
    name = specs[0]["preset"]
    roi = build_manual_roi(args)
    roi_dir = outdir / "manual_rois"
    roi_map = roi_dir / f"{name}.bin"
    roi_tsv = roi_dir / f"{name}.tsv"
    roi_summary = roi_dir / f"{name}_summary.json"
    payload = roi.reshape(-1).astype(np.uint8).tobytes()
    roi_dir.mkdir(parents=True, exist_ok=True)
    roi_map.write_bytes(payload)
    write_roi_tsv(roi_tsv, roi)
    summary = {
        "name": name,
        "roi_map": str(roi_map),
        "roi_tsv": str(roi_tsv),
        "qmap_bytes": len(payload),
        "roi_blocks": int(np.count_nonzero(roi)),
        "background_blocks": int(roi.size - np.count_nonzero(roi)),
        "roi_pct": float(np.count_nonzero(roi) * 100.0 / roi.size),
        "pattern": args.manual_roi_pattern,
        "rect": args.manual_roi_rect,
    }
    write_json(roi_summary, summary)
    summary["roi_summary_json"] = str(roi_summary)
    return summary


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def resolved_existing_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def path_check(root: Path, path: Path, *, executable_required: bool = False, directory: bool = False) -> dict[str, Any]:
    resolved = resolved_existing_path(root, path)
    exists = resolved.exists()
    ok = exists
    if exists and directory:
        ok = resolved.is_dir()
    if exists and executable_required:
        ok = resolved.is_file() and (resolved.stat().st_mode & 0o111) != 0
    return {
        "path": str(resolved),
        "exists": bool(exists),
        "ok": bool(ok),
        "is_file": bool(resolved.is_file()) if exists else False,
        "is_dir": bool(resolved.is_dir()) if exists else False,
        "executable": bool((resolved.stat().st_mode & 0o111) != 0) if exists and resolved.is_file() else False,
    }


def build_run_manifest(
    args: argparse.Namespace,
    root: Path,
    outdir: Path,
    dataset_dir: Path,
    crops_all: list[Crop],
    selected_crops: list[Crop] | None = None,
) -> dict[str, Any]:
    disk = shutil.disk_usage(outdir)
    return {
        "created_at": now_iso(),
        "argv": sys.argv,
        "repo_root": str(root),
        "output_dir": str(outdir),
        "dataset_dir": str(dataset_dir),
        "mode": args.mode,
        "resume": bool(args.resume),
        "keep_heavy": args.keep_heavy,
        "max_crops": args.max_crops,
        "smoke_crops": args.smoke_crops,
        "valid_raws_total": len(crops_all),
        "selected_crops": [c.stem for c in selected_crops] if selected_crops is not None else [],
        "image_config": {
            "bands": args.bands,
            "height": args.height,
            "width": args.width,
            "block_size": args.block_size,
            "expected_raw_bytes": args.bands * args.height * args.width * 2,
        },
        "quality_config": {
            "lambda_value": args.lambda_value,
            "max_lambda": args.max_lambda,
            "q_mean": args.q_mean,
            "adaptive_strength": args.adaptive_strength,
        },
        "paths": {
            "calibration": str(args.calibration),
            "encoder_weights": str(args.encoder_weights),
            "decoder_weights": str(args.decoder_weights),
            "fq_bin": str(args.fq_bin),
            "semantic_bin": str(args.semantic_bin),
            "compressor": str(args.compressor),
            "decompressor": str(args.decompressor),
        },
        "presets": USEFUL_PRESETS,
        "manual_roi_presets": manual_roi_specs(args),
        "missing_band_presets": MISSING_BAND_PRESETS,
        "semantic_policies": SEMANTIC_POLICIES,
        "control_policies": CONTROL_POLICIES,
        "manual_roi": {
            "enabled": bool(args.include_manual_roi),
            "pattern": args.manual_roi_pattern,
            "rect": args.manual_roi_rect,
        },
        "disk": {
            "total_bytes": int(disk.total),
            "used_bytes": int(disk.used),
            "free_bytes": int(disk.free),
        },
    }


def write_run_manifest(
    args: argparse.Namespace,
    root: Path,
    outdir: Path,
    dataset_dir: Path,
    crops_all: list[Crop],
    selected_crops: list[Crop] | None = None,
) -> None:
    write_json(outdir / "run_manifest.json", build_run_manifest(args, root, outdir, dataset_dir, crops_all, selected_crops))


def run_preflight(
    args: argparse.Namespace,
    root: Path,
    outdir: Path,
    dataset_dir: Path,
    crops_all: list[Crop],
    manifest_rows: list[dict[str, Any]],
    qmap_dir: Path,
    tsv_dir: Path,
    logs_dir: Path,
    band_map: Path,
) -> int:
    checks: list[dict[str, Any]] = []
    checks.append({"name": "dataset_dir", **path_check(root, dataset_dir, directory=True)})
    checks.append({"name": "calibration", **path_check(root, args.calibration)})
    checks.append({"name": "encoder_weights", **path_check(root, args.encoder_weights, directory=True)})
    checks.append({"name": "decoder_weights", **path_check(root, args.decoder_weights, directory=True)})
    checks.append({"name": "fq_bin", **path_check(root, args.fq_bin, executable_required=True)})
    checks.append({"name": "semantic_bin", **path_check(root, args.semantic_bin, executable_required=True)})
    checks.append({"name": "compressor", **path_check(root, args.compressor, executable_required=True)})
    checks.append({"name": "decompressor", **path_check(root, args.decompressor, executable_required=True)})

    write_csv(outdir / "dataset_manifest.csv", manifest_rows, ["file", "crop", "size_bytes", "expected_bytes", "status"])
    make_band_map(band_map)
    probe = outdir / ".write_probe"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink(missing_ok=True)

    control_rows: list[dict[str, Any]] = []
    for mode in CONTROL_POLICIES:
        qmap = qmap_dir / "controls" / f"{mode}.bin"
        tsv = tsv_dir / "controls" / f"{mode}.tsv"
        qmap.parent.mkdir(parents=True, exist_ok=True)
        tsv.parent.mkdir(parents=True, exist_ok=True)
        cmd = fq_qmap_cmd(args, qmap, tsv, mode)
        elapsed = run_cmd(cmd, root, logs_dir / "preflight" / f"qmap_{mode}.log")
        qmap_size = qmap.stat().st_size if qmap.exists() else 0
        control_rows.append(
            {
                "control": mode,
                "qmap": str(qmap),
                "summary_tsv": str(tsv),
                "qmap_bytes": qmap_size,
                "elapsed_s": elapsed,
                "status": "ok" if qmap_size == QMAP_BYTES else "bad_qmap_size",
            }
        )

    manual_roi = write_manual_roi_artifacts(args, outdir)
    manual_rows: list[dict[str, Any]] = []
    if manual_roi is not None:
        roi_map = Path(manual_roi["roi_map"])
        preset = manual_roi["name"]
        for policy_spec in SEMANTIC_POLICIES:
            policy = policy_spec["policy"]
            qmap = qmap_dir / policy / preset / f"{preset}.bin"
            tsv = tsv_dir / policy / preset / f"{preset}.tsv"
            qmap.parent.mkdir(parents=True, exist_ok=True)
            tsv.parent.mkdir(parents=True, exist_ok=True)
            cmd = manual_qmap_cmd(args, roi_map, qmap, tsv, policy_spec["args"])
            elapsed = run_cmd(cmd, root, logs_dir / "preflight" / "manual_roi" / f"qmap_{policy}.log")
            qmap_size = qmap.stat().st_size if qmap.exists() else 0
            manual_rows.append(
                {
                    "preset": preset,
                    "policy": policy,
                    "qmap": str(qmap),
                    "summary_tsv": str(tsv),
                    "qmap_bytes": qmap_size,
                    "elapsed_s": elapsed,
                    "status": "ok" if qmap_size == QMAP_BYTES else "bad_qmap_size",
                }
            )

    disk = shutil.disk_usage(outdir)
    manual_count = len(manual_roi_specs(args))
    evaluated_preset_count = len(USEFUL_PRESETS) + manual_count
    expected_rows_full = len(crops_all) * evaluated_preset_count * (len(SEMANTIC_POLICIES) + len(CONTROL_POLICIES))
    estimated_runtime_cases = len(crops_all) * (
        len(CONTROL_POLICIES) + evaluated_preset_count * len(SEMANTIC_POLICIES)
    )
    summary = {
        "status": "ok",
        "created_at": now_iso(),
        "valid_raws": len(crops_all),
        "expected_valid_raws": 120,
        "all_raws_valid": len(crops_all) == 120 and all(r["status"] == "ok" for r in manifest_rows),
        "path_checks": checks,
        "control_qmaps": control_rows,
        "manual_roi": manual_roi,
        "manual_qmaps": manual_rows,
        "space": {
            "free_bytes": int(disk.free),
            "free_gb": float(disk.free / (1024**3)),
            "estimated_checkpoint_gb": 2.0,
            "note": "Estimacion conservadora con keep-heavy=representative; los artefactos temporales se eliminan.",
        },
        "full_run_shape": {
            "crops": len(crops_all),
            "useful_presets": len(USEFUL_PRESETS),
            "manual_roi_presets": manual_count,
            "semantic_policies": len(SEMANTIC_POLICIES),
            "control_policies": len(CONTROL_POLICIES),
            "result_rows_expected": expected_rows_full,
            "pipeline_cases_expected": estimated_runtime_cases,
        },
    }
    if any(not c["ok"] for c in checks):
        summary["status"] = "failed_path_checks"
    if any(r["status"] != "ok" for r in control_rows):
        summary["status"] = "failed_control_qmaps"
    if any(r["status"] != "ok" for r in manual_rows):
        summary["status"] = "failed_manual_qmaps"
    if not summary["all_raws_valid"]:
        summary["status"] = "failed_dataset_validation"

    write_json(outdir / "preflight_summary.json", summary)
    write_run_manifest(args, root, outdir, dataset_dir, crops_all, [])
    print(f"[OK] Preflight completado: {outdir}" if summary["status"] == "ok" else f"[FAIL] Preflight: {summary['status']}")
    print(f"RAW validos: {len(crops_all)}; espacio libre: {summary['space']['free_gb']:.2f} GiB")
    return 0 if summary["status"] == "ok" else 2


def semantic_qmap_cmd(
    args: argparse.Namespace,
    raw: Path,
    preset: str,
    qmap: Path,
    tsv: Path,
    policy_args: list[str],
    *,
    band_map: Path | None = None,
    threshold: float | None = None,
) -> list[str]:
    cmd = [
        executable(args.semantic_bin),
        "--input",
        str(raw),
        "--calibration",
        str(args.calibration),
        "--preset",
        preset,
        "--output-qmap",
        str(qmap),
        "--summary-tsv",
        str(tsv),
        "--bands",
        str(args.bands),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--q-mean",
        str(args.q_mean),
        "--adaptive-strength",
        str(args.adaptive_strength),
        *policy_args,
    ]
    if band_map is not None:
        cmd.extend(["--band-map", str(band_map)])
    else:
        cmd.extend(["--band-layout", "sentinel2-8"])
    if threshold is not None:
        cmd.extend(["--threshold", str(threshold)])
    return cmd


def manual_qmap_cmd(
    args: argparse.Namespace,
    roi_map: Path,
    qmap: Path,
    tsv: Path,
    policy_args: list[str],
) -> list[str]:
    return [
        executable(args.semantic_bin),
        "--calibration",
        str(args.calibration),
        "--preset",
        "manual",
        "--roi-map",
        str(roi_map),
        "--output-qmap",
        str(qmap),
        "--summary-tsv",
        str(tsv),
        "--bands",
        str(args.bands),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--q-mean",
        str(args.q_mean),
        "--adaptive-strength",
        str(args.adaptive_strength),
        *policy_args,
    ]


def fq_qmap_cmd(args: argparse.Namespace, qmap: Path, tsv: Path, mode: str) -> list[str]:
    cmd = [
        executable(args.fq_bin),
        "--calibration",
        str(args.calibration),
        "--output-qmap",
        str(qmap),
        "--summary-tsv",
        str(tsv),
    ]
    if mode == "q204":
        cmd.extend(["--target-from-q", "204"])
    elif mode == "adaptive_s8":
        cmd.extend(["--adaptive-difficulty", "--q-mean", str(args.q_mean), "--adaptive-strength", str(args.adaptive_strength)])
    else:
        raise ValueError(mode)
    return cmd


def run_pipeline(
    args: argparse.Namespace,
    root: Path,
    raw: Path,
    qmap: Path,
    bitstream: Path,
    recon: Path,
    logs_dir: Path,
    key: str,
) -> tuple[float, float]:
    bitstream.parent.mkdir(parents=True, exist_ok=True)
    recon.parent.mkdir(parents=True, exist_ok=True)
    compress_cmd = [
        executable(args.compressor),
        str(raw),
        f"{args.lambda_value:.8g}",
        str(bitstream),
        str(args.encoder_weights),
        f"{args.max_lambda:.8g}",
        str(qmap),
    ]
    compress_s = run_cmd(compress_cmd, root, logs_dir / f"compress_{key}.log")
    decompress_cmd = [
        executable(args.decompressor),
        str(bitstream),
        str(recon),
        str(args.decoder_weights),
        f"{args.max_lambda:.8g}",
    ]
    decompress_s = run_cmd(decompress_cmd, root, logs_dir / f"decompress_{key}.log")
    return compress_s, decompress_s


def pipeline_metrics(
    original: np.ndarray,
    recon_path: Path,
    bitstream_path: Path,
    qmap_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    recon = load_raw_u16(recon_path, args.bands, args.height, args.width)
    global_metrics = metrics_for_arrays(original, recon)
    blocks = block_metrics(original, recon, args.block_size)
    header, embedded_qmap, latents = read_bitstream(bitstream_path)
    qmap = load_qmap(qmap_path)
    if not np.array_equal(qmap, embedded_qmap):
        raise ValueError(f"{bitstream_path}: Q-map embebido no coincide con {qmap_path}")
    return {
        "global": global_metrics,
        "blocks": blocks,
        "header": header,
        "qmap": qmap,
        "latents": latents,
        "latent": latent_stats(latents, args.bands * args.height * args.width),
    }


def run_or_load_control_case(
    args: argparse.Namespace,
    root: Path,
    outdir: Path,
    progress_path: Path,
    crop: Crop,
    control: str,
    original: np.ndarray,
    qmap_path: Path,
    bitstream: Path,
    recon: Path,
    logs_dir: Path,
    keep: bool,
) -> tuple[dict[str, Any], dict[str, float], bool]:
    case_path = control_case_path(outdir, crop, control)
    if args.resume:
        cached = load_done_case(case_path)
        if cached is not None:
            print(f"  RESUME control {crop.stem}/{control}", flush=True)
            ts = now_iso()
            write_progress(
                progress_path,
                crop=crop.stem,
                preset="_control",
                policy=control,
                status="done",
                started_at=ts,
                finished_at=ts,
                elapsed_s=0.0,
                log_path="resume-cache",
                case_result_path=case_path,
            )
            return metrics_from_record(cached["metrics"]), cached.get("timings", {"compress_s": 0.0, "decompress_s": 0.0}), True

    started = now_iso()
    t0 = time.perf_counter()
    write_progress(
        progress_path,
        crop=crop.stem,
        preset="_control",
        policy=control,
        status="running",
        started_at=started,
        log_path=logs_dir,
        case_result_path=case_path,
    )
    try:
        if not bitstream.exists() or not recon.exists():
            c_s, d_s = run_pipeline(
                args,
                root,
                crop.path,
                qmap_path,
                bitstream,
                recon,
                logs_dir,
                f"{crop.stem}_{control}",
            )
            timings = {"compress_s": c_s, "decompress_s": d_s}
        else:
            timings = {"compress_s": 0.0, "decompress_s": 0.0}
        metrics = pipeline_metrics(original, recon, bitstream, qmap_path, args)
        write_case_result(
            case_path,
            {
                "case_status": "done",
                "case_kind": "control",
                "crop": crop.stem,
                "policy": control,
                "timings": timings,
                "metrics": metrics_to_record(metrics),
                "artifacts": {
                    "qmap_path": str(qmap_path),
                    "bitstream_path": str(bitstream) if keep else "",
                    "reconstruction_path": str(recon) if keep else "",
                    "logs_dir": str(logs_dir),
                },
                "started_at": started,
                "finished_at": now_iso(),
            },
        )
        write_progress(
            progress_path,
            crop=crop.stem,
            preset="_control",
            policy=control,
            status="done",
            started_at=started,
            finished_at=now_iso(),
            elapsed_s=time.perf_counter() - t0,
            log_path=logs_dir,
            case_result_path=case_path,
        )
        return metrics, timings, False
    except Exception as exc:
        write_case_result(
            case_path,
            {
                "case_status": "failed",
                "case_kind": "control",
                "crop": crop.stem,
                "policy": control,
                "error": repr(exc),
                "started_at": started,
                "finished_at": now_iso(),
            },
        )
        write_progress(
            progress_path,
            crop=crop.stem,
            preset="_control",
            policy=control,
            status="failed",
            started_at=started,
            finished_at=now_iso(),
            elapsed_s=time.perf_counter() - t0,
            log_path=logs_dir,
            case_result_path=case_path,
        )
        raise


def make_result_row(
    crop: Crop,
    preset_spec: dict[str, Any],
    policy: str,
    policy_kind: str,
    semantic: dict[str, Any],
    metrics: dict[str, Any],
    adaptive_metrics: dict[str, Any],
    qmap: np.ndarray,
    adaptive_qmap: np.ndarray,
    timings: dict[str, float],
    artifact_paths: dict[str, str],
) -> dict[str, Any]:
    roi = semantic["roi"]
    background = ~roi
    roi_blocks = int(np.sum(roi))
    bg_blocks = int(np.sum(background))
    status = status_from_semantic(roi_blocks, int(semantic["missing_bands"]))
    group = roi_group(roi_blocks)

    global_m = metrics["global"]
    block_m = metrics["blocks"]
    latent_m = metrics["latent"]
    adaptive_global = adaptive_metrics["global"]
    adaptive_block = adaptive_metrics["blocks"]
    adaptive_latent = adaptive_metrics["latent"]

    roi_m = group_summary(block_m, roi)
    bg_m = group_summary(block_m, background)
    adaptive_roi = group_summary(adaptive_block, roi)
    adaptive_bg = group_summary(adaptive_block, background)
    q_m = q_summary(qmap, roi)
    adaptive_q = q_summary(adaptive_qmap, roi)

    return {
        "crop": crop.stem,
        "file": str(crop.path),
        "preset": preset_spec["preset"],
        "index": preset_spec["index"],
        "bands": preset_spec["bands"],
        "threshold": preset_spec["threshold"],
        "policy": policy,
        "policy_kind": policy_kind,
        "status": status,
        "roi_group": group,
        "roi_blocks": roi_blocks,
        "background_blocks": bg_blocks,
        "roi_pct": float(100.0 * roi_blocks / QMAP_BYTES),
        "missing_bands": int(semantic["missing_bands"]),
        "index_mean": semantic["index_mean"],
        "index_min": semantic["index_min"],
        "index_max": semantic["index_max"],
        "global_mse": global_m["mse"],
        "global_psnr_db": global_m["psnr_db"],
        "roi_mse": roi_m["mse"],
        "roi_psnr_db": roi_m["psnr_db"],
        "background_mse": bg_m["mse"],
        "background_psnr_db": bg_m["psnr_db"],
        "adaptive_global_psnr_db": adaptive_global["psnr_db"],
        "adaptive_roi_psnr_db": adaptive_roi["psnr_db"],
        "adaptive_background_psnr_db": adaptive_bg["psnr_db"],
        "delta_global_psnr_vs_adaptive": global_m["psnr_db"] - adaptive_global["psnr_db"],
        "delta_roi_psnr_vs_adaptive": roi_m["psnr_db"] - adaptive_roi["psnr_db"] if roi_blocks else math.nan,
        "delta_background_psnr_vs_adaptive": bg_m["psnr_db"] - adaptive_bg["psnr_db"] if bg_blocks else math.nan,
        "q_min": q_m["q_min"],
        "q_max": q_m["q_max"],
        "q_mean": q_m["q_mean"],
        "q_roi_mean": q_m["q_roi_mean"],
        "q_background_mean": q_m["q_background_mean"],
        "q_unique": q_m["q_unique"],
        "q_mean_delta_vs_adaptive": q_m["q_mean"] - adaptive_q["q_mean"],
        "latent_zero_pct": latent_m["zero_pct"],
        "latent_entropy_bits_per_symbol": latent_m["entropy_bits_per_symbol"],
        "latent_ideal_bps_per_input_sample": latent_m["ideal_bps_per_input_sample"],
        "latent_zlib_bps_per_input_sample": latent_m["zlib_bps_per_input_sample"],
        "latent_mean_abs": latent_m["mean_abs"],
        "latent_unique_values": latent_m["unique_values"],
        "delta_zero_pct_vs_adaptive": latent_m["zero_pct"] - adaptive_latent["zero_pct"],
        "delta_entropy_vs_adaptive": latent_m["entropy_bits_per_symbol"] - adaptive_latent["entropy_bits_per_symbol"],
        "delta_zlib_bps_vs_adaptive": latent_m["zlib_bps_per_input_sample"] - adaptive_latent["zlib_bps_per_input_sample"],
        "compress_s": timings.get("compress_s", math.nan),
        "decompress_s": timings.get("decompress_s", math.nan),
        **artifact_paths,
    }


def run_or_load_semantic_case(
    args: argparse.Namespace,
    root: Path,
    outdir: Path,
    progress_path: Path,
    crop: Crop,
    preset_spec: dict[str, Any],
    policy: str,
    sem: dict[str, Any],
    semantic_qmap: np.ndarray,
    adaptive_qmap: np.ndarray,
    adaptive_metrics: dict[str, Any],
    original: np.ndarray,
    qmap: Path,
    bitstream: Path,
    recon: Path,
    logs_dir: Path,
    key: str,
    artifact_paths: dict[str, str],
    keep: bool,
    policy_kind: str = "semantic_focus",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, bool]:
    preset = preset_spec["preset"]
    case_path = semantic_case_path(outdir, crop, preset, policy)
    if args.resume:
        cached = load_done_case(case_path)
        if cached is not None:
            print(f"  RESUME semantic {crop.stem}/{preset}/{policy}", flush=True)
            ts = now_iso()
            write_progress(
                progress_path,
                crop=crop.stem,
                preset=preset,
                policy=policy,
                status="done",
                started_at=ts,
                finished_at=ts,
                elapsed_s=0.0,
                log_path="resume-cache",
                case_result_path=case_path,
            )
            return cached["row"], cached["latent_row"], None, True

    started = now_iso()
    t0 = time.perf_counter()
    write_progress(
        progress_path,
        crop=crop.stem,
        preset=preset,
        policy=policy,
        status="running",
        started_at=started,
        log_path=logs_dir,
        case_result_path=case_path,
    )
    try:
        c_s, d_s = run_pipeline(args, root, crop.path, qmap, bitstream, recon, logs_dir, key)
        metrics = pipeline_metrics(original, recon, bitstream, qmap, args)
        row = make_result_row(
            crop,
            preset_spec,
            policy,
            policy_kind,
            sem,
            metrics,
            adaptive_metrics,
            semantic_qmap,
            adaptive_qmap,
            {"compress_s": c_s, "decompress_s": d_s},
            artifact_paths,
        )
        latent_row = {
            "crop": crop.stem,
            "preset": preset,
            "policy": policy,
            "status": row["status"],
            "roi_group": row["roi_group"],
            "zero_pct": row["latent_zero_pct"],
            "entropy_bits_per_symbol": row["latent_entropy_bits_per_symbol"],
            "ideal_bps_per_input_sample": row["latent_ideal_bps_per_input_sample"],
            "zlib_bps_per_input_sample": row["latent_zlib_bps_per_input_sample"],
            "mean_abs": row["latent_mean_abs"],
            "unique_values": row["latent_unique_values"],
            "delta_zero_pct_vs_adaptive": row["delta_zero_pct_vs_adaptive"],
            "delta_entropy_vs_adaptive": row["delta_entropy_vs_adaptive"],
        }
        write_case_result(
            case_path,
            {
                "case_status": "done",
                "case_kind": policy_kind,
                "crop": crop.stem,
                "preset": preset,
                "policy": policy,
                "row": row,
                "latent_row": latent_row,
                "timings": {"compress_s": c_s, "decompress_s": d_s},
                "artifacts": artifact_paths,
                "started_at": started,
                "finished_at": now_iso(),
            },
        )
        write_progress(
            progress_path,
            crop=crop.stem,
            preset=preset,
            policy=policy,
            status="done",
            started_at=started,
            finished_at=now_iso(),
            elapsed_s=time.perf_counter() - t0,
            log_path=logs_dir,
            case_result_path=case_path,
        )
        return row, latent_row, metrics, False
    except Exception as exc:
        write_case_result(
            case_path,
            {
                "case_status": "failed",
                "case_kind": "semantic_focus",
                "crop": crop.stem,
                "preset": preset,
                "policy": policy,
                "error": repr(exc),
                "started_at": started,
                "finished_at": now_iso(),
            },
        )
        write_progress(
            progress_path,
            crop=crop.stem,
            preset=preset,
            policy=policy,
            status="failed",
            started_at=started,
            finished_at=now_iso(),
            elapsed_s=time.perf_counter() - t0,
            log_path=logs_dir,
            case_result_path=case_path,
        )
        raise


def mean(values: list[float]) -> float:
    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else math.nan


def median(values: list[float]) -> float:
    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(np.median(finite)) if finite else math.nan


def summarize_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    focus_kinds = {"semantic_focus", "semantic_focus_threshold_sweep", "semantic_preserve_roi", "manual_focus"}
    keys = sorted({(r["preset"], r["policy"], r["roi_group"]) for r in rows if r["policy_kind"] in focus_kinds})
    for preset, policy, group in keys:
        items = [r for r in rows if r["preset"] == preset and r["policy"] == policy and r["roi_group"] == group]
        if not items:
            continue
        roi_deltas = [r["delta_roi_psnr_vs_adaptive"] for r in items]
        bg_deltas = [r["delta_background_psnr_vs_adaptive"] for r in items]
        q_deltas = [r["q_mean_delta_vs_adaptive"] for r in items]
        entropy_deltas = [r["delta_entropy_vs_adaptive"] for r in items]
        zero_deltas = [r["delta_zero_pct_vs_adaptive"] for r in items]
        valid_roi = [v for v in roi_deltas if isinstance(v, (int, float)) and math.isfinite(float(v))]
        valid_bg = [v for v in bg_deltas if isinstance(v, (int, float)) and math.isfinite(float(v))]
        out.append(
            {
                "preset": preset,
                "policy": policy,
                "roi_group": group,
                "runs": len(items),
                "valid_runs": sum(1 for r in items if r["status"] == "valid"),
                "roi_pct_mean": mean([r["roi_pct"] for r in items]),
                "global_psnr_mean": mean([r["global_psnr_db"] for r in items]),
                "roi_delta_psnr_mean": mean(roi_deltas),
                "roi_delta_psnr_median": median(roi_deltas),
                "roi_delta_psnr_worst": float(np.min(valid_roi)) if valid_roi else math.nan,
                "background_delta_psnr_mean": mean(bg_deltas),
                "background_delta_psnr_median": median(bg_deltas),
                "background_delta_psnr_worst_degradation": float(np.max(valid_bg)) if valid_bg else math.nan,
                "q_mean_delta_mean": mean(q_deltas),
                "entropy_delta_mean": mean(entropy_deltas),
                "zero_delta_mean": mean(zero_deltas),
                "roi_kept_or_improved_pct": float(np.mean([v >= -0.05 for v in valid_roi]) * 100.0) if valid_roi else math.nan,
                "background_degraded_ge_0_3db_pct": float(np.mean([v <= -0.3 for v in valid_bg]) * 100.0) if valid_bg else math.nan,
                "q_mean_lower_pct": float(np.mean([v < 0.0 for v in q_deltas]) * 100.0) if q_deltas else math.nan,
                "latent_simpler_pct": float(
                    np.mean(
                        [
                            (r["delta_entropy_vs_adaptive"] < 0.0) or (r["delta_zero_pct_vs_adaptive"] > 0.0)
                            for r in items
                        ]
                    )
                    * 100.0
                ),
                "success_runs": sum(
                    1
                    for r in items
                    if r["status"] == "valid"
                    and math.isfinite(float(r["delta_roi_psnr_vs_adaptive"]))
                    and r["delta_roi_psnr_vs_adaptive"] >= -0.05
                    and math.isfinite(float(r["delta_background_psnr_vs_adaptive"]))
                    and r["delta_background_psnr_vs_adaptive"] <= -0.3
                    and r["q_mean_delta_vs_adaptive"] < 0.0
                    and ((r["delta_entropy_vs_adaptive"] < 0.0) or (r["delta_zero_pct_vs_adaptive"] > 0.0))
                ),
            }
        )
    return out


def choose_smoke_crops(prescan_rows: list[dict[str, Any]], crops: list[Crop], smoke_crops: int) -> list[Crop]:
    by_stem = {c.stem: c for c in crops}
    selected: list[str] = []

    def add(stem: str) -> None:
        if stem not in selected:
            selected.append(stem)

    for preset in [p["preset"] for p in USEFUL_PRESETS]:
        rows = [r for r in prescan_rows if r["preset"] == preset]
        if not rows:
            continue
        mid = [r for r in rows if r["roi_group"] == "mid_roi"]
        if mid:
            add(min(mid, key=lambda r: abs(r["roi_pct"] - 40.0))["crop"])
        low = [r for r in rows if r["roi_group"] in {"low_roi", "no_roi"}]
        if low:
            add(max(low, key=lambda r: r["roi_pct"])["crop"])
        high = [r for r in rows if r["roi_group"] in {"high_roi", "all_roi"}]
        if high:
            add(min(high, key=lambda r: abs(r["roi_pct"] - 85.0))["crop"])
        if len(selected) >= smoke_crops:
            break

    if len(selected) < smoke_crops:
        ranked = sorted(prescan_rows, key=lambda r: (r["roi_group"] != "mid_roi", abs(r["roi_pct"] - 40.0)))
        for row in ranked:
            add(row["crop"])
            if len(selected) >= smoke_crops:
                break

    if len(selected) < smoke_crops:
        for crop in crops:
            add(crop.stem)
            if len(selected) >= smoke_crops:
                break
    return [by_stem[s] for s in selected[:smoke_crops]]


def write_recommendations(path: Path, group_rows: list[dict[str, Any]], missing_rows: list[dict[str, Any]], mode: str) -> None:
    lines = [
        "# Semantic Preset Dataset Recommendations",
        "",
        f"Modo ejecutado: `{mode}`.",
        "",
        "La prioridad interpretativa es `mid_roi`, porque permite separar zona de interes y fondo.",
        "",
        "## Mid ROI Summary",
        "",
        "| Preset | Policy | Runs | ROI delta | BG delta | Q delta | Entropy delta | Success |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    mid_rows = [r for r in group_rows if r["roi_group"] == "mid_roi"]
    for row in sorted(mid_rows, key=lambda r: (r["preset"], r["policy"])):
        lines.append(
            f"| {row['preset']} | {row['policy']} | {row['runs']} | "
            f"{row['roi_delta_psnr_mean']:.4f} | {row['background_delta_psnr_mean']:.4f} | "
            f"{row['q_mean_delta_mean']:.4f} | {row['entropy_delta_mean']:.4f} | "
            f"{row['success_runs']}/{row['valid_runs']} |"
        )

    viable = [
        r
        for r in mid_rows
        if r["valid_runs"] > 0
        and r["success_runs"] > 0
        and r["roi_delta_psnr_mean"] >= -0.05
        and r["background_delta_psnr_mean"] <= -0.3
        and r["q_mean_delta_mean"] < 0.0
    ]
    if viable:
        robust_pool = [r for r in viable if r["valid_runs"] >= 3]
        pool = robust_pool if robust_pool else viable
        best = sorted(
            pool,
            key=lambda r: (
                r["success_runs"],
                r["valid_runs"],
                -r["entropy_delta_mean"],
                -r["q_mean_delta_mean"],
            ),
            reverse=True,
        )[0]
        lines.extend(
            [
                "",
                "## Recomendacion provisional",
                "",
                f"Preset/politica con mejor evidencia en `mid_roi`: `{best['preset']}` + `{best['policy']}`.",
                "Esta recomendacion es provisional hasta ejecutar el modo completo sobre los 120 crops si el checkpoint actual es smoke.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Recomendacion provisional",
                "",
                "Ningun preset/politica cumple todos los criterios en `mid_roi` dentro de este lote. "
                "Se debe ampliar la muestra o revisar umbrales antes de fijar una politica por defecto.",
            ]
        )

    lines.extend(
        [
            "",
            "## Missing Bands",
            "",
            "| Preset | Expected missing | Runs | Missing blocks mean |",
            "|---|---|---:|---:|",
        ]
    )
    for row in missing_rows:
        lines.append(
            f"| {row['preset']} | {row['expected_missing']} | {row['runs']} | {row['missing_bands_mean']:.2f} |"
        )

    lines.extend(
        [
            "",
            "La reduccion real de ancho de banda sigue pendiente del codificador entropico. "
            "Las metricas de ceros, entropia y `zlib` son proxies estadisticos sobre latentes `int32`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalua presets semanticos focus sobre dataset Sentinel-2A de 8 bandas.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/Sentinel2A_crop_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/checkpoints/20260520_semantic_presets_dataset_smoke"))
    parser.add_argument("--mode", choices=["preflight", "smoke", "full"], default="smoke")
    parser.add_argument("--resume", action="store_true", help="Reutiliza cases/*/case_result.json con case_status=done.")
    parser.add_argument("--smoke-crops", type=int, default=12)
    parser.add_argument("--max-crops", type=int, default=0, help="0 usa todos los crops del modo seleccionado.")
    parser.add_argument("--keep-heavy", choices=["none", "representative", "all"], default="representative")
    parser.add_argument("--calibration", type=Path, default=Path("output/checkpoints/20260507_c_fixed_quality_qmap_wide/fq_calibration.tsv"))
    parser.add_argument("--fq-bin", type=Path, default=Path("./sorteny_fq_qmap"))
    parser.add_argument("--semantic-bin", type=Path, default=Path("./sorteny_semantic_qmap"))
    parser.add_argument("--compressor", type=Path, default=Path("./sorteny_compressor"))
    parser.add_argument("--decompressor", type=Path, default=Path("./sorteny_decompressor"))
    parser.add_argument("--encoder-weights", type=Path, default=Path("weights/encoder"))
    parser.add_argument("--decoder-weights", type=Path, default=Path("weights/decoder"))
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--lambda-value", type=float, default=0.1)
    parser.add_argument("--max-lambda", type=float, default=0.125)
    parser.add_argument("--q-mean", type=int, default=204)
    parser.add_argument("--adaptive-strength", type=float, default=8.0)
    parser.add_argument(
        "--include-manual-roi",
        action="store_true",
        help="Incluye un control de foco manual consumido por sorteny_semantic_qmap --preset manual.",
    )
    parser.add_argument(
        "--manual-roi-pattern",
        choices=["center", "top_left", "bottom_right"],
        default="center",
        help="Patron manual reproducible si no se especifica --manual-roi-rect.",
    )
    parser.add_argument(
        "--manual-roi-rect",
        type=int,
        nargs=4,
        default=None,
        metavar=("Y0", "X0", "Y1", "X1"),
        help="ROI manual half-open en coordenadas de bloque 32x32.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    outdir = resolve(root, args.output_dir)
    dataset_dir = resolve(root, args.dataset_dir)
    args.calibration = resolve(root, args.calibration)
    args.encoder_weights = resolve(root, args.encoder_weights)
    args.decoder_weights = resolve(root, args.decoder_weights)

    qmap_dir = outdir / "qmaps"
    tsv_dir = outdir / "semantic_tsv"
    bitstream_dir = outdir / "bitstreams"
    recon_dir = outdir / "reconstructions"
    logs_dir = outdir / "logs"
    tmp_dir = outdir / "tmp_runtime"
    maps_dir = outdir / "maps"
    cases_dir = outdir / "cases"
    progress_path = outdir / "progress.csv"
    for d in [qmap_dir, tsv_dir, bitstream_dir, recon_dir, logs_dir, tmp_dir, maps_dir, cases_dir]:
        d.mkdir(parents=True, exist_ok=True)

    expected_bytes = args.bands * args.height * args.width * 2
    raw_paths = sorted(dataset_dir.glob("*.raw"))
    crops_all: list[Crop] = []
    manifest_rows: list[dict[str, Any]] = []
    for raw in raw_paths:
        size = raw.stat().st_size
        ok = size == expected_bytes
        manifest_rows.append(
            {
                "file": str(raw),
                "crop": raw.stem,
                "size_bytes": size,
                "expected_bytes": expected_bytes,
                "status": "ok" if ok else "bad_size",
            }
        )
        if ok:
            crops_all.append(Crop(raw, raw.stem, size))
    crops = crops_all[: args.max_crops] if args.max_crops > 0 else list(crops_all)
    if not crops:
        raise FileNotFoundError(f"No hay crops validos en {dataset_dir}")

    band_map = outdir / "sentinel2_8_band_map.tsv"
    make_band_map(band_map)

    if args.mode == "preflight":
        return run_preflight(
            args,
            root,
            outdir,
            dataset_dir,
            crops_all,
            manifest_rows,
            qmap_dir,
            tsv_dir,
            logs_dir,
            band_map,
        )

    manual_specs = manual_roi_specs(args)
    manual_roi_artifacts = write_manual_roi_artifacts(args, outdir)
    manual_qmaps: dict[tuple[str, str], Path] = {}
    manual_tsvs: dict[tuple[str, str], Path] = {}
    if manual_roi_artifacts is not None:
        roi_map = Path(manual_roi_artifacts["roi_map"])
        for manual_spec in manual_specs:
            preset = manual_spec["preset"]
            for policy_spec in SEMANTIC_POLICIES:
                policy = policy_spec["policy"]
                qmap = qmap_dir / policy / preset / f"{preset}.bin"
                tsv = tsv_dir / policy / preset / f"{preset}.tsv"
                qmap.parent.mkdir(parents=True, exist_ok=True)
                tsv.parent.mkdir(parents=True, exist_ok=True)
                if not qmap.exists() or qmap.stat().st_size != QMAP_BYTES:
                    cmd = manual_qmap_cmd(args, roi_map, qmap, tsv, policy_spec["args"])
                    run_cmd(cmd, root, logs_dir / "manual_roi" / preset / f"qmap_{policy}.log")
                manual_qmaps[(preset, policy)] = qmap
                manual_tsvs[(preset, policy)] = tsv

    control_qmaps: dict[str, Path] = {}
    control_tsv: dict[str, Path] = {}
    for mode in CONTROL_POLICIES:
        control_qmaps[mode] = qmap_dir / "controls" / f"{mode}.bin"
        control_tsv[mode] = tsv_dir / "controls" / f"{mode}.tsv"
        control_qmaps[mode].parent.mkdir(parents=True, exist_ok=True)
        control_tsv[mode].parent.mkdir(parents=True, exist_ok=True)
        if not control_qmaps[mode].exists() or control_qmaps[mode].stat().st_size != QMAP_BYTES:
            cmd = fq_qmap_cmd(args, control_qmaps[mode], control_tsv[mode], mode)
            run_cmd(cmd, root, logs_dir / "controls" / f"qmap_{mode}.log")
    adaptive_qmap = load_qmap(control_qmaps["adaptive_s8"])
    q204_qmap = load_qmap(control_qmaps["q204"])

    prescan_rows: list[dict[str, Any]] = []
    for crop in crops:
        for preset_spec in USEFUL_PRESETS:
            preset = preset_spec["preset"]
            qmap = qmap_dir / "prescan_bgpen24" / preset / f"{crop.stem}.bin"
            tsv = tsv_dir / "prescan_bgpen24" / preset / f"{crop.stem}.tsv"
            qmap.parent.mkdir(parents=True, exist_ok=True)
            tsv.parent.mkdir(parents=True, exist_ok=True)
            if not qmap.exists() or qmap.stat().st_size != QMAP_BYTES:
                cmd = semantic_qmap_cmd(
                    args,
                    crop.path,
                    preset,
                    qmap,
                    tsv,
                    SEMANTIC_POLICIES[0]["args"],
                    band_map=band_map,
                    threshold=preset_spec["threshold"],
                )
                run_cmd(cmd, root, logs_dir / "prescan_bgpen24" / preset / f"{crop.stem}.log")
            sem = read_semantic_tsv(tsv)
            roi_blocks = int(np.sum(sem["roi"]))
            prescan_rows.append(
                {
                    "crop": crop.stem,
                    "file": str(crop.path),
                    "preset": preset,
                    "index": preset_spec["index"],
                    "roi_blocks": roi_blocks,
                    "roi_pct": float(100.0 * roi_blocks / QMAP_BYTES),
                    "roi_group": roi_group(roi_blocks),
                    "missing_bands": sem["missing_bands"],
                    "qmap": str(qmap),
                    "summary_tsv": str(tsv),
                }
            )

    selected_crops = crops if args.mode == "full" else choose_smoke_crops(prescan_rows, crops, args.smoke_crops)
    if args.max_crops > 0:
        selected_crops = selected_crops[: args.max_crops]
    selected_stems = {c.stem for c in selected_crops}
    selected_prescan_rows = [r for r in prescan_rows if r["crop"] in selected_stems]

    representative: dict[str, str] = {}
    for preset_spec in USEFUL_PRESETS:
        preset = preset_spec["preset"]
        rows = [r for r in selected_prescan_rows if r["preset"] == preset]
        mid = [r for r in rows if r["roi_group"] == "mid_roi"]
        if mid:
            representative[preset] = min(mid, key=lambda r: abs(r["roi_pct"] - 40.0))["crop"]
        elif rows:
            representative[preset] = max(rows, key=lambda r: min(r["roi_pct"], 100.0 - r["roi_pct"]))["crop"]
    if selected_crops:
        for manual_spec in manual_specs:
            representative[manual_spec["preset"]] = selected_crops[0].stem

    write_run_manifest(args, root, outdir, dataset_dir, crops_all, selected_crops)

    result_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    command_rows: list[dict[str, Any]] = []

    for index, crop in enumerate(selected_crops, start=1):
        print(f"[{index}/{len(selected_crops)}] {crop.stem}", flush=True)
        original = load_raw_u16(crop.path, args.bands, args.height, args.width)
        crop_metrics: dict[str, dict[str, Any]] = {}
        crop_timings: dict[str, dict[str, float]] = {}

        for control in CONTROL_POLICIES:
            keep = args.keep_heavy == "all"
            bitstream = (bitstream_dir if keep else tmp_dir) / control / f"{crop.stem}.bin"
            recon = (recon_dir if keep else tmp_dir) / control / f"{crop.stem}.raw"
            metrics, timings, loaded = run_or_load_control_case(
                args,
                root,
                outdir,
                progress_path,
                crop,
                control,
                original,
                control_qmaps[control],
                bitstream,
                recon,
                logs_dir / crop.stem / control,
                keep,
            )
            crop_timings[control] = timings
            crop_metrics[control] = metrics
            if not keep and not loaded:
                bitstream.unlink(missing_ok=True)
                recon.unlink(missing_ok=True)

        for preset_spec in USEFUL_PRESETS:
            preset = preset_spec["preset"]
            prescan_tsv = tsv_dir / "prescan_bgpen24" / preset / f"{crop.stem}.tsv"
            sem_for_mask = read_semantic_tsv(prescan_tsv)
            for control in CONTROL_POLICIES:
                metrics = crop_metrics[control]
                row = make_result_row(
                    crop,
                    preset_spec,
                    control,
                    "control",
                    sem_for_mask,
                    metrics,
                    crop_metrics["adaptive_s8"],
                    q204_qmap if control == "q204" else adaptive_qmap,
                    adaptive_qmap,
                    crop_timings[control],
                    {
                        "qmap_path": str(control_qmaps[control]),
                        "summary_tsv": str(control_tsv[control]),
                        "bitstream_path": "",
                        "reconstruction_path": "",
                    },
                )
                result_rows.append(row)

            for policy_spec in SEMANTIC_POLICIES:
                policy = policy_spec["policy"]
                qmap = qmap_dir / policy / preset / f"{crop.stem}.bin"
                tsv = tsv_dir / policy / preset / f"{crop.stem}.tsv"
                qmap.parent.mkdir(parents=True, exist_ok=True)
                tsv.parent.mkdir(parents=True, exist_ok=True)
                if policy == "focus_bgpen24":
                    src_qmap = qmap_dir / "prescan_bgpen24" / preset / f"{crop.stem}.bin"
                    src_tsv = tsv_dir / "prescan_bgpen24" / preset / f"{crop.stem}.tsv"
                    qmap.parent.mkdir(parents=True, exist_ok=True)
                    tsv.parent.mkdir(parents=True, exist_ok=True)
                    if not qmap.exists():
                        shutil.copy2(src_qmap, qmap)
                    if not tsv.exists():
                        shutil.copy2(src_tsv, tsv)
                elif not qmap.exists() or qmap.stat().st_size != QMAP_BYTES:
                    cmd = semantic_qmap_cmd(
                        args,
                        crop.path,
                        preset,
                        qmap,
                        tsv,
                        policy_spec["args"],
                        band_map=band_map,
                        threshold=preset_spec["threshold"],
                    )
                    elapsed = run_cmd(cmd, root, logs_dir / crop.stem / policy / preset / "qmap.log")
                    command_rows.append({"crop": crop.stem, "preset": preset, "policy": policy, "command": " ".join(cmd), "elapsed_s": elapsed})
                sem = read_semantic_tsv(tsv)
                semantic_qmap = load_qmap(qmap)
                if qmap.stat().st_size != QMAP_BYTES:
                    raise ValueError(f"{qmap}: Q-map no tiene 1024 bytes")

                keep = args.keep_heavy == "all" or (
                    args.keep_heavy == "representative" and representative.get(preset) == crop.stem and policy == "focus_bgq128"
                )
                bitstream = (bitstream_dir if keep else tmp_dir) / policy / preset / f"{crop.stem}.bin"
                recon = (recon_dir if keep else tmp_dir) / policy / preset / f"{crop.stem}.raw"
                artifact_paths = {
                    "qmap_path": str(qmap),
                    "summary_tsv": str(tsv),
                    "bitstream_path": str(bitstream) if keep else "",
                    "reconstruction_path": str(recon) if keep else "",
                }
                row, latent_row, metrics, loaded = run_or_load_semantic_case(
                    args,
                    root,
                    outdir,
                    progress_path,
                    crop,
                    preset_spec,
                    policy,
                    sem,
                    semantic_qmap,
                    adaptive_qmap,
                    crop_metrics["adaptive_s8"],
                    original,
                    qmap,
                    bitstream,
                    recon,
                    logs_dir / crop.stem / policy / preset,
                    f"{crop.stem}_{preset}_{policy}",
                    artifact_paths,
                    keep,
                )
                result_rows.append(row)
                latent_rows.append(latent_row)
                if (
                    not loaded
                    and keep
                    and metrics is not None
                    and representative.get(preset) == crop.stem
                    and policy == "focus_bgq128"
                ):
                    write_representative_maps(
                        maps_dir,
                        preset,
                        crop.stem,
                        sem["roi"],
                        adaptive_qmap,
                        semantic_qmap,
                        crop_metrics["adaptive_s8"]["blocks"],
                        metrics["blocks"],
                        metrics["latents"],
                    )
                if not keep and not loaded:
                    bitstream.unlink(missing_ok=True)
                    recon.unlink(missing_ok=True)

        for manual_spec in manual_specs:
            preset = manual_spec["preset"]
            mask_tsv = manual_tsvs[(preset, SEMANTIC_POLICIES[0]["policy"])]
            sem_for_mask = read_semantic_tsv(mask_tsv)
            for control in CONTROL_POLICIES:
                metrics = crop_metrics[control]
                row = make_result_row(
                    crop,
                    manual_spec,
                    control,
                    "control",
                    sem_for_mask,
                    metrics,
                    crop_metrics["adaptive_s8"],
                    q204_qmap if control == "q204" else adaptive_qmap,
                    adaptive_qmap,
                    crop_timings[control],
                    {
                        "qmap_path": str(control_qmaps[control]),
                        "summary_tsv": str(control_tsv[control]),
                        "bitstream_path": "",
                        "reconstruction_path": "",
                    },
                )
                result_rows.append(row)

            for policy_spec in SEMANTIC_POLICIES:
                policy = policy_spec["policy"]
                qmap = manual_qmaps[(preset, policy)]
                tsv = manual_tsvs[(preset, policy)]
                sem = read_semantic_tsv(tsv)
                manual_qmap = load_qmap(qmap)
                if qmap.stat().st_size != QMAP_BYTES:
                    raise ValueError(f"{qmap}: Q-map no tiene 1024 bytes")

                keep = args.keep_heavy == "all" or (
                    args.keep_heavy == "representative" and representative.get(preset) == crop.stem and policy == "focus_bgq128"
                )
                bitstream = (bitstream_dir if keep else tmp_dir) / policy / preset / f"{crop.stem}.bin"
                recon = (recon_dir if keep else tmp_dir) / policy / preset / f"{crop.stem}.raw"
                artifact_paths = {
                    "qmap_path": str(qmap),
                    "summary_tsv": str(tsv),
                    "bitstream_path": str(bitstream) if keep else "",
                    "reconstruction_path": str(recon) if keep else "",
                }
                row, latent_row, metrics, loaded = run_or_load_semantic_case(
                    args,
                    root,
                    outdir,
                    progress_path,
                    crop,
                    manual_spec,
                    policy,
                    sem,
                    manual_qmap,
                    adaptive_qmap,
                    crop_metrics["adaptive_s8"],
                    original,
                    qmap,
                    bitstream,
                    recon,
                    logs_dir / crop.stem / policy / preset,
                    f"{crop.stem}_{preset}_{policy}",
                    artifact_paths,
                    keep,
                    policy_kind="manual_focus",
                )
                result_rows.append(row)
                latent_rows.append(latent_row)
                if (
                    not loaded
                    and keep
                    and metrics is not None
                    and representative.get(preset) == crop.stem
                    and policy == "focus_bgq128"
                ):
                    write_representative_maps(
                        maps_dir,
                        preset,
                        crop.stem,
                        sem["roi"],
                        adaptive_qmap,
                        manual_qmap,
                        crop_metrics["adaptive_s8"]["blocks"],
                        metrics["blocks"],
                        metrics["latents"],
                    )
                if not keep and not loaded:
                    bitstream.unlink(missing_ok=True)
                    recon.unlink(missing_ok=True)

    missing_audit_rows: list[dict[str, Any]] = []
    missing_detail_rows: list[dict[str, Any]] = []
    for preset_spec in MISSING_BAND_PRESETS:
        preset = preset_spec["preset"]
        rows_for_preset = []
        for crop in selected_crops:
            case_path = missing_case_path(outdir, crop, preset)
            if args.resume:
                cached = load_done_case(case_path)
                if cached is not None:
                    detail = cached["detail"]
                    ts = now_iso()
                    write_progress(
                        progress_path,
                        crop=crop.stem,
                        preset=preset,
                        policy="missing_bands_audit",
                        status="done",
                        started_at=ts,
                        finished_at=ts,
                        elapsed_s=0.0,
                        log_path="resume-cache",
                        case_result_path=case_path,
                    )
                    missing_detail_rows.append(detail)
                    rows_for_preset.append(detail)
                    continue
            qmap = qmap_dir / "missing_bands_audit" / preset / f"{crop.stem}.bin"
            tsv = tsv_dir / "missing_bands_audit" / preset / f"{crop.stem}.tsv"
            qmap.parent.mkdir(parents=True, exist_ok=True)
            tsv.parent.mkdir(parents=True, exist_ok=True)
            started = now_iso()
            t0 = time.perf_counter()
            write_progress(
                progress_path,
                crop=crop.stem,
                preset=preset,
                policy="missing_bands_audit",
                status="running",
                started_at=started,
                log_path=logs_dir / crop.stem / "missing_bands" / preset,
                case_result_path=case_path,
            )
            if not qmap.exists() or qmap.stat().st_size != QMAP_BYTES:
                cmd = semantic_qmap_cmd(
                    args,
                    crop.path,
                    preset,
                    qmap,
                    tsv,
                    ["--semantic-policy", "boost-only", "--foreground-boost", "8"],
                    band_map=band_map,
                )
                run_cmd(cmd, root, logs_dir / crop.stem / "missing_bands" / preset / "qmap.log")
            sem = read_semantic_tsv(tsv)
            detail = {
                "crop": crop.stem,
                "file": str(crop.path),
                "preset": preset,
                "index": preset_spec["index"],
                "expected_missing": preset_spec["missing"],
                "missing_bands": sem["missing_bands"],
                "semantic_matches": sem["semantic_matches"],
                "status": "missing_bands" if sem["missing_bands"] else "unexpected_available",
                "qmap_bytes": qmap.stat().st_size,
            }
            write_case_result(
                case_path,
                {
                    "case_status": "done",
                    "case_kind": "missing_bands_audit",
                    "crop": crop.stem,
                    "preset": preset,
                    "policy": "missing_bands_audit",
                    "detail": detail,
                    "started_at": started,
                    "finished_at": now_iso(),
                },
            )
            write_progress(
                progress_path,
                crop=crop.stem,
                preset=preset,
                policy="missing_bands_audit",
                status="done",
                started_at=started,
                finished_at=now_iso(),
                elapsed_s=time.perf_counter() - t0,
                log_path=logs_dir / crop.stem / "missing_bands" / preset,
                case_result_path=case_path,
            )
            missing_detail_rows.append(detail)
            rows_for_preset.append(detail)
        missing_audit_rows.append(
            {
                "preset": preset,
                "index": preset_spec["index"],
                "bands": preset_spec["bands"],
                "expected_missing": preset_spec["missing"],
                "runs": len(rows_for_preset),
                "missing_bands_mean": mean([r["missing_bands"] for r in rows_for_preset]),
                "missing_bands_min": int(np.min([r["missing_bands"] for r in rows_for_preset])),
                "missing_bands_max": int(np.max([r["missing_bands"] for r in rows_for_preset])),
                "all_missing_1024": all(r["missing_bands"] == QMAP_BYTES for r in rows_for_preset),
            }
        )

    group_rows = summarize_groups(result_rows)

    result_fields = [
        "crop",
        "file",
        "preset",
        "index",
        "bands",
        "threshold",
        "policy",
        "policy_kind",
        "status",
        "roi_group",
        "roi_blocks",
        "background_blocks",
        "roi_pct",
        "missing_bands",
        "index_mean",
        "index_min",
        "index_max",
        "global_mse",
        "global_psnr_db",
        "roi_mse",
        "roi_psnr_db",
        "background_mse",
        "background_psnr_db",
        "adaptive_global_psnr_db",
        "adaptive_roi_psnr_db",
        "adaptive_background_psnr_db",
        "delta_global_psnr_vs_adaptive",
        "delta_roi_psnr_vs_adaptive",
        "delta_background_psnr_vs_adaptive",
        "q_min",
        "q_max",
        "q_mean",
        "q_roi_mean",
        "q_background_mean",
        "q_unique",
        "q_mean_delta_vs_adaptive",
        "latent_zero_pct",
        "latent_entropy_bits_per_symbol",
        "latent_ideal_bps_per_input_sample",
        "latent_zlib_bps_per_input_sample",
        "latent_mean_abs",
        "latent_unique_values",
        "delta_zero_pct_vs_adaptive",
        "delta_entropy_vs_adaptive",
        "delta_zlib_bps_vs_adaptive",
        "compress_s",
        "decompress_s",
        "qmap_path",
        "summary_tsv",
        "bitstream_path",
        "reconstruction_path",
    ]
    group_fields = [
        "preset",
        "policy",
        "roi_group",
        "runs",
        "valid_runs",
        "roi_pct_mean",
        "global_psnr_mean",
        "roi_delta_psnr_mean",
        "roi_delta_psnr_median",
        "roi_delta_psnr_worst",
        "background_delta_psnr_mean",
        "background_delta_psnr_median",
        "background_delta_psnr_worst_degradation",
        "q_mean_delta_mean",
        "entropy_delta_mean",
        "zero_delta_mean",
        "roi_kept_or_improved_pct",
        "background_degraded_ge_0_3db_pct",
        "q_mean_lower_pct",
        "latent_simpler_pct",
        "success_runs",
    ]

    write_csv(outdir / "dataset_manifest.csv", manifest_rows, ["file", "crop", "size_bytes", "expected_bytes", "status"])
    write_csv(outdir / "smoke_selection.csv", selected_prescan_rows, ["crop", "file", "preset", "index", "roi_blocks", "roi_pct", "roi_group", "missing_bands", "qmap", "summary_tsv"])
    write_csv(outdir / "semantic_preset_dataset_results.csv", result_rows, result_fields)
    write_json(outdir / "semantic_preset_dataset_results.json", {"rows": result_rows})
    write_csv(outdir / "semantic_preset_group_summary.csv", group_rows, group_fields)
    write_json(outdir / "semantic_preset_group_summary.json", {"rows": group_rows})
    write_csv(
        outdir / "latent_proxy_by_preset.csv",
        latent_rows,
        [
            "crop",
            "preset",
            "policy",
            "status",
            "roi_group",
            "zero_pct",
            "entropy_bits_per_symbol",
            "ideal_bps_per_input_sample",
            "zlib_bps_per_input_sample",
            "mean_abs",
            "unique_values",
            "delta_zero_pct_vs_adaptive",
            "delta_entropy_vs_adaptive",
        ],
    )
    write_csv(
        outdir / "missing_bands_summary.csv",
        missing_audit_rows,
        ["preset", "index", "bands", "expected_missing", "runs", "missing_bands_mean", "missing_bands_min", "missing_bands_max", "all_missing_1024"],
    )
    write_csv(
        outdir / "missing_bands_detail.csv",
        missing_detail_rows,
        ["crop", "file", "preset", "index", "expected_missing", "missing_bands", "semantic_matches", "status", "qmap_bytes"],
    )
    write_json(
        outdir / "checkpoint_summary.json",
        {
            "mode": args.mode,
            "resume": bool(args.resume),
            "selected_crops": [c.stem for c in selected_crops],
            "valid_raws": len(crops_all),
            "processed_crops": len(selected_crops),
            "result_rows": len(result_rows),
            "group_summary_rows": group_rows,
            "missing_bands_summary": missing_audit_rows,
            "representative_crops": representative,
            "keep_heavy": args.keep_heavy,
            "manual_roi": manual_roi_artifacts,
        },
    )
    write_recommendations(outdir / "semantic_preset_recommendations.md", group_rows, missing_audit_rows, args.mode)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[OK] Validacion semantica dataset completada: {outdir}")
    print(f"Modo: {args.mode}; crops procesados: {len(selected_crops)}; filas: {len(result_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
