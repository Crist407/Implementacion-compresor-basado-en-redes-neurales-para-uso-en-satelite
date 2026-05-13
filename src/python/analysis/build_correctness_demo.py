#!/usr/bin/env python3
"""
Construye una demo reproducible de funcionamiento correcto.

Este script es auxiliar: orquesta binarios C, calcula metricas y genera
evidencia local. La decision de Q-map sigue ocurriendo en C.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import struct
import subprocess
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MAX_U16 = 65535.0


@dataclass(frozen=True)
class Policy:
    key: str
    label: str
    qmap: Path
    q_summary: Path
    latent: Path
    recon: Path


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
        raise ValueError(f"{path}: raw size {data.size}, expected {expected}")
    return data.reshape(bands, height, width)


def load_qmap(path: Path, q_height: int, q_width: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    expected = q_height * q_width
    if data.size != expected:
        raise ValueError(f"{path}: qmap bytes {data.size}, expected {expected}")
    return data.reshape(q_height, q_width)


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
    bh = height // block_size
    bw = width // block_size
    mse = np.zeros((bh, bw), dtype=np.float64)
    mae = np.zeros((bh, bw), dtype=np.float64)
    max_abs = np.zeros((bh, bw), dtype=np.float64)
    exact_pct = np.zeros((bh, bw), dtype=np.float64)
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    abs_diff = np.abs(diff)
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


def read_semantic_tsv(path: Path, q_height: int, q_width: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    ndvi = np.full((q_height, q_width), np.nan, dtype=np.float64)
    roi = np.zeros((q_height, q_width), dtype=bool)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            by = int(row["block_y"])
            bx = int(row["block_x"])
            value = float(row["index_mean"])
            match = int(row["semantic_match"]) != 0
            expected = value >= threshold
            if match != expected:
                raise ValueError(
                    f"{path}: ROI mismatch block ({by},{bx}), index={value}, "
                    f"semantic_match={match}, threshold={threshold}"
                )
            ndvi[by, bx] = value
            roi[by, bx] = match
    if np.isnan(ndvi).any():
        raise ValueError(f"{path}: missing NDVI values")
    return ndvi, roi


def group_summary(metrics: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    if not np.any(mask):
        return {"blocks": 0, "mse": math.nan, "psnr_db": math.nan, "mae": math.nan, "max_abs": math.nan, "exact_pct": math.nan}
    mse = float(np.mean(metrics["mse"][mask]))
    return {
        "blocks": int(np.sum(mask)),
        "mse": mse,
        "psnr_db": psnr_from_mse(mse),
        "mae": float(np.mean(metrics["mae"][mask])),
        "max_abs": float(np.max(metrics["max_abs"][mask])),
        "exact_pct": float(np.mean(metrics["exact_pct"][mask])),
    }


def q_summary(qmap: np.ndarray, roi: np.ndarray) -> dict[str, Any]:
    background = ~roi
    return {
        "q_min": int(np.min(qmap)),
        "q_max": int(np.max(qmap)),
        "q_mean": float(np.mean(qmap)),
        "q_unique": int(np.unique(qmap).size),
        "q_roi_mean": float(np.mean(qmap[roi])) if np.any(roi) else math.nan,
        "q_background_mean": float(np.mean(qmap[background])) if np.any(background) else math.nan,
        "qmap_bytes": int(qmap.size),
    }


def entropy_of_values(values: np.ndarray) -> float:
    flat = values.reshape(-1)
    _, counts = np.unique(flat, return_counts=True)
    probs = counts.astype(np.float64) / float(flat.size)
    return float(-np.sum(probs * np.log2(probs)))


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
        "ideal_bytes": ideal_bits / 8.0,
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
    x = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(x * 255.0).astype(np.uint8)


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
    path32 = base_dir / "32x32" / name
    path512 = base_dir / "512x512" / name
    if rgb:
        write_ppm(path32, image)
        write_ppm(path512, upsample(image, factor))
    else:
        write_pgm(path32, image)
        write_pgm(path512, upsample(image, factor))


def write_maps(maps_dir: Path, ndvi: np.ndarray, roi: np.ndarray, policies: dict[str, dict[str, Any]]) -> None:
    write_map_pair(maps_dir, "ndvi_minus1_plus1.pgm", normalize_to_u8(ndvi, -1.0, 1.0))
    write_map_pair(maps_dir, "roi_vegetation_ndvi_ge_040.pgm", roi.astype(np.uint8) * 255)
    baseline_q = policies["q204"]["qmap"].astype(np.float64)
    adaptive = policies["adaptive_s8"]
    focus = policies["vegetation_focus_bgq128"]
    for key, data in policies.items():
        write_map_pair(maps_dir, f"qmap_{key}.pgm", normalize_to_u8(data["qmap"], 128.0, 255.0))
        write_map_pair(maps_dir, f"mse_{key}.pgm", normalize_to_u8(data["block_metrics"]["mse"]))
        write_map_pair(maps_dir, f"psnr_{key}.pgm", normalize_to_u8(data["block_metrics"]["psnr_db"]))
        write_map_pair(maps_dir, f"latent_zero_pct_{key}.pgm", normalize_to_u8(data["latent_zero_map"], 0.0, 100.0))
        write_map_pair(maps_dir, f"latent_entropy_{key}.pgm", normalize_to_u8(data["latent_entropy_map"]))
        write_map_pair(maps_dir, f"delta_q_{key}_minus_q204.ppm", diverging_rgb(data["qmap"].astype(np.float64) - baseline_q), rgb=True)
    write_map_pair(
        maps_dir,
        "delta_mse_focus_minus_adaptive.ppm",
        diverging_rgb(focus["block_metrics"]["mse"] - adaptive["block_metrics"]["mse"]),
        rgb=True,
    )
    write_map_pair(
        maps_dir,
        "delta_psnr_focus_minus_adaptive.ppm",
        diverging_rgb(focus["block_metrics"]["psnr_db"] - adaptive["block_metrics"]["psnr_db"]),
        rgb=True,
    )
    write_map_pair(
        maps_dir,
        "delta_zero_focus_minus_adaptive.ppm",
        diverging_rgb(focus["latent_zero_map"] - adaptive["latent_zero_map"]),
        rgb=True,
    )


def write_block_evidence(path: Path, ndvi: np.ndarray, roi: np.ndarray, policies: dict[str, dict[str, Any]]) -> None:
    keys = list(policies.keys())
    fields = ["block_y", "block_x", "ndvi", "roi"]
    for key in keys:
        fields.extend(
            [
                f"q_{key}",
                f"mse_{key}",
                f"psnr_db_{key}",
                f"mae_{key}",
                f"latent_zero_pct_{key}",
                f"latent_entropy_{key}",
            ]
        )
    fields.extend(
        [
            "delta_q_focus_vs_adaptive",
            "delta_mse_focus_vs_adaptive",
            "delta_psnr_focus_vs_adaptive",
            "delta_zero_pct_focus_vs_adaptive",
        ]
    )
    focus = policies["vegetation_focus_bgq128"]
    adaptive = policies["adaptive_s8"]
    rows: list[dict[str, Any]] = []
    for by in range(ndvi.shape[0]):
        for bx in range(ndvi.shape[1]):
            row: dict[str, Any] = {"block_y": by, "block_x": bx, "ndvi": float(ndvi[by, bx]), "roi": int(roi[by, bx])}
            for key in keys:
                data = policies[key]
                row[f"q_{key}"] = int(data["qmap"][by, bx])
                row[f"mse_{key}"] = float(data["block_metrics"]["mse"][by, bx])
                row[f"psnr_db_{key}"] = float(data["block_metrics"]["psnr_db"][by, bx])
                row[f"mae_{key}"] = float(data["block_metrics"]["mae"][by, bx])
                row[f"latent_zero_pct_{key}"] = float(data["latent_zero_map"][by, bx])
                row[f"latent_entropy_{key}"] = float(data["latent_entropy_map"][by, bx])
            row["delta_q_focus_vs_adaptive"] = int(focus["qmap"][by, bx]) - int(adaptive["qmap"][by, bx])
            row["delta_mse_focus_vs_adaptive"] = float(focus["block_metrics"]["mse"][by, bx] - adaptive["block_metrics"]["mse"][by, bx])
            row["delta_psnr_focus_vs_adaptive"] = float(focus["block_metrics"]["psnr_db"][by, bx] - adaptive["block_metrics"]["psnr_db"][by, bx])
            row["delta_zero_pct_focus_vs_adaptive"] = float(focus["latent_zero_map"][by, bx] - adaptive["latent_zero_map"][by, bx])
            rows.append(row)
    write_csv(path, rows, fields)


def write_report(path: Path, summary_rows: list[dict[str, Any]], validation: dict[str, Any]) -> None:
    lines = [
        "# Correct Functioning Demo",
        "",
        "Demo reproducible local del pipeline SORTENY C con Q-map constante, adaptativo y semantico focus.",
        "",
        "| Policy | Q mean | PSNR global | PSNR ROI | PSNR background | Zeros | Entropy | zlib bps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['policy']} | {row['q_mean']:.4f} | {row['global_psnr_db']:.4f} | "
            f"{row['roi_psnr_db']:.4f} | {row['background_psnr_db']:.4f} | "
            f"{row['latent_zero_pct']:.2f}% | {row['latent_entropy_bits_per_symbol']:.4f} | "
            f"{row['zlib_bps_per_input_sample']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            f"- Q204 PSNR reference ok: {validation['q204_psnr_reference_ok']}.",
            f"- Focus ROI >= adaptive ROI: {validation['focus_roi_kept_or_improved_vs_adaptive']}.",
            f"- Focus background degraded vs adaptive: {validation['focus_background_degraded_vs_adaptive']}.",
            f"- Focus Q mean lower than adaptive: {validation['focus_q_mean_lower_than_adaptive']}.",
            f"- Focus zeros higher than adaptive: {validation['focus_zeros_higher_than_adaptive']}.",
            f"- Focus entropy lower than adaptive: {validation['focus_entropy_lower_than_adaptive']}.",
            f"- All Q-maps are 1024 bytes: {validation['all_qmaps_1024_bytes']}.",
            "",
            "La reduccion real de ancho de banda queda pendiente del codificador entropico: "
            "el bitstream actual sigue escribiendo los latentes como `int32`. Esta demo demuestra "
            "que la distribucion estadistica de latentes es mas favorable.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye la demo reproducible de funcionamiento correcto.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("output/checkpoints/20260517_correct_functioning_demo"))
    parser.add_argument("--input", type=Path, default=Path("data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw"))
    parser.add_argument("--calibration", type=Path, default=Path("output/checkpoints/20260507_c_fixed_quality_qmap_wide/fq_calibration.tsv"))
    parser.add_argument("--encoder-weights", type=Path, default=Path("weights/encoder"))
    parser.add_argument("--decoder-weights", type=Path, default=Path("weights/decoder"))
    parser.add_argument("--fq-bin", type=Path, default=Path("./sorteny_fq_qmap"))
    parser.add_argument("--semantic-bin", type=Path, default=Path("./sorteny_semantic_qmap"))
    parser.add_argument("--compressor", type=Path, default=Path("./sorteny_compressor"))
    parser.add_argument("--decompressor", type=Path, default=Path("./sorteny_decompressor"))
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--lambda-value", type=float, default=0.1)
    parser.add_argument("--max-lambda", type=float, default=0.125)
    parser.add_argument("--threshold", type=float, default=0.40)
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    outdir = resolve(root, args.output_dir)
    qmap_dir = outdir / "qmaps"
    recon_dir = outdir / "reconstructions"
    latent_dir = outdir / "bitstreams"
    logs_dir = outdir / "logs"
    summaries_dir = outdir / "summaries"
    maps_dir = outdir / "maps"
    for d in [qmap_dir, recon_dir, latent_dir, logs_dir, summaries_dir, maps_dir]:
        d.mkdir(parents=True, exist_ok=True)

    input_path = resolve(root, args.input)
    calibration = resolve(root, args.calibration)
    encoder_weights = resolve(root, args.encoder_weights)
    decoder_weights = resolve(root, args.decoder_weights)
    for path in [input_path, calibration, encoder_weights, decoder_weights]:
        if not path.exists():
            raise FileNotFoundError(path)
    q_height = args.height // args.block_size
    q_width = args.width // args.block_size
    input_samples = args.bands * args.height * args.width

    policies = [
        Policy("q204", "Q constante 204", qmap_dir / "q204.bin", summaries_dir / "q204.tsv", latent_dir / "q204.bin", recon_dir / "q204.raw"),
        Policy("adaptive_s8", "Adaptativo dificultad s=8", qmap_dir / "adaptive_s8.bin", summaries_dir / "adaptive_s8.tsv", latent_dir / "adaptive_s8.bin", recon_dir / "adaptive_s8.raw"),
        Policy(
            "vegetation_focus_bgq128",
            "Vegetation focus fg=16 bgQ=128",
            qmap_dir / "vegetation_focus_bgq128.bin",
            summaries_dir / "vegetation_focus_bgq128.tsv",
            latent_dir / "vegetation_focus_bgq128.bin",
            recon_dir / "vegetation_focus_bgq128.raw",
        ),
    ]

    commands: list[dict[str, Any]] = []
    cmd = [
        executable(args.fq_bin),
        "--calibration",
        str(calibration),
        "--target-from-q",
        "204",
        "--output-qmap",
        str(policies[0].qmap),
        "--summary-tsv",
        str(policies[0].q_summary),
    ]
    commands.append({"name": "qmap_q204", "cmd": cmd, "elapsed_s": run_cmd(cmd, root, logs_dir / "qmap_q204.log")})

    cmd = [
        executable(args.fq_bin),
        "--calibration",
        str(calibration),
        "--adaptive-difficulty",
        "--q-mean",
        "204",
        "--adaptive-strength",
        "8",
        "--output-qmap",
        str(policies[1].qmap),
        "--summary-tsv",
        str(policies[1].q_summary),
    ]
    commands.append({"name": "qmap_adaptive_s8", "cmd": cmd, "elapsed_s": run_cmd(cmd, root, logs_dir / "qmap_adaptive_s8.log")})

    cmd = [
        executable(args.semantic_bin),
        "--input",
        str(input_path),
        "--calibration",
        str(calibration),
        "--preset",
        "vegetation",
        "--semantic-policy",
        "focus",
        "--foreground-boost",
        "16",
        "--background-q",
        "128",
        "--threshold",
        f"{args.threshold:.6g}",
        "--output-qmap",
        str(policies[2].qmap),
        "--summary-tsv",
        str(policies[2].q_summary),
        "--bands",
        str(args.bands),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--band-layout",
        "sentinel2-8",
        "--q-mean",
        "204",
        "--adaptive-strength",
        "8",
    ]
    commands.append({"name": "qmap_vegetation_focus_bgq128", "cmd": cmd, "elapsed_s": run_cmd(cmd, root, logs_dir / "qmap_vegetation_focus_bgq128.log")})

    for policy in policies:
        cmd = [
            executable(args.compressor),
            str(input_path),
            f"{args.lambda_value:.8g}",
            str(policy.latent),
            str(encoder_weights),
            f"{args.max_lambda:.8g}",
            str(policy.qmap),
        ]
        commands.append({"name": f"compress_{policy.key}", "cmd": cmd, "elapsed_s": run_cmd(cmd, root, logs_dir / f"compress_{policy.key}.log")})
        cmd = [
            executable(args.decompressor),
            str(policy.latent),
            str(policy.recon),
            str(decoder_weights),
            f"{args.max_lambda:.8g}",
        ]
        commands.append({"name": f"decompress_{policy.key}", "cmd": cmd, "elapsed_s": run_cmd(cmd, root, logs_dir / f"decompress_{policy.key}.log")})

    commands_path = outdir / "commands.json"
    write_json(commands_path, commands)
    (outdir / "commands.txt").write_text("\n".join(" ".join(c["cmd"]) for c in commands) + "\n", encoding="utf-8")

    original = load_raw_u16(input_path, args.bands, args.height, args.width)
    ndvi, roi = read_semantic_tsv(policies[2].q_summary, q_height, q_width, args.threshold)
    background = ~roi
    policy_data: dict[str, dict[str, Any]] = {}
    for policy in policies:
        qmap_external = load_qmap(policy.qmap, q_height, q_width)
        header, qmap_from_bitstream, latents = read_bitstream(policy.latent)
        if not np.array_equal(qmap_external, qmap_from_bitstream):
            raise ValueError(f"{policy.key}: qmap file differs from bitstream qmap")
        if (header["bands"], header["height"], header["width"]) != (args.bands, args.height, args.width):
            raise ValueError(f"{policy.key}: bitstream header mismatch {header}")
        recon = load_raw_u16(policy.recon, args.bands, args.height, args.width)
        bm = block_metrics(original, recon, args.block_size)
        zmap = latent_zero_map(latents)
        emap = latent_entropy_map(latents)
        policy_data[policy.key] = {
            "label": policy.label,
            "qmap": qmap_external,
            "latent": latents,
            "block_metrics": bm,
            "latent_zero_map": zmap,
            "latent_entropy_map": emap,
            "global": metrics_for_arrays(original, recon),
            "roi": group_summary(bm, roi),
            "background": group_summary(bm, background),
            "q": q_summary(qmap_external, roi),
            "latent_stats": latent_stats(latents, input_samples),
            "artifacts": {
                "qmap": str(policy.qmap.relative_to(root)),
                "summary_tsv": str(policy.q_summary.relative_to(root)),
                "bitstream": str(policy.latent.relative_to(root)),
                "reconstruction": str(policy.recon.relative_to(root)),
            },
        }
        shutil.copyfile(policy.q_summary, summaries_dir / f"{policy.key}_source_summary.tsv")

    summary_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    for key, data in policy_data.items():
        row = {
            "policy": key,
            "label": data["label"],
            "q_mean": data["q"]["q_mean"],
            "q_min": data["q"]["q_min"],
            "q_max": data["q"]["q_max"],
            "q_unique": data["q"]["q_unique"],
            "q_roi_mean": data["q"]["q_roi_mean"],
            "q_background_mean": data["q"]["q_background_mean"],
            "global_mse": data["global"]["mse"],
            "global_psnr_db": data["global"]["psnr_db"],
            "global_mae": data["global"]["mae"],
            "roi_blocks": data["roi"]["blocks"],
            "roi_mse": data["roi"]["mse"],
            "roi_psnr_db": data["roi"]["psnr_db"],
            "background_blocks": data["background"]["blocks"],
            "background_mse": data["background"]["mse"],
            "background_psnr_db": data["background"]["psnr_db"],
            "latent_zero_pct": data["latent_stats"]["zero_pct"],
            "latent_entropy_bits_per_symbol": data["latent_stats"]["entropy_bits_per_symbol"],
            "ideal_bps_per_input_sample": data["latent_stats"]["ideal_bps_per_input_sample"],
            "zlib_bps_per_input_sample": data["latent_stats"]["zlib_bps_per_input_sample"],
            "latent_mean_abs": data["latent_stats"]["mean_abs"],
            "latent_unique_values": data["latent_stats"]["unique_values"],
        }
        summary_rows.append(row)
        latent_rows.append({"policy": key, **data["latent_stats"]})

    adaptive = policy_data["adaptive_s8"]
    focus = policy_data["vegetation_focus_bgq128"]
    q204 = policy_data["q204"]
    validation = {
        "all_qmaps_1024_bytes": all(data["q"]["qmap_bytes"] == 1024 for data in policy_data.values()),
        "q204_psnr_reference": 76.7255,
        "q204_psnr_db": q204["global"]["psnr_db"],
        "q204_psnr_reference_ok": abs(q204["global"]["psnr_db"] - 76.7255) <= 0.02,
        "roi_blocks": int(np.sum(roi)),
        "background_blocks": int(np.sum(background)),
        "focus_roi_delta_vs_adaptive_db": focus["roi"]["psnr_db"] - adaptive["roi"]["psnr_db"],
        "focus_background_delta_vs_adaptive_db": focus["background"]["psnr_db"] - adaptive["background"]["psnr_db"],
        "focus_global_delta_vs_adaptive_db": focus["global"]["psnr_db"] - adaptive["global"]["psnr_db"],
        "focus_q_mean_delta_vs_adaptive": focus["q"]["q_mean"] - adaptive["q"]["q_mean"],
        "focus_zero_delta_vs_adaptive_pp": focus["latent_stats"]["zero_pct"] - adaptive["latent_stats"]["zero_pct"],
        "focus_entropy_delta_vs_adaptive": focus["latent_stats"]["entropy_bits_per_symbol"] - adaptive["latent_stats"]["entropy_bits_per_symbol"],
        "focus_ideal_bps_delta_vs_adaptive": focus["latent_stats"]["ideal_bps_per_input_sample"] - adaptive["latent_stats"]["ideal_bps_per_input_sample"],
        "focus_zlib_bps_delta_vs_adaptive": focus["latent_stats"]["zlib_bps_per_input_sample"] - adaptive["latent_stats"]["zlib_bps_per_input_sample"],
    }
    validation.update(
        {
            "focus_roi_kept_or_improved_vs_adaptive": validation["focus_roi_delta_vs_adaptive_db"] >= 0.0,
            "focus_background_degraded_vs_adaptive": validation["focus_background_delta_vs_adaptive_db"] < 0.0,
            "focus_q_mean_lower_than_adaptive": validation["focus_q_mean_delta_vs_adaptive"] < 0.0,
            "focus_zeros_higher_than_adaptive": validation["focus_zero_delta_vs_adaptive_pp"] > 0.0,
            "focus_entropy_lower_than_adaptive": validation["focus_entropy_delta_vs_adaptive"] < 0.0,
        }
    )

    write_maps(maps_dir, ndvi, roi, policy_data)
    write_block_evidence(outdir / "block_correctness_evidence.csv", ndvi, roi, policy_data)

    summary_fields = [
        "policy",
        "label",
        "q_mean",
        "q_min",
        "q_max",
        "q_unique",
        "q_roi_mean",
        "q_background_mean",
        "global_mse",
        "global_psnr_db",
        "global_mae",
        "roi_blocks",
        "roi_mse",
        "roi_psnr_db",
        "background_blocks",
        "background_mse",
        "background_psnr_db",
        "latent_zero_pct",
        "latent_entropy_bits_per_symbol",
        "ideal_bps_per_input_sample",
        "zlib_bps_per_input_sample",
        "latent_mean_abs",
        "latent_unique_values",
    ]
    write_csv(outdir / "correctness_demo_summary.csv", summary_rows, summary_fields)
    write_csv(
        outdir / "latent_policy_summary.csv",
        latent_rows,
        [
            "policy",
            "latent_samples",
            "zero_pct",
            "entropy_bits_per_symbol",
            "ideal_bits",
            "ideal_bytes",
            "ideal_bps_per_input_sample",
            "zlib_bytes_level9",
            "zlib_bps_per_input_sample",
            "zlib_ratio_vs_int32",
            "mean_abs",
            "max_abs",
            "unique_values",
        ],
    )
    summary = {
        "checkpoint": str(args.output_dir),
        "input": {
            "raw": str(args.input),
            "bands": args.bands,
            "height": args.height,
            "width": args.width,
            "block_size": args.block_size,
            "band_layout": "sentinel2-8",
            "threshold": args.threshold,
            "lambda": args.lambda_value,
            "max_lambda": args.max_lambda,
        },
        "commands": commands,
        "summary_rows": summary_rows,
        "validation": validation,
        "artifacts": {
            "summary_csv": str(args.output_dir / "correctness_demo_summary.csv"),
            "summary_md": str(args.output_dir / "correctness_demo_report.md"),
            "block_evidence": str(args.output_dir / "block_correctness_evidence.csv"),
            "latent_policy_summary": str(args.output_dir / "latent_policy_summary.csv"),
            "maps": str(args.output_dir / "maps"),
        },
    }
    write_json(outdir / "correctness_demo_summary.json", summary)
    write_report(outdir / "correctness_demo_report.md", summary_rows, validation)

    print(f"[OK] Demo reproducible completada: {outdir}")
    print(f"q204_psnr={validation['q204_psnr_db']:.4f} dB")
    print(
        "focus_vs_adaptive: "
        f"roi_delta={validation['focus_roi_delta_vs_adaptive_db']:.4f} dB, "
        f"background_delta={validation['focus_background_delta_vs_adaptive_db']:.4f} dB, "
        f"zero_delta={validation['focus_zero_delta_vs_adaptive_pp']:.4f} pp, "
        f"entropy_delta={validation['focus_entropy_delta_vs_adaptive']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
