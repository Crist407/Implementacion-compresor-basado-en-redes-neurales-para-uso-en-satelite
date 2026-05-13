#!/usr/bin/env python3
"""
Construye un paquete de evidencia para la politica semantica focus.

El script es auxiliar: no participa en la ruta final de Raspberry. Lee
artefactos ya generados por C, cruza calidad local, NDVI, Q-map y estadistica
de latentes, y produce tablas/mapas para documentar que la ROI conserva
calidad mientras el fondo se degrada.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MAX_U16 = 65535.0


@dataclass(frozen=True)
class PolicySpec:
    key: str
    label: str
    bitstream: Path
    reconstruction: Path
    qmap: Path | None = None


def psnr_from_mse(mse: float) -> float:
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((MAX_U16 * MAX_U16) / mse)


def finite_float(value: float) -> float | str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
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


def load_raw_u16(path: Path, bands: int, height: int, width: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint16)
    expected = bands * height * width
    if data.size != expected:
        raise ValueError(f"{path}: tamano inesperado {data.size}, esperado {expected}")
    return data.reshape(bands, height, width)


def load_qmap(path: Path, q_height: int, q_width: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    expected = q_height * q_width
    if data.size != expected:
        raise ValueError(f"{path}: Q-map tiene {data.size} bytes, esperado {expected}")
    return data.reshape(q_height, q_width)


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
        latent = np.fromfile(f, dtype=np.int32)

    expected_latents = bands * num_filters * q_height * q_width
    if latent.size != expected_latents:
        raise ValueError(f"{path}: latentes {latent.size}, esperado {expected_latents}")

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
    latent = latent.reshape(bands, num_filters, q_height, q_width)
    return header, qmap, latent


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


def read_semantic_tsv(path: Path, q_height: int, q_width: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    ndvi = np.full((q_height, q_width), np.nan, dtype=np.float64)
    roi = np.zeros((q_height, q_width), dtype=bool)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            by = int(row["block_y"])
            bx = int(row["block_x"])
            index_mean = float(row["index_mean"])
            match = int(row["semantic_match"]) == 1
            expected = index_mean >= threshold
            if match != expected:
                raise ValueError(
                    f"{path}: ROI inconsistente en bloque ({by},{bx}): "
                    f"semantic_match={match}, index_mean={index_mean}, threshold={threshold}"
                )
            ndvi[by, bx] = index_mean
            roi[by, bx] = match
    if np.isnan(ndvi).any():
        raise ValueError(f"{path}: faltan valores NDVI")
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
    }


def latent_stats(latent: np.ndarray) -> dict[str, Any]:
    flat = latent.reshape(-1)
    abs_flat = np.abs(flat.astype(np.int64))
    values, counts = np.unique(flat, return_counts=True)
    probs = counts.astype(np.float64) / float(flat.size)
    entropy = float(-np.sum(probs * np.log2(probs)))
    return {
        "samples": int(flat.size),
        "mean_abs": float(np.mean(abs_flat)),
        "max_abs": int(np.max(abs_flat)),
        "zero_pct": float(np.mean(flat == 0) * 100.0),
        "unique_values": int(values.size),
        "entropy_bits_per_symbol": entropy,
    }


def latent_zero_map(latent: np.ndarray) -> np.ndarray:
    return np.mean(latent == 0, axis=(0, 1)) * 100.0


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def normalize_to_u8(data: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        return np.zeros(data.shape, dtype=np.uint8)
    x = (data.astype(np.float64) - vmin) / (vmax - vmin)
    x = np.clip(x, 0.0, 1.0)
    return np.rint(x * 255.0).astype(np.uint8)


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


def upsample(image: np.ndarray, factor: int) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def diverging_rgb(data: np.ndarray, max_abs: float | None = None) -> np.ndarray:
    arr = data.astype(np.float64)
    if max_abs is None:
        max_abs = float(np.max(np.abs(arr)))
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


def write_map_pair(path32: Path, path512: Path, image: np.ndarray, is_rgb: bool = False, factor: int = 16) -> None:
    if is_rgb:
        write_ppm(path32, image)
        write_ppm(path512, upsample(image, factor))
    else:
        write_pgm(path32, image)
        write_pgm(path512, upsample(image, factor))


def write_maps(
    outdir: Path,
    ndvi: np.ndarray,
    roi: np.ndarray,
    policies: dict[str, dict[str, Any]],
    adaptive_key: str,
    focus_key: str,
) -> None:
    maps32 = outdir / "maps_32x32"
    maps512 = outdir / "maps_512x512"

    write_map_pair(
        maps32 / "ndvi_fixed_minus1_plus1.pgm",
        maps512 / "ndvi_fixed_minus1_plus1.pgm",
        normalize_to_u8(ndvi, -1.0, 1.0),
    )
    write_map_pair(
        maps32 / "roi_mask_ndvi_ge_040.pgm",
        maps512 / "roi_mask_ndvi_ge_040.pgm",
        (roi.astype(np.uint8) * 255),
    )

    for key, data in policies.items():
        q_img = normalize_to_u8(data["qmap"], 128.0, 255.0)
        zero_img = normalize_to_u8(data["latent_zero_map"], 0.0, 100.0)
        mse_img = normalize_to_u8(data["block_metrics"]["mse"], 0.0, float(np.max(data["block_metrics"]["mse"])))
        write_map_pair(maps32 / f"qmap_{key}.pgm", maps512 / f"qmap_{key}.pgm", q_img)
        write_map_pair(maps32 / f"latent_zero_pct_{key}.pgm", maps512 / f"latent_zero_pct_{key}.pgm", zero_img)
        write_map_pair(maps32 / f"mse_{key}.pgm", maps512 / f"mse_{key}.pgm", mse_img)

    focus = policies[focus_key]
    adaptive = policies[adaptive_key]
    delta_q = focus["qmap"].astype(np.float64) - adaptive["qmap"].astype(np.float64)
    delta_mse = focus["block_metrics"]["mse"] - adaptive["block_metrics"]["mse"]
    delta_psnr = focus["block_metrics"]["psnr_db"] - adaptive["block_metrics"]["psnr_db"]
    delta_zero = focus["latent_zero_map"] - adaptive["latent_zero_map"]

    for name, arr in [
        ("delta_q_focus_minus_adaptive", delta_q),
        ("delta_mse_focus_minus_adaptive", delta_mse),
        ("delta_psnr_focus_minus_adaptive", delta_psnr),
        ("delta_zero_pct_focus_minus_adaptive", delta_zero),
    ]:
        rgb = diverging_rgb(arr)
        write_map_pair(maps32 / f"{name}.ppm", maps512 / f"{name}.ppm", rgb, is_rgb=True)


def write_latent_histograms(path: Path, policies: dict[str, dict[str, Any]]) -> None:
    fields = ["policy", "latent_value", "count", "probability"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, data in policies.items():
            flat = data["latent"].reshape(-1)
            values, counts = np.unique(flat, return_counts=True)
            total = float(flat.size)
            for value, count in zip(values, counts):
                writer.writerow(
                    {
                        "policy": key,
                        "latent_value": int(value),
                        "count": int(count),
                        "probability": float(count / total),
                    }
                )


def write_block_evidence(path: Path, ndvi: np.ndarray, roi: np.ndarray, policies: dict[str, dict[str, Any]]) -> None:
    keys = list(policies.keys())
    fields = ["block_y", "block_x", "ndvi", "roi"]
    for key in keys:
        fields.extend([f"q_{key}", f"mse_{key}", f"psnr_db_{key}", f"latent_zero_pct_{key}"])
    fields.extend(
        [
            "delta_q_focus_vs_adaptive",
            "delta_mse_focus_vs_adaptive",
            "delta_psnr_focus_vs_adaptive",
            "delta_zero_pct_focus_vs_adaptive",
        ]
    )

    rows: list[dict[str, Any]] = []
    focus = policies["focus_bgq128"]
    adaptive = policies["adaptive_s8"]
    for by in range(ndvi.shape[0]):
        for bx in range(ndvi.shape[1]):
            row: dict[str, Any] = {
                "block_y": by,
                "block_x": bx,
                "ndvi": float(ndvi[by, bx]),
                "roi": int(roi[by, bx]),
            }
            for key in keys:
                row[f"q_{key}"] = int(policies[key]["qmap"][by, bx])
                row[f"mse_{key}"] = float(policies[key]["block_metrics"]["mse"][by, bx])
                row[f"psnr_db_{key}"] = float(policies[key]["block_metrics"]["psnr_db"][by, bx])
                row[f"latent_zero_pct_{key}"] = float(policies[key]["latent_zero_map"][by, bx])
            row["delta_q_focus_vs_adaptive"] = int(focus["qmap"][by, bx]) - int(adaptive["qmap"][by, bx])
            row["delta_mse_focus_vs_adaptive"] = float(
                focus["block_metrics"]["mse"][by, bx] - adaptive["block_metrics"]["mse"][by, bx]
            )
            row["delta_psnr_focus_vs_adaptive"] = float(
                focus["block_metrics"]["psnr_db"][by, bx] - adaptive["block_metrics"]["psnr_db"][by, bx]
            )
            row["delta_zero_pct_focus_vs_adaptive"] = float(
                focus["latent_zero_map"][by, bx] - adaptive["latent_zero_map"][by, bx]
            )
            rows.append(row)

    write_csv(path, rows, fields)


def write_markdown(path: Path, summary_rows: list[dict[str, Any]], validation: dict[str, Any]) -> None:
    lines = [
        "# Focus Evidence Report",
        "",
        "Comparacion local de politicas sobre la imagen canonica Sentinel-2 de 8 bandas.",
        "",
        "| Politica | Q medio | PSNR global | PSNR ROI | PSNR fondo | Zeros lat. | Entropia |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['policy']} | {row['q_mean']:.4f} | {row['global_psnr_db']:.4f} | "
            f"{row['roi_psnr_db']:.4f} | {row['background_psnr_db']:.4f} | "
            f"{row['latent_zero_pct']:.2f}% | {row['latent_entropy_bits_per_symbol']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Validacion clave",
            "",
            f"- ROI focus vs adaptive: {validation['roi_psnr_delta_focus_vs_adaptive_db']:.4f} dB.",
            f"- Fondo focus vs adaptive: {validation['background_psnr_delta_focus_vs_adaptive_db']:.4f} dB.",
            f"- Q medio focus: {validation['focus_q_mean']:.4f}.",
            f"- Zeros latentes focus: {validation['focus_latent_zero_pct']:.4f}%.",
            f"- Entropia latente focus: {validation['focus_latent_entropy_bits_per_symbol']:.4f} bits/simbolo.",
            "",
            "La reduccion real de ancho de banda queda pendiente del codificador entropico; "
            "esta evidencia mide condiciones estadisticas favorables para ese paso.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera evidencia visual/estadistica para focus vegetation.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/checkpoints/20260512_focus_evidence_report"),
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw"),
    )
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.40)
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def require_files(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Faltan artefactos requeridos:\n" + "\n".join(missing))


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    outdir = resolve(root, args.output_dir)
    original_path = resolve(root, args.original)
    q_height = args.height // args.block_size
    q_width = args.width // args.block_size

    specs = [
        PolicySpec(
            "q204",
            "Q constante 204",
            Path("output/checkpoints/20260506_baseline_constant_qmap/latent.bin"),
            Path("output/checkpoints/20260506_baseline_constant_qmap/reconstructed.raw"),
            None,
        ),
        PolicySpec(
            "adaptive_s8",
            "Adaptativo s=8",
            Path("output/checkpoints/20260507_adaptive_difficulty_qmap/latent_adaptive_s8.bin"),
            Path("output/checkpoints/20260507_adaptive_difficulty_qmap/reconstructed_adaptive_s8.raw"),
            Path("output/checkpoints/20260507_adaptive_difficulty_qmap/qmap_adaptive_s8.bin"),
        ),
        PolicySpec(
            "semantic_boost_th040_b16",
            "Vegetation boost-only th=0.40 b=16",
            Path("output/checkpoints/20260509_semantic_validation_vegetation/latent_th040_b16.bin"),
            Path("output/checkpoints/20260509_semantic_validation_vegetation/reconstructed_th040_b16.raw"),
            Path("output/checkpoints/20260509_semantic_validation_vegetation/qmap_th040_b16.bin"),
        ),
        PolicySpec(
            "focus_bgq128",
            "Focus vegetation fg=16 bgQ=128",
            Path("output/checkpoints/20260511_semantic_background_q_vegetation/latent_bgq128.bin"),
            Path("output/checkpoints/20260511_semantic_background_q_vegetation/reconstructed_bgq128.raw"),
            Path("output/checkpoints/20260511_semantic_background_q_vegetation/qmap_bgq128.bin"),
        ),
    ]
    semantic_tsv = resolve(
        root,
        Path("output/checkpoints/20260511_semantic_background_q_vegetation/semantic_bgq128_summary.tsv"),
    )
    required = [original_path, semantic_tsv]
    for spec in specs:
        required.extend([resolve(root, spec.bitstream), resolve(root, spec.reconstruction)])
        if spec.qmap is not None:
            required.append(resolve(root, spec.qmap))
    require_files(required)

    outdir.mkdir(parents=True, exist_ok=True)
    original = load_raw_u16(original_path, args.bands, args.height, args.width)
    ndvi, roi = read_semantic_tsv(semantic_tsv, q_height, q_width, args.threshold)
    background = ~roi

    policies: dict[str, dict[str, Any]] = {}
    for spec in specs:
        bitstream_path = resolve(root, spec.bitstream)
        recon_path = resolve(root, spec.reconstruction)
        header, bitstream_qmap, latent = read_bitstream(bitstream_path)
        if (header["bands"], header["height"], header["width"]) != (args.bands, args.height, args.width):
            raise ValueError(f"{bitstream_path}: dimensiones incompatibles {header}")
        if spec.qmap is not None:
            qmap = load_qmap(resolve(root, spec.qmap), q_height, q_width)
            if not np.array_equal(qmap, bitstream_qmap):
                raise ValueError(f"{spec.key}: Q-map externo no coincide con Q-map del bitstream")
        else:
            qmap = bitstream_qmap
        recon = load_raw_u16(recon_path, args.bands, args.height, args.width)
        bm = block_metrics(original, recon, args.block_size)
        policies[spec.key] = {
            "label": spec.label,
            "bitstream": str(spec.bitstream),
            "reconstruction": str(spec.reconstruction),
            "qmap": qmap,
            "latent": latent,
            "latent_zero_map": latent_zero_map(latent),
            "global": metrics_for_arrays(original, recon),
            "block_metrics": bm,
            "roi": group_summary(bm, roi),
            "background": group_summary(bm, background),
            "q": q_summary(qmap, roi),
            "latent_stats": latent_stats(latent),
            "top10_worst_mean_mse": float(np.mean(np.sort(bm["mse"].reshape(-1))[-10:])),
        }

    summary_rows: list[dict[str, Any]] = []
    for key, data in policies.items():
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
            "roi_blocks": data["roi"]["blocks"],
            "roi_mse": data["roi"]["mse"],
            "roi_psnr_db": data["roi"]["psnr_db"],
            "background_blocks": data["background"]["blocks"],
            "background_mse": data["background"]["mse"],
            "background_psnr_db": data["background"]["psnr_db"],
            "latent_zero_pct": data["latent_stats"]["zero_pct"],
            "latent_entropy_bits_per_symbol": data["latent_stats"]["entropy_bits_per_symbol"],
            "latent_mean_abs": data["latent_stats"]["mean_abs"],
            "latent_unique_values": data["latent_stats"]["unique_values"],
            "top10_worst_mean_mse": data["top10_worst_mean_mse"],
        }
        summary_rows.append(row)

    adaptive = policies["adaptive_s8"]
    focus = policies["focus_bgq128"]
    validation = {
        "roi_blocks": int(np.sum(roi)),
        "background_blocks": int(np.sum(background)),
        "roi_matches_threshold": True,
        "focus_q_mean": focus["q"]["q_mean"],
        "focus_latent_zero_pct": focus["latent_stats"]["zero_pct"],
        "focus_latent_entropy_bits_per_symbol": focus["latent_stats"]["entropy_bits_per_symbol"],
        "roi_psnr_delta_focus_vs_adaptive_db": focus["roi"]["psnr_db"] - adaptive["roi"]["psnr_db"],
        "background_psnr_delta_focus_vs_adaptive_db": focus["background"]["psnr_db"] - adaptive["background"]["psnr_db"],
        "global_psnr_delta_focus_vs_adaptive_db": focus["global"]["psnr_db"] - adaptive["global"]["psnr_db"],
        "q_mean_delta_focus_vs_semantic_boost": focus["q"]["q_mean"] - policies["semantic_boost_th040_b16"]["q"]["q_mean"],
        "latent_zero_delta_focus_vs_adaptive": focus["latent_stats"]["zero_pct"] - adaptive["latent_stats"]["zero_pct"],
        "latent_entropy_delta_focus_vs_adaptive": (
            focus["latent_stats"]["entropy_bits_per_symbol"] - adaptive["latent_stats"]["entropy_bits_per_symbol"]
        ),
    }

    summary = {
        "checkpoint": str(args.output_dir),
        "input": {
            "original": str(args.original),
            "bands": args.bands,
            "height": args.height,
            "width": args.width,
            "block_size": args.block_size,
            "threshold": args.threshold,
            "semantic_tsv": str(semantic_tsv.relative_to(root) if semantic_tsv.is_relative_to(root) else semantic_tsv),
        },
        "policies": {key: {k: v for k, v in data.items() if k not in {"qmap", "latent", "latent_zero_map", "block_metrics"}} for key, data in policies.items()},
        "summary_rows": summary_rows,
        "validation": validation,
        "artifacts": {
            "summary_json": str(args.output_dir / "focus_evidence_summary.json"),
            "summary_csv": str(args.output_dir / "focus_evidence_summary.csv"),
            "summary_md": str(args.output_dir / "focus_evidence_summary.md"),
            "block_evidence_csv": str(args.output_dir / "block_evidence.csv"),
            "latent_histograms_csv": str(args.output_dir / "latent_histograms.csv"),
            "maps_32x32": str(args.output_dir / "maps_32x32"),
            "maps_512x512": str(args.output_dir / "maps_512x512"),
        },
    }

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
        "roi_blocks",
        "roi_mse",
        "roi_psnr_db",
        "background_blocks",
        "background_mse",
        "background_psnr_db",
        "latent_zero_pct",
        "latent_entropy_bits_per_symbol",
        "latent_mean_abs",
        "latent_unique_values",
        "top10_worst_mean_mse",
    ]
    write_json(outdir / "focus_evidence_summary.json", summary)
    write_csv(outdir / "focus_evidence_summary.csv", summary_rows, summary_fields)
    write_markdown(outdir / "focus_evidence_summary.md", summary_rows, validation)
    write_block_evidence(outdir / "block_evidence.csv", ndvi, roi, policies)
    write_latent_histograms(outdir / "latent_histograms.csv", policies)
    write_maps(outdir, ndvi, roi, policies, adaptive_key="adaptive_s8", focus_key="focus_bgq128")

    print(f"[OK] Evidencia focus generada en {outdir}")
    print(
        "focus_bgq128: "
        f"q_mean={validation['focus_q_mean']:.4f}, "
        f"roi_delta={validation['roi_psnr_delta_focus_vs_adaptive_db']:.4f} dB, "
        f"background_delta={validation['background_psnr_delta_focus_vs_adaptive_db']:.4f} dB, "
        f"zeros={validation['focus_latent_zero_pct']:.2f}%, "
        f"entropy={validation['focus_latent_entropy_bits_per_symbol']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
