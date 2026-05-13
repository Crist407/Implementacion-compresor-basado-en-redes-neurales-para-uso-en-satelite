#!/usr/bin/env python3
"""
Verifica la coherencia de la formula fixed-quality y resume latentes.

Este script es auxiliar de auditoria: no forma parte de la ruta final de
Raspberry. Relee artefactos ya generados por C/Python, recomputa las decisiones
de Q-map y mide proxies reproducibles para un codificador entropico futuro.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MAX_U16 = 65535.0


@dataclass(frozen=True)
class CalibrationBlock:
    block_y: int
    block_x: int
    c0: float
    c1: float
    r2: float
    valid: bool
    q_baseline: int
    q_min: int
    q_max: int
    max_lambda: float
    mod_a: float
    mod_b: float
    mse_at_baseline: float


@dataclass(frozen=True)
class PolicySpec:
    key: str
    label: str
    bitstream: Path
    qmap: Path | None = None
    semantic_tsv: Path | None = None


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


def c_lround(value: float) -> int:
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def psnr_from_mse(mse: float) -> float:
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((MAX_U16 * MAX_U16) / mse)


def mse_from_psnr(psnr_db: float) -> float:
    return (MAX_U16 * MAX_U16) / (10.0 ** (psnr_db / 10.0))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_calibration(path: Path, q_height: int, q_width: int) -> np.ndarray:
    blocks: np.ndarray = np.empty((q_height, q_width), dtype=object)
    loaded = np.zeros((q_height, q_width), dtype=bool)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t")
        for row in reader:
            by = int(row["block_y"])
            bx = int(row["block_x"])
            blocks[by, bx] = CalibrationBlock(
                block_y=by,
                block_x=bx,
                c0=float(row["c0"]),
                c1=float(row["c1"]),
                r2=float(row["r2"]),
                valid=bool(int(row["valid"])),
                q_baseline=int(row["q_baseline"]),
                q_min=int(row["q_min"]),
                q_max=int(row["q_max"]),
                max_lambda=float(row["max_lambda"]),
                mod_a=float(row["mod_a"]),
                mod_b=float(row["mod_b"]),
                mse_at_baseline=float(row["mse_at_baseline"]),
            )
            loaded[by, bx] = True
    if not np.all(loaded):
        missing = np.argwhere(~loaded)
        raise ValueError(f"{path}: faltan bloques de calibracion, primero {missing[0].tolist()}")
    return blocks


def lambda_from_q(q: int, max_lambda: float) -> float:
    return (float(q) / 255.0) * max_lambda


def mod_from_lambda(c: CalibrationBlock, lamb: float) -> float:
    return c.mod_a * lamb + c.mod_b


def predicted_mse_for_q(c: CalibrationBlock, q: int) -> float:
    m = mod_from_lambda(c, lambda_from_q(q, c.max_lambda))
    if m <= 0.0:
        m = 1e-12
    return c.c0 + (c.c1 / (m * m))


def select_q_for_target(c: CalibrationBlock, target_mse: float, target_q: int | None = None) -> int:
    if not c.valid or c.c1 <= 0.0 or c.mod_a <= 0.0 or c.max_lambda <= 0.0:
        return int(np.clip(c.q_baseline, c.q_min, c.q_max))

    local_target = target_mse
    if target_q is not None:
        q_ref = int(np.clip(target_q, c.q_min, c.q_max))
        local_target = predicted_mse_for_q(c, q_ref)

    if not (local_target > c.c0):
        return c.q_max
    target_mod = math.sqrt(c.c1 / (local_target - c.c0))
    lamb = (target_mod - c.mod_b) / c.mod_a
    q_float = (lamb / c.max_lambda) * 255.0
    if not math.isfinite(q_float):
        return int(np.clip(c.q_baseline, c.q_min, c.q_max))
    return int(np.clip(c_lround(q_float), c.q_min, c.q_max))


def recompute_qmap_from_q(blocks: np.ndarray, q_ref: int) -> np.ndarray:
    out = np.zeros(blocks.shape, dtype=np.uint8)
    for by in range(blocks.shape[0]):
        for bx in range(blocks.shape[1]):
            out[by, bx] = select_q_for_target(blocks[by, bx], 0.0, target_q=q_ref)
    return out


def recompute_qmap_target_psnr(blocks: np.ndarray, target_psnr: float) -> np.ndarray:
    target_mse = mse_from_psnr(target_psnr)
    out = np.zeros(blocks.shape, dtype=np.uint8)
    for by in range(blocks.shape[0]):
        for bx in range(blocks.shape[1]):
            out[by, bx] = select_q_for_target(blocks[by, bx], target_mse)
    return out


def recompute_qmap_adaptive(blocks: np.ndarray, q_mean: int, strength: float) -> np.ndarray:
    values = [blocks[by, bx].mse_at_baseline for by in range(blocks.shape[0]) for bx in range(blocks.shape[1]) if blocks[by, bx].valid and blocks[by, bx].mse_at_baseline > 0.0]
    logs = np.log(np.array(values, dtype=np.float64))
    log_mean = float(np.mean(logs)) if logs.size else 0.0
    log_std = float(np.std(logs)) if logs.size else 1.0
    if log_std <= 0.0:
        log_std = 1.0
    out = np.zeros(blocks.shape, dtype=np.uint8)
    for by in range(blocks.shape[0]):
        for bx in range(blocks.shape[1]):
            c = blocks[by, bx]
            if not c.valid or c.mse_at_baseline <= 0.0:
                q = int(np.clip(c.q_baseline, c.q_min, c.q_max))
            else:
                z = (math.log(c.mse_at_baseline) - log_mean) / log_std
                q = int(np.clip(c_lround(float(q_mean) + strength * z), c.q_min, c.q_max))
            out[by, bx] = q
    return out


def load_qmap(path: Path, q_height: int, q_width: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    expected = q_height * q_width
    if data.size != expected:
        raise ValueError(f"{path}: Q-map {data.size} bytes, esperado {expected}")
    return data.reshape(q_height, q_width)


def compare_qmaps(name: str, expected_path: Path, recomputed: np.ndarray) -> dict[str, Any]:
    expected = load_qmap(expected_path, recomputed.shape[0], recomputed.shape[1])
    diff = expected.astype(np.int16) - recomputed.astype(np.int16)
    return {
        "name": name,
        "c_qmap": str(expected_path),
        "byte_equal": bool(np.array_equal(expected, recomputed)),
        "different_blocks": int(np.count_nonzero(diff)),
        "max_abs_diff": int(np.max(np.abs(diff))) if diff.size else 0,
        "c_q_mean": float(np.mean(expected)),
        "py_q_mean": float(np.mean(recomputed)),
        "c_q_min": int(np.min(expected)),
        "c_q_max": int(np.max(expected)),
        "py_q_min": int(np.min(recomputed)),
        "py_q_max": int(np.max(recomputed)),
    }


def load_block_quality_csv(path: Path, q_height: int, q_width: int) -> np.ndarray:
    arr = np.zeros((q_height, q_width), dtype=np.float64)
    seen = np.zeros((q_height, q_width), dtype=bool)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by = int(row["block_y"])
            bx = int(row["block_x"])
            arr[by, bx] = float(row["mse"])
            seen[by, bx] = True
    if not np.all(seen):
        raise ValueError(f"{path}: faltan bloques")
    return arr


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = float(np.sum((actual - predicted) ** 2))
    mean = float(np.mean(actual))
    ss_tot = float(np.sum((actual - mean) ** 2))
    if ss_tot <= 0.0:
        return math.nan
    return 1.0 - (ss_res / ss_tot)


def verify_formula_against_sweep(blocks: np.ndarray, sweep_json: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sweep = json.loads(sweep_json.read_text(encoding="utf-8"))
    q_values = [int(x["q"]) for x in sweep["results"]]
    actual_stack = []
    predicted_stack = []
    rows = []
    for item in sweep["results"]:
        q = int(item["q"])
        directory = Path(item["directory"])
        quality_csv = directory / "block_quality.csv"
        actual = load_block_quality_csv(quality_csv, blocks.shape[0], blocks.shape[1])
        predicted = np.zeros_like(actual)
        for by in range(blocks.shape[0]):
            for bx in range(blocks.shape[1]):
                predicted[by, bx] = predicted_mse_for_q(blocks[by, bx], q)
        residual = predicted - actual
        actual_stack.append(actual)
        predicted_stack.append(predicted)
        actual_mean = float(np.mean(actual))
        predicted_mean = float(np.mean(predicted))
        rows.append(
            {
                "q": q,
                "actual_mse_mean": actual_mean,
                "predicted_mse_mean": predicted_mean,
                "mean_error_pred_minus_actual": float(np.mean(residual)),
                "mae_mse": float(np.mean(np.abs(residual))),
                "rmse_mse": float(math.sqrt(np.mean(residual * residual))),
                "actual_psnr_db": psnr_from_mse(actual_mean),
                "predicted_psnr_db": psnr_from_mse(predicted_mean),
                "psnr_error_pred_minus_actual_db": psnr_from_mse(predicted_mean) - psnr_from_mse(actual_mean),
                "r2_blocks": r2_score(actual.reshape(-1), predicted.reshape(-1)),
            }
        )

    actual_all = np.stack(actual_stack, axis=0)
    predicted_all = np.stack(predicted_stack, axis=0)
    residual_all = predicted_all - actual_all
    global_mse = [float(np.mean(x)) for x in actual_stack]
    global_psnr = [psnr_from_mse(x) for x in global_mse]
    block_mono_ok = 0
    block_mono_violations = 0
    for by in range(blocks.shape[0]):
        for bx in range(blocks.shape[1]):
            seq = actual_all[:, by, bx]
            if np.all(np.diff(seq) <= 1e-9):
                block_mono_ok += 1
            else:
                block_mono_violations += 1
    summary = {
        "q_values": q_values,
        "global_mse_nonincreasing_with_q": bool(all(a >= b for a, b in zip(global_mse, global_mse[1:]))),
        "global_psnr_nondecreasing_with_q": bool(all(a <= b for a, b in zip(global_psnr, global_psnr[1:]))),
        "block_mse_nonincreasing_count": block_mono_ok,
        "block_mse_nonincreasing_violations": block_mono_violations,
        "overall_mae_mse": float(np.mean(np.abs(residual_all))),
        "overall_rmse_mse": float(math.sqrt(np.mean(residual_all * residual_all))),
        "overall_mean_error_pred_minus_actual": float(np.mean(residual_all)),
        "overall_r2": r2_score(actual_all.reshape(-1), predicted_all.reshape(-1)),
        "mean_block_r2_from_calibration": float(np.mean([blocks[by, bx].r2 for by in range(blocks.shape[0]) for bx in range(blocks.shape[1])])),
    }
    return rows, summary


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
    expected = int(bands) * int(num_filters) * q_height * q_width
    if latents.size != expected:
        raise ValueError(f"{path}: latentes {latents.size}, esperado {expected}")
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
    latents = latents.reshape(int(bands), int(num_filters), q_height, q_width)
    return header, qmap, latents


def entropy_of_values(values: np.ndarray) -> float:
    if values.size == 0:
        return math.nan
    _vals, counts = np.unique(values, return_counts=True)
    probs = counts.astype(np.float64) / float(values.size)
    return float(-np.sum(probs * np.log2(probs)))


def stats_for_values(values: np.ndarray, *, zlib_level: int = 9) -> dict[str, Any]:
    flat = values.reshape(-1)
    abs_flat = np.abs(flat.astype(np.int64))
    entropy = entropy_of_values(flat)
    compressed = zlib.compress(flat.astype("<i4", copy=False).tobytes(), level=zlib_level)
    return {
        "samples": int(flat.size),
        "zero_pct": float(np.mean(flat == 0) * 100.0),
        "entropy_bits_per_symbol": entropy,
        "ideal_bits": float(entropy * float(flat.size)) if math.isfinite(entropy) else math.nan,
        "ideal_bytes": float((entropy * float(flat.size)) / 8.0) if math.isfinite(entropy) else math.nan,
        "mean_abs": float(np.mean(abs_flat)),
        "max_abs": int(np.max(abs_flat)) if flat.size else 0,
        "unique_values": int(np.unique(flat).size),
        "zlib_bytes_level9": int(len(compressed)),
        "raw_int32_bytes": int(flat.size * 4),
        "zlib_ratio_vs_int32": float(len(compressed) / float(flat.size * 4)) if flat.size else math.nan,
    }


def read_roi_from_semantic_tsv(path: Path, q_height: int, q_width: int) -> np.ndarray:
    roi = np.zeros((q_height, q_width), dtype=bool)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            roi[int(row["block_y"]), int(row["block_x"])] = int(row["semantic_match"]) != 0
    return roi


def write_pgm(path: Path, image: np.ndarray) -> None:
    img = image.astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode("ascii"))
        f.write(img.tobytes())


def normalize_to_u8(data: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    arr = data.astype(np.float64)
    lo = float(np.nanmin(arr)) if vmin is None else vmin
    hi = float(np.nanmax(arr)) if vmax is None else vmax
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    x = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(x * 255.0).astype(np.uint8)


def latent_zero_map(latents: np.ndarray) -> np.ndarray:
    return np.mean(latents == 0, axis=(0, 1)) * 100.0


def latent_entropy_map(latents: np.ndarray) -> np.ndarray:
    out = np.zeros(latents.shape[2:], dtype=np.float64)
    for by in range(latents.shape[2]):
        for bx in range(latents.shape[3]):
            out[by, bx] = entropy_of_values(latents[:, :, by, bx].reshape(-1))
    return out


def summarize_latent_policies(policies: list[PolicySpec], outdir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    hist_rows: list[dict[str, Any]] = []
    validation: dict[str, Any] = {}
    maps_dir = outdir / "maps_32x32"

    roi = None
    for policy in policies:
        if policy.semantic_tsv and policy.semantic_tsv.exists():
            header, _qmap_tmp, _lat_tmp = read_bitstream(policy.bitstream)
            roi = read_roi_from_semantic_tsv(policy.semantic_tsv, header["q_height"], header["q_width"])
            break

    for policy in policies:
        header, qmap, latents = read_bitstream(policy.bitstream)
        if policy.qmap is not None and policy.qmap.exists():
            external_qmap = load_qmap(policy.qmap, header["q_height"], header["q_width"])
            if not np.array_equal(qmap, external_qmap):
                raise ValueError(f"{policy.key}: Q-map externo no coincide con bitstream")

        stats = stats_for_values(latents)
        input_samples = header["bands"] * header["height"] * header["width"]
        row = {
            "policy": policy.key,
            "label": policy.label,
            "bitstream": str(policy.bitstream),
            "q_min": int(np.min(qmap)),
            "q_max": int(np.max(qmap)),
            "q_mean": float(np.mean(qmap)),
            "q_unique": int(np.unique(qmap).size),
            **stats,
            "ideal_bps_per_input_sample": float(stats["ideal_bits"] / float(input_samples)),
            "zlib_bps_per_input_sample": float((stats["zlib_bytes_level9"] * 8) / float(input_samples)),
        }
        if roi is not None:
            background = ~roi
            row["roi_blocks"] = int(np.sum(roi))
            row["background_blocks"] = int(np.sum(background))
            row["roi_zero_pct"] = stats_for_values(latents[:, :, roi])["zero_pct"] if np.any(roi) else math.nan
            row["background_zero_pct"] = stats_for_values(latents[:, :, background])["zero_pct"] if np.any(background) else math.nan
            row["roi_entropy_bits_per_symbol"] = stats_for_values(latents[:, :, roi])["entropy_bits_per_symbol"] if np.any(roi) else math.nan
            row["background_entropy_bits_per_symbol"] = stats_for_values(latents[:, :, background])["entropy_bits_per_symbol"] if np.any(background) else math.nan
        summary_rows.append(row)

        zero_map = latent_zero_map(latents)
        entropy_map = latent_entropy_map(latents)
        write_pgm(maps_dir / f"latent_zero_pct_{policy.key}.pgm", normalize_to_u8(zero_map, 0.0, 100.0))
        write_pgm(maps_dir / f"latent_entropy_{policy.key}.pgm", normalize_to_u8(entropy_map))

        values, counts = np.unique(latents.reshape(-1), return_counts=True)
        total = float(latents.size)
        for value, count in zip(values, counts):
            hist_rows.append(
                {
                    "policy": policy.key,
                    "latent_value": int(value),
                    "count": int(count),
                    "probability": float(count / total),
                }
            )

        for band in range(header["bands"]):
            band_stats = stats_for_values(latents[band, :, :, :])
            band_rows.append({"policy": policy.key, "band": band, **band_stats})
        for channel in range(header["num_filters"]):
            ch_stats = stats_for_values(latents[:, channel, :, :])
            channel_rows.append({"policy": policy.key, "latent_channel": channel, **ch_stats})
        for by in range(header["q_height"]):
            for bx in range(header["q_width"]):
                block_stats = stats_for_values(latents[:, :, by, bx])
                block_rows.append(
                    {
                        "policy": policy.key,
                        "block_y": by,
                        "block_x": bx,
                        "q": int(qmap[by, bx]),
                        "roi": int(roi[by, bx]) if roi is not None else "",
                        **block_stats,
                    }
                )

    by_key = {row["policy"]: row for row in summary_rows}
    if "adaptive_s8" in by_key and "focus_bgq128" in by_key:
        validation["focus_zero_delta_vs_adaptive_pct"] = by_key["focus_bgq128"]["zero_pct"] - by_key["adaptive_s8"]["zero_pct"]
        validation["focus_entropy_delta_vs_adaptive"] = (
            by_key["focus_bgq128"]["entropy_bits_per_symbol"] - by_key["adaptive_s8"]["entropy_bits_per_symbol"]
        )
        validation["focus_ideal_bps_delta_vs_adaptive"] = (
            by_key["focus_bgq128"]["ideal_bps_per_input_sample"] - by_key["adaptive_s8"]["ideal_bps_per_input_sample"]
        )
        validation["focus_zlib_bps_delta_vs_adaptive"] = (
            by_key["focus_bgq128"]["zlib_bps_per_input_sample"] - by_key["adaptive_s8"]["zlib_bps_per_input_sample"]
        )
        validation["focus_increases_zeros_vs_adaptive"] = validation["focus_zero_delta_vs_adaptive_pct"] > 0.0
        validation["focus_reduces_entropy_vs_adaptive"] = validation["focus_entropy_delta_vs_adaptive"] < 0.0
    return summary_rows, band_rows, channel_rows, block_rows, hist_rows, validation


def write_formula_mapping(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Correspondencia Formula Paper -> SORTENY C",
                "",
                "La auditoria usa la relacion del paper de fixed-quality:",
                "",
                "- Modelo del paper: `MSE_hat = MSE0 + alpha*MSE0/(4*M(lambda)^2)`.",
                "- Ajuste lineal del modulador: `M(lambda) = a*lambda + b`.",
                "- Inversion hacia calidad objetivo: `lambda_hat = (sqrt(alpha*MSE0/(4*(MSE_target-MSE0))) - b) / a`.",
                "- Cuantizacion lateral: `Q = round(255*(lambda-lambda_min)/(lambda_max-lambda_min))`.",
                "",
                "En la implementacion actual:",
                "",
                "- `c0` representa el termino base equivalente a `MSE0` para cada bloque.",
                "- `c1` absorbe `alpha*MSE0/4`, de modo que el modelo queda `MSE ~= c0 + c1/M(lambda)^2`.",
                "- `mod_a` y `mod_b` son los coeficientes globales del ajuste `M(lambda)=mod_a*lambda+mod_b`.",
                "- `max_lambda=0.125` y `lambda_min=0`, asi que `lambda_Q = Q/255*max_lambda`.",
                "- `sorteny_fq_qmap` invierte el modelo por bloque, redondea con semantica tipo C `lround` y clampa a `q_min..q_max` calibrados.",
                "",
                "Esta equivalencia no demuestra por si sola exactitud cientifica: la precision real se mide comparando el MSE predicho contra el MSE observado en el barrido de Q.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_formula_markdown(path: Path, qmap_rows: list[dict[str, Any]], sweep_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Formula Verification",
        "",
        "## Q-map recomputation",
        "",
        "| Caso | Byte equal | Bloques distintos | Q medio C | Q medio Python |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in qmap_rows:
        lines.append(f"| {row['name']} | {row['byte_equal']} | {row['different_blocks']} | {row['c_q_mean']:.4f} | {row['py_q_mean']:.4f} |")
    lines.extend(
        [
            "",
            "## Sweep prediction error",
            "",
            f"- MSE global monotono con Q: {summary['global_mse_nonincreasing_with_q']}.",
            f"- PSNR global monotono con Q: {summary['global_psnr_nondecreasing_with_q']}.",
            f"- MAE MSE global por bloque/Q: {summary['overall_mae_mse']:.6f}.",
            f"- RMSE MSE global por bloque/Q: {summary['overall_rmse_mse']:.6f}.",
            f"- R2 global sobre todos los puntos bloque/Q: {summary['overall_r2']:.6f}.",
            f"- R2 medio por bloque desde calibracion: {summary['mean_block_r2_from_calibration']:.6f}.",
            "",
            "| Q | MSE real | MSE pred. | Error PSNR pred-real | R2 bloques |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sweep_rows:
        lines.append(
            f"| {row['q']} | {row['actual_mse_mean']:.6f} | {row['predicted_mse_mean']:.6f} | "
            f"{row['psnr_error_pred_minus_actual_db']:.6f} | {row['r2_blocks']:.6f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latent_markdown(path: Path, rows: list[dict[str, Any]], validation: dict[str, Any]) -> None:
    lines = [
        "# Latent Coding Proxy Summary",
        "",
        "Los latentes siguen escritos como `int32`; estas metricas estiman condiciones favorables para un codificador entropico futuro.",
        "",
        "| Politica | Q medio | Zeros | Entropia | Bits ideales/input sample | zlib bps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['q_mean']:.4f} | {row['zero_pct']:.4f}% | "
            f"{row['entropy_bits_per_symbol']:.6f} | {row['ideal_bps_per_input_sample']:.6f} | "
            f"{row['zlib_bps_per_input_sample']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Focus vs adaptive",
            "",
            f"- Delta zeros focus-adaptive: {validation.get('focus_zero_delta_vs_adaptive_pct', math.nan):.6f} puntos porcentuales.",
            f"- Delta entropia focus-adaptive: {validation.get('focus_entropy_delta_vs_adaptive', math.nan):.6f} bits/simbolo.",
            f"- Delta bits ideales/input sample: {validation.get('focus_ideal_bps_delta_vs_adaptive', math.nan):.6f}.",
            f"- Delta zlib bps/input sample: {validation.get('focus_zlib_bps_delta_vs_adaptive', math.nan):.6f}.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verifica formula fixed-quality y evidencia de latentes.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/checkpoints/20260516_formula_and_latent_coding_evidence"))
    parser.add_argument("--calibration", type=Path, default=Path("output/checkpoints/20260507_c_fixed_quality_qmap_wide/fq_calibration.tsv"))
    parser.add_argument("--sweep-json", type=Path, default=Path("output/checkpoints/20260507_q_sweep_calibration_wide/sweep_results.json"))
    parser.add_argument("--qmap-q204", type=Path, default=Path("output/checkpoints/20260507_c_fixed_quality_qmap_wide/qmap_from_q204.bin"))
    parser.add_argument("--qmap-target-psnr", type=Path, default=Path("output/checkpoints/20260507_c_fixed_quality_qmap_wide/qmap_target_psnr_76_8.bin"))
    parser.add_argument("--qmap-adaptive-s8", type=Path, default=Path("output/checkpoints/20260507_adaptive_difficulty_qmap/qmap_adaptive_s8.bin"))
    parser.add_argument("--target-psnr", type=float, default=76.8)
    parser.add_argument("--q-mean", type=int, default=204)
    parser.add_argument("--adaptive-strength", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    blocks = load_calibration(args.calibration, 32, 32)

    qmap_checks = [
        compare_qmaps("target_from_q_204", args.qmap_q204, recompute_qmap_from_q(blocks, 204)),
        compare_qmaps("target_psnr_76_8", args.qmap_target_psnr, recompute_qmap_target_psnr(blocks, args.target_psnr)),
        compare_qmaps(
            "adaptive_difficulty_s8",
            args.qmap_adaptive_s8,
            recompute_qmap_adaptive(blocks, args.q_mean, args.adaptive_strength),
        ),
    ]
    sweep_rows, sweep_summary = verify_formula_against_sweep(blocks, args.sweep_json)

    policies = [
        PolicySpec(
            "q204",
            "Q constante 204",
            Path("output/checkpoints/20260506_baseline_constant_qmap/latent.bin"),
        ),
        PolicySpec(
            "adaptive_s8",
            "Adaptativo s=8",
            Path("output/checkpoints/20260507_adaptive_difficulty_qmap/latent_adaptive_s8.bin"),
            Path("output/checkpoints/20260507_adaptive_difficulty_qmap/qmap_adaptive_s8.bin"),
        ),
        PolicySpec(
            "vegetation_boost",
            "Vegetation boost-only th=0.40 b=16",
            Path("output/checkpoints/20260509_semantic_validation_vegetation/latent_th040_b16.bin"),
            Path("output/checkpoints/20260509_semantic_validation_vegetation/qmap_th040_b16.bin"),
            Path("output/checkpoints/20260509_semantic_validation_vegetation/semantic_th040_b16_summary.tsv"),
        ),
        PolicySpec(
            "focus_bgq128",
            "Vegetation focus background Q=128",
            Path("output/checkpoints/20260511_semantic_background_q_vegetation/latent_bgq128.bin"),
            Path("output/checkpoints/20260511_semantic_background_q_vegetation/qmap_bgq128.bin"),
            Path("output/checkpoints/20260511_semantic_background_q_vegetation/semantic_bgq128_summary.tsv"),
        ),
        PolicySpec(
            "clouds",
            "Clouds CBY basic",
            Path("output/checkpoints/20260515_semantic_preset_catalog_audit/full_pipeline_smoke/latent_clouds.bin"),
            Path("output/checkpoints/20260515_semantic_preset_catalog_audit/canonical_qmaps/clouds.bin"),
        ),
    ]
    latent_summary, band_rows, channel_rows, block_rows, hist_rows, latent_validation = summarize_latent_policies(policies, outdir)

    formula_validation = {
        "formula_mapping": {
            "paper_model": "MSE_hat = MSE0 + alpha*MSE0/(4*M(lambda)^2)",
            "repo_model": "MSE ~= c0 + c1/(mod_a*lambda+mod_b)^2",
            "c0": "MSE0",
            "c1": "alpha*MSE0/4",
            "modulator": "M(lambda)=mod_a*lambda+mod_b",
            "lambda_from_q": "lambda_Q=Q/255*max_lambda, lambda_min=0",
        },
        "qmap_checks": qmap_checks,
        "sweep_summary": sweep_summary,
    }
    checkpoint = {
        "checkpoint": str(outdir),
        "calibration": str(args.calibration),
        "sweep": str(args.sweep_json),
        "formula_validation": formula_validation,
        "latent_validation": latent_validation,
    }

    write_formula_mapping(outdir / "paper_formula_mapping.md")
    write_csv(
        outdir / "formula_qmap_recompute.csv",
        qmap_checks,
        ["name", "byte_equal", "different_blocks", "max_abs_diff", "c_q_mean", "py_q_mean", "c_q_min", "c_q_max", "py_q_min", "py_q_max", "c_qmap"],
    )
    write_csv(
        outdir / "formula_sweep_prediction.csv",
        sweep_rows,
        [
            "q",
            "actual_mse_mean",
            "predicted_mse_mean",
            "mean_error_pred_minus_actual",
            "mae_mse",
            "rmse_mse",
            "actual_psnr_db",
            "predicted_psnr_db",
            "psnr_error_pred_minus_actual_db",
            "r2_blocks",
        ],
    )
    write_json(outdir / "formula_verification.json", formula_validation)
    write_markdown = write_formula_markdown
    write_markdown(outdir / "formula_verification.md", qmap_checks, sweep_rows, sweep_summary)
    write_csv(
        outdir / "latent_coding_proxy_summary.csv",
        latent_summary,
        [
            "policy",
            "label",
            "q_min",
            "q_max",
            "q_mean",
            "q_unique",
            "samples",
            "zero_pct",
            "entropy_bits_per_symbol",
            "ideal_bits",
            "ideal_bytes",
            "ideal_bps_per_input_sample",
            "zlib_bytes_level9",
            "zlib_bps_per_input_sample",
            "mean_abs",
            "max_abs",
            "unique_values",
            "roi_zero_pct",
            "background_zero_pct",
            "roi_entropy_bits_per_symbol",
            "background_entropy_bits_per_symbol",
            "bitstream",
        ],
    )
    write_json(outdir / "latent_coding_proxy_summary.json", {"rows": latent_summary, "validation": latent_validation})
    write_latent_markdown(outdir / "latent_coding_proxy_summary.md", latent_summary, latent_validation)
    write_csv(
        outdir / "latent_band_summary.csv",
        band_rows,
        ["policy", "band", "samples", "zero_pct", "entropy_bits_per_symbol", "ideal_bits", "ideal_bytes", "mean_abs", "max_abs", "unique_values", "zlib_bytes_level9", "raw_int32_bytes", "zlib_ratio_vs_int32"],
    )
    write_csv(
        outdir / "latent_channel_summary.csv",
        channel_rows,
        ["policy", "latent_channel", "samples", "zero_pct", "entropy_bits_per_symbol", "ideal_bits", "ideal_bytes", "mean_abs", "max_abs", "unique_values", "zlib_bytes_level9", "raw_int32_bytes", "zlib_ratio_vs_int32"],
    )
    write_csv(
        outdir / "latent_block_summary.csv",
        block_rows,
        ["policy", "block_y", "block_x", "q", "roi", "samples", "zero_pct", "entropy_bits_per_symbol", "ideal_bits", "ideal_bytes", "mean_abs", "max_abs", "unique_values", "zlib_bytes_level9", "raw_int32_bytes", "zlib_ratio_vs_int32"],
    )
    write_csv(outdir / "latent_histograms.csv", hist_rows, ["policy", "latent_value", "count", "probability"])
    write_json(outdir / "checkpoint_summary.json", checkpoint)

    print(f"[OK] Verificacion formula/latentes completada: {outdir}")
    print(f"Q204 byte_equal: {qmap_checks[0]['byte_equal']}")
    print(f"Adaptive s8 byte_equal: {qmap_checks[2]['byte_equal']}")
    print(f"Sweep global MSE monotono: {sweep_summary['global_mse_nonincreasing_with_q']}")
    print(
        "Focus vs adaptive: "
        f"delta_zero={latent_validation.get('focus_zero_delta_vs_adaptive_pct', math.nan):.4f} pp, "
        f"delta_entropy={latent_validation.get('focus_entropy_delta_vs_adaptive', math.nan):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
