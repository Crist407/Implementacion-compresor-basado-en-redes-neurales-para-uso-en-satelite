#!/usr/bin/env python3
"""
Auditoria reproducible del catalogo de presets semanticos SORTENY.

La decision de Q-map sigue estando en C (sorteny_semantic_qmap). Este script
solo orquesta ejecuciones, resume los TSV generados por C y guarda evidencia.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


MAX_U16 = 65535.0
QMAP_BYTES = 32 * 32


PRESETS: list[dict[str, Any]] = [
    {
        "preset": "vegetation",
        "index": "NDVI",
        "eval_mode": "normalized_diff",
        "bands": "B08,B04",
        "threshold": 0.4,
        "sentinel2_8": "yes",
        "sentinel2_13": "yes",
        "notes": "Vegetacion; validable con el dataset actual.",
    },
    {
        "preset": "water",
        "index": "NDMI",
        "eval_mode": "normalized_diff",
        "bands": "B08,B11",
        "threshold": 0.2,
        "sentinel2_8": "missing_B11",
        "sentinel2_13": "yes",
        "notes": "Humedad/agua; requiere SWIR1.",
    },
    {
        "preset": "burned",
        "index": "NBR",
        "eval_mode": "normalized_diff",
        "bands": "B08,B12",
        "threshold": 0.1,
        "sentinel2_8": "missing_B12",
        "sentinel2_13": "yes",
        "notes": "Quemado; requiere SWIR2.",
    },
    {
        "preset": "snow",
        "index": "NDSI",
        "eval_mode": "normalized_diff",
        "bands": "B03,B11",
        "threshold": 0.4,
        "sentinel2_8": "missing_B11",
        "sentinel2_13": "yes",
        "notes": "Nieve; requiere SWIR1.",
    },
    {
        "preset": "water_body",
        "index": "NDWI",
        "eval_mode": "normalized_diff",
        "bands": "B03,B08",
        "threshold": 0.3,
        "sentinel2_8": "yes",
        "sentinel2_13": "yes",
        "notes": "Masas de agua con Green/NIR.",
    },
    {
        "preset": "chlorophyll",
        "index": "NDCI",
        "eval_mode": "normalized_diff",
        "bands": "B05,B04",
        "threshold": 0.1,
        "sentinel2_8": "yes",
        "sentinel2_13": "yes",
        "notes": "Clorofila/red-edge.",
    },
    {
        "preset": "vegetation_green",
        "index": "GNDVI",
        "eval_mode": "normalized_diff",
        "bands": "B08,B03",
        "threshold": 0.5,
        "sentinel2_8": "yes",
        "sentinel2_13": "yes",
        "notes": "Vegetacion alternativa con banda verde.",
    },
    {
        "preset": "clouds",
        "index": "CBY",
        "eval_mode": "cloud_cby",
        "bands": "B03,B04[,B11]",
        "threshold": 0.5,
        "sentinel2_8": "basic_without_B11",
        "sentinel2_13": "enhanced_with_B11",
        "notes": "Con 8 bandas usa version basica; precision pendiente de validar.",
    },
    {
        "preset": "barren_soil",
        "index": "BSI",
        "eval_mode": "bsi",
        "bands": "B02,B04,B08,B11",
        "threshold": 0.0,
        "sentinel2_8": "missing_B11",
        "sentinel2_13": "yes",
        "notes": "Suelo desnudo; requiere SWIR1.",
    },
    {
        "preset": "burned_area",
        "index": "BAIS2",
        "eval_mode": "bais2",
        "bands": "B04,B06,B07,B8A,B12",
        "threshold": 0.5,
        "sentinel2_8": "missing_B12",
        "sentinel2_13": "yes",
        "notes": "Area quemada BAIS2; requiere SWIR2.",
    },
    {
        "preset": "uniform",
        "index": "none",
        "eval_mode": "none",
        "bands": "-",
        "threshold": math.nan,
        "sentinel2_8": "yes",
        "sentinel2_13": "yes",
        "notes": "Control sin semantica.",
    },
    {
        "preset": "manual",
        "index": "manual_roi",
        "eval_mode": "manual",
        "bands": "-",
        "threshold": 0.5,
        "sentinel2_8": "yes",
        "sentinel2_13": "yes",
        "notes": "ROI manual de 32x32 bloques.",
    },
]


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
        return float(obj)
    if isinstance(obj, float):
        if math.isinf(obj):
            return "inf"
        if math.isnan(obj):
            return "nan"
    return obj


def run_cmd(cmd: list[str], log_path: Path | None = None) -> float:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    dt = time.perf_counter() - t0
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "$ " + " ".join(cmd) + "\n\nSTDOUT:\n" + proc.stdout + "\nSTDERR:\n" + proc.stderr,
            encoding="utf-8",
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return dt


def executable(path: Path) -> str:
    return str(path.resolve()) if path.exists() else str(path)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def make_manual_roi(path: Path, q_width: int = 32, q_height: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    roi = np.zeros((q_height, q_width), dtype=np.uint8)
    roi[8:24, 8:24] = 1
    roi.tofile(path)


def read_semantic_tsv(path: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    if len(rows) != QMAP_BYTES:
        raise ValueError(f"{path}: {len(rows)} bloques, esperado {QMAP_BYTES}")

    q = np.array([int(r["final_q"]) for r in rows], dtype=np.int32)
    base_q = np.array([int(r["base_q"]) for r in rows], dtype=np.int32)
    matches = np.array([int(r["semantic_match"]) for r in rows], dtype=np.int32)
    missing = sum(1 for r in rows if r["reason"] == "missing_bands")
    no_valid = sum(1 for r in rows if r["reason"] == "no_valid_pixels")
    boosted = sum(1 for r in rows if int(r["foreground_boost_applied"]) != 0)
    penalized = sum(1 for r in rows if int(r["background_penalty_applied"]) != 0)
    fixed_bg = sum(1 for r in rows if int(r["background_fixed_q_applied"]) != 0)
    valid_index = [float(r["index_mean"]) for r in rows if r["index_mean"] != "nan"]

    return {
        "blocks": len(rows),
        "semantic_matches": int(np.sum(matches)),
        "missing_bands": int(missing),
        "no_valid_pixels": int(no_valid),
        "boosted_blocks": int(boosted),
        "penalized_blocks": int(penalized),
        "fixed_background_blocks": int(fixed_bg),
        "index_mean": float(np.mean(valid_index)) if valid_index else math.nan,
        "index_min": float(np.min(valid_index)) if valid_index else math.nan,
        "index_max": float(np.max(valid_index)) if valid_index else math.nan,
        "q_min": int(np.min(q)),
        "q_max": int(np.max(q)),
        "q_mean": float(np.mean(q)),
        "base_q_mean": float(np.mean(base_q)),
    }


def psnr_from_mse(mse: float) -> float:
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((MAX_U16 * MAX_U16) / mse)


def load_raw(path: Path, bands: int, height: int, width: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint16)
    expected = bands * height * width
    if data.size != expected:
        raise ValueError(f"{path}: {data.size} muestras, esperado {expected}")
    return data.reshape(bands, height, width)


def quality_metrics(original: Path, reconstructed: Path, bands: int, height: int, width: int) -> dict[str, float]:
    a = load_raw(original, bands, height, width).astype(np.float64)
    b = load_raw(reconstructed, bands, height, width).astype(np.float64)
    d = a - b
    ad = np.abs(d)
    mse = float(np.mean(d * d))
    return {
        "mse": mse,
        "psnr_db": psnr_from_mse(mse),
        "mae": float(np.mean(ad)),
        "max_abs": float(np.max(ad)),
        "exact_pct": float(np.mean(ad == 0.0) * 100.0),
    }


def latent_stats(bitstream: Path) -> dict[str, Any]:
    with bitstream.open("rb") as f:
        header = np.fromfile(f, dtype=np.uint16, count=5)
        if header.size != 5:
            raise ValueError(f"{bitstream}: cabecera incompleta")
        bands, height, width, _datatype, num_filters = [int(x) for x in header]
        q_size = (height // 16) * (width // 16)
        qmap = np.fromfile(f, dtype=np.uint8, count=q_size)
        if qmap.size != q_size:
            raise ValueError(f"{bitstream}: Q-map incompleto")
        latents = np.fromfile(f, dtype=np.int32)
    expected = bands * num_filters * (height // 16) * (width // 16)
    if latents.size != expected:
        raise ValueError(f"{bitstream}: latentes {latents.size}, esperado {expected}")
    values, counts = np.unique(latents, return_counts=True)
    probs = counts.astype(np.float64) / float(latents.size)
    return {
        "samples": int(latents.size),
        "mean_abs": float(np.mean(np.abs(latents.astype(np.int64)))),
        "max_abs": int(np.max(np.abs(latents.astype(np.int64)))),
        "zero_pct": float(np.mean(latents == 0) * 100.0),
        "unique_values": int(values.size),
        "entropy_bits_per_symbol": float(-np.sum(probs * np.log2(probs))),
    }


def base_semantic_cmd(args: argparse.Namespace, preset: str, qmap: Path, tsv: Path) -> list[str]:
    cmd = [
        executable(args.semantic_bin),
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
        "--foreground-boost",
        str(args.foreground_boost),
        "--semantic-policy",
        "boost-only",
    ]
    return cmd


def run_preset(
    args: argparse.Namespace,
    preset: str,
    raw: Path | None,
    qmap: Path,
    tsv: Path,
    log: Path,
    *,
    band_map: Path | None = None,
    roi_map: Path | None = None,
) -> dict[str, Any]:
    qmap.parent.mkdir(parents=True, exist_ok=True)
    tsv.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = base_semantic_cmd(args, preset, qmap, tsv)
    if preset == "manual":
        cmd.extend(["--roi-map", str(roi_map)])
    elif raw is not None:
        cmd.extend(["--input", str(raw)])

    if band_map is not None and preset != "manual":
        cmd.extend(["--band-map", str(band_map)])
    elif preset != "manual":
        cmd.extend(["--band-layout", "sentinel2-8"])

    dt = run_cmd(cmd, log_path=log)
    if qmap.stat().st_size != QMAP_BYTES:
        raise ValueError(f"{qmap}: {qmap.stat().st_size} bytes, esperado {QMAP_BYTES}")
    stats = read_semantic_tsv(tsv)
    return {
        "preset": preset,
        "time_s": dt,
        "qmap_bytes": qmap.stat().st_size,
        "qmap": str(qmap),
        "summary_tsv": str(tsv),
        **stats,
    }


def write_markdown_report(path: Path, catalog_rows: list[dict[str, Any]], dataset_summary: list[dict[str, Any]], validation: dict[str, Any]) -> None:
    lines = [
        "# Semantic Preset Catalog Audit",
        "",
        f"Checkpoint: `{validation['checkpoint']}`",
        f"Dataset RAW validos: {validation['dataset_valid_raw']}/{validation['dataset_raw_files']}",
        f"Vegetation regression byte-identical: {validation['vegetation_regression_equal']}",
        f"Band-map equivalente a `sentinel2-8`: {validation['band_map_equivalence']}",
        f"Pipeline smoke preset: `{validation['full_pipeline']['preset']}`, PSNR {validation['full_pipeline']['quality']['psnr_db']:.4f} dB",
        "",
        "## Canonical Presets",
        "",
        "| Preset | Index | Bands | Eval | 8-band | Matches | Missing | Q mean |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in catalog_rows:
        lines.append(
            f"| {row['preset']} | {row['index']} | {row['bands']} | {row['eval_mode']} | "
            f"{row['sentinel2_8']} | {row['semantic_matches']} | {row['missing_bands']} | {row['q_mean']:.4f} |"
        )

    lines.extend(["", "## Dataset Summary", "", "| Preset | Files | Avg matches | Max matches | Avg missing | Avg Q |", "|---|---:|---:|---:|---:|---:|"])
    for row in dataset_summary:
        lines.append(
            f"| {row['preset']} | {row['files']} | {row['semantic_matches_mean']:.4f} | "
            f"{row['semantic_matches_max']} | {row['missing_bands_mean']:.4f} | {row['q_mean_mean']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `clouds` usa la variante CBY basica con 8 bandas; el filtro B11 queda pendiente de dataset de 13 bandas.",
            "- `water`, `burned`, `snow`, `barren_soil` y `burned_area` no inventan mascaras cuando faltan B11/B12.",
            "- Python solo resume evidencia; la generacion de Q-map se ejecuta en C.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita los presets semanticos de sorteny_semantic_qmap.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/Sentinel2A_crop_test"))
    parser.add_argument("--canonical", type=Path, default=Path("data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/checkpoints/20260515_semantic_preset_catalog_audit"))
    parser.add_argument("--calibration", type=Path, default=Path("output/checkpoints/20260507_c_fixed_quality_qmap_wide/fq_calibration.tsv"))
    parser.add_argument("--semantic-bin", type=Path, default=Path("./sorteny_semantic_qmap"))
    parser.add_argument("--encoder", type=Path, default=Path("./sorteny_compressor"))
    parser.add_argument("--decoder", type=Path, default=Path("./sorteny_decompressor"))
    parser.add_argument("--analyzer", type=Path, default=Path("src/python/analysis/analyze_block_quality.py"))
    parser.add_argument("--encoder-weights", type=Path, default=Path("weights/encoder"))
    parser.add_argument("--decoder-weights", type=Path, default=Path("weights/decoder"))
    parser.add_argument("--previous-vegetation-qmap", type=Path, default=Path("output/checkpoints/20260508_semantic_qmap_c/qmap_semantic_vegetation.bin"))
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--q-mean", type=int, default=204)
    parser.add_argument("--adaptive-strength", type=float, default=8.0)
    parser.add_argument("--foreground-boost", type=int, default=8)
    parser.add_argument("--max-files", type=int, default=0, help="0 procesa todos los RAW del dataset.")
    parser.add_argument("--full-pipeline-preset", default="clouds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = args.output_dir
    canonical_qmaps = outdir / "canonical_qmaps"
    canonical_tsv = outdir / "canonical_tsv"
    dataset_qmaps = outdir / "dataset_qmaps"
    dataset_tsv = outdir / "dataset_tsv"
    logs = outdir / "logs"
    smoke_dir = outdir / "full_pipeline_smoke"
    for path in [canonical_qmaps, canonical_tsv, dataset_qmaps, dataset_tsv, logs, smoke_dir]:
        path.mkdir(parents=True, exist_ok=True)

    band_map = outdir / "sentinel2_8_band_map.tsv"
    roi_map = outdir / "manual_roi_center.bin"
    make_band_map(band_map)
    make_manual_roi(roi_map)

    expected_bytes = args.bands * args.height * args.width * 2
    raw_files = sorted(args.dataset_dir.glob("*.raw"))
    if args.max_files > 0:
        raw_files = raw_files[: args.max_files]

    manifest_rows = []
    valid_raws = []
    for raw in raw_files:
        size = raw.stat().st_size
        ok = size == expected_bytes
        manifest_rows.append(
            {
                "file": str(raw),
                "size_bytes": size,
                "expected_bytes": expected_bytes,
                "status": "ok" if ok else "bad_size",
            }
        )
        if ok:
            valid_raws.append(raw)
    if not valid_raws:
        raise FileNotFoundError(f"No hay RAW validos en {args.dataset_dir}")

    canonical_rows: list[dict[str, Any]] = []
    for spec in PRESETS:
        preset = spec["preset"]
        row = run_preset(
            args,
            preset,
            args.canonical if preset != "manual" else None,
            canonical_qmaps / f"{preset}.bin",
            canonical_tsv / f"{preset}.tsv",
            logs / f"canonical_{preset}.log",
            roi_map=roi_map,
        )
        canonical_rows.append({**spec, **row})

    qmap_layout = canonical_qmaps / "vegetation_layout.bin"
    tsv_layout = canonical_tsv / "vegetation_layout.tsv"
    run_preset(
        args,
        "vegetation",
        args.canonical,
        qmap_layout,
        tsv_layout,
        logs / "vegetation_layout_regression.log",
        roi_map=roi_map,
    )
    qmap_by_map = canonical_qmaps / "vegetation_band_map.bin"
    tsv_by_map = canonical_tsv / "vegetation_band_map.tsv"
    run_preset(
        args,
        "vegetation",
        args.canonical,
        qmap_by_map,
        tsv_by_map,
        logs / "vegetation_band_map_equivalence.log",
        band_map=band_map,
        roi_map=roi_map,
    )
    band_map_equivalence = qmap_layout.read_bytes() == qmap_by_map.read_bytes()

    previous_exists = args.previous_vegetation_qmap.exists()
    vegetation_regression_equal = (
        previous_exists and args.previous_vegetation_qmap.read_bytes() == (canonical_qmaps / "vegetation.bin").read_bytes()
    )

    dataset_rows: list[dict[str, Any]] = []
    for raw in valid_raws:
        stem = raw.stem
        for spec in PRESETS:
            preset = spec["preset"]
            preset_qmap_dir = dataset_qmaps / preset
            preset_tsv_dir = dataset_tsv / preset
            row = run_preset(
                args,
                preset,
                raw if preset != "manual" else None,
                preset_qmap_dir / f"{stem}.bin",
                preset_tsv_dir / f"{stem}.tsv",
                logs / "dataset" / preset / f"{stem}.log",
                roi_map=roi_map,
            )
            dataset_rows.append(
                {
                    **spec,
                    "file": str(raw),
                    **row,
                }
            )

    dataset_summary: list[dict[str, Any]] = []
    for spec in PRESETS:
        preset = spec["preset"]
        rows = [r for r in dataset_rows if r["preset"] == preset]
        dataset_summary.append(
            {
                **spec,
                "files": len(rows),
                "semantic_matches_mean": float(np.mean([r["semantic_matches"] for r in rows])),
                "semantic_matches_min": int(np.min([r["semantic_matches"] for r in rows])),
                "semantic_matches_max": int(np.max([r["semantic_matches"] for r in rows])),
                "missing_bands_mean": float(np.mean([r["missing_bands"] for r in rows])),
                "missing_bands_min": int(np.min([r["missing_bands"] for r in rows])),
                "missing_bands_max": int(np.max([r["missing_bands"] for r in rows])),
                "q_mean_mean": float(np.mean([r["q_mean"] for r in rows])),
                "q_mean_min": float(np.min([r["q_mean"] for r in rows])),
                "q_mean_max": float(np.max([r["q_mean"] for r in rows])),
            }
        )

    smoke_preset = args.full_pipeline_preset
    smoke_qmap = canonical_qmaps / f"{smoke_preset}.bin"
    smoke_latent = smoke_dir / f"latent_{smoke_preset}.bin"
    smoke_recon = smoke_dir / f"reconstructed_{smoke_preset}.raw"
    smoke_quality_json = smoke_dir / f"block_quality_{smoke_preset}.json"
    smoke_quality_csv = smoke_dir / f"block_quality_{smoke_preset}.csv"
    run_cmd(
        [
            executable(args.encoder),
            str(args.canonical),
            "0.1",
            str(smoke_latent),
            str(args.encoder_weights),
            "0.125",
            str(smoke_qmap),
        ],
        log_path=logs / f"smoke_compress_{smoke_preset}.log",
    )
    run_cmd(
        [executable(args.decoder), str(smoke_latent), str(smoke_recon), str(args.decoder_weights), "0.125"],
        log_path=logs / f"smoke_decompress_{smoke_preset}.log",
    )
    run_cmd(
        [
            "python3",
            str(args.analyzer),
            str(args.canonical),
            str(smoke_recon),
            "--bands",
            str(args.bands),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--bitstream",
            str(smoke_latent),
            "--output-json",
            str(smoke_quality_json),
            "--output-csv",
            str(smoke_quality_csv),
        ],
        log_path=logs / f"smoke_analyze_{smoke_preset}.log",
    )

    validation = {
        "checkpoint": str(outdir),
        "dataset_raw_files": len(raw_files),
        "dataset_valid_raw": len(valid_raws),
        "expected_raw_bytes": expected_bytes,
        "qmap_bytes_expected": QMAP_BYTES,
        "qmap_size_all_ok": all(r["qmap_bytes"] == QMAP_BYTES for r in canonical_rows + dataset_rows),
        "band_map_equivalence": band_map_equivalence,
        "vegetation_regression_reference": str(args.previous_vegetation_qmap),
        "vegetation_regression_reference_exists": previous_exists,
        "vegetation_regression_equal": vegetation_regression_equal,
        "full_pipeline": {
            "preset": smoke_preset,
            "input": str(args.canonical),
            "qmap": str(smoke_qmap),
            "latent": str(smoke_latent),
            "reconstruction": str(smoke_recon),
            "quality": quality_metrics(args.canonical, smoke_recon, args.bands, args.height, args.width),
            "latent_stats": latent_stats(smoke_latent),
        },
    }

    catalog_fields = [
        "preset",
        "index",
        "bands",
        "eval_mode",
        "threshold",
        "sentinel2_8",
        "sentinel2_13",
        "semantic_matches",
        "missing_bands",
        "q_min",
        "q_max",
        "q_mean",
        "index_mean",
        "notes",
        "qmap",
        "summary_tsv",
    ]
    dataset_fields = [
        "file",
        *catalog_fields,
        "blocks",
        "qmap_bytes",
    ]
    dataset_summary_fields = [
        "preset",
        "index",
        "bands",
        "eval_mode",
        "threshold",
        "sentinel2_8",
        "sentinel2_13",
        "files",
        "semantic_matches_mean",
        "semantic_matches_min",
        "semantic_matches_max",
        "missing_bands_mean",
        "missing_bands_min",
        "missing_bands_max",
        "q_mean_mean",
        "q_mean_min",
        "q_mean_max",
        "notes",
    ]

    write_csv(outdir / "dataset_manifest.csv", manifest_rows, ["file", "size_bytes", "expected_bytes", "status"])
    write_csv(outdir / "preset_catalog.csv", canonical_rows, catalog_fields)
    write_csv(outdir / "canonical_preset_audit.csv", canonical_rows, catalog_fields + ["blocks", "qmap_bytes"])
    write_csv(outdir / "dataset_preset_audit.csv", dataset_rows, dataset_fields)
    write_csv(outdir / "dataset_preset_summary.csv", dataset_summary, dataset_summary_fields)
    write_json(outdir / "preset_catalog.json", {"rows": canonical_rows})
    write_json(outdir / "dataset_preset_audit.json", {"rows": dataset_rows})
    write_json(outdir / "dataset_preset_summary.json", {"rows": dataset_summary})
    write_json(outdir / "validation_summary.json", validation)
    write_json(
        outdir / "checkpoint_summary.json",
        {
            "validation": validation,
            "canonical_presets": canonical_rows,
            "dataset_summary": dataset_summary,
        },
    )
    write_markdown_report(outdir / "preset_catalog_audit.md", canonical_rows, dataset_summary, validation)

    print(f"[OK] Auditoria semantica completada: {outdir}")
    print(f"RAW validos: {len(valid_raws)}/{len(raw_files)}")
    print(f"Vegetation regression byte-identical: {vegetation_regression_equal}")
    print(f"Band-map equivalente a sentinel2-8: {band_map_equivalence}")
    print(f"Smoke {smoke_preset} PSNR: {validation['full_pipeline']['quality']['psnr_db']:.4f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
