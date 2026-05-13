#!/usr/bin/env python3
"""
Validacion ligera de Q-maps semanticos sobre un dataset Sentinel-2 de crops.

La ruta final sigue siendo C. Este script solo orquesta ejecuciones de
sorteny_semantic_qmap y resume evidencias para un lote de RAW BSQ uint16.
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
    if log_path:
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


def latent_stats(bitstream: Path) -> dict[str, float | int]:
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
    entropy = float(-np.sum(probs * np.log2(probs)))
    return {
        "samples": int(latents.size),
        "mean_abs": float(np.mean(np.abs(latents.astype(np.int64)))),
        "max_abs": int(np.max(np.abs(latents.astype(np.int64)))),
        "zero_pct": float(np.mean(latents == 0) * 100.0),
        "unique_values": int(values.size),
        "entropy_bits_per_symbol": entropy,
    }


def read_semantic_tsv(path: Path) -> dict[str, Any]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: TSV vacio")

    q = np.array([int(r["final_q"]) for r in rows], dtype=np.int32)
    base_q = np.array([int(r["base_q"]) for r in rows], dtype=np.int32)
    matches = np.array([int(r["semantic_match"]) for r in rows], dtype=np.int32)
    missing = sum(1 for r in rows if r["reason"] == "missing_bands")
    valid_index = []
    for r in rows:
        if r["index_mean"] != "nan":
            valid_index.append(float(r["index_mean"]))

    roi = matches == 1
    background = ~roi
    return {
        "blocks": len(rows),
        "semantic_matches": int(np.sum(matches)),
        "missing_bands": int(missing),
        "index_mean": float(np.mean(valid_index)) if valid_index else math.nan,
        "index_min": float(np.min(valid_index)) if valid_index else math.nan,
        "index_max": float(np.max(valid_index)) if valid_index else math.nan,
        "q_min": int(np.min(q)),
        "q_max": int(np.max(q)),
        "q_mean": float(np.mean(q)),
        "base_q_mean": float(np.mean(base_q)),
        "q_roi_mean": float(np.mean(q[roi])) if np.any(roi) else math.nan,
        "q_background_mean": float(np.mean(q[background])) if np.any(background) else math.nan,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke semantico batch para crops Sentinel-2 de 8 bandas.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/Sentinel2A_crop_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/checkpoints/20260513_sentinel2a_8band_dataset_validation"))
    parser.add_argument("--calibration", type=Path, default=Path("output/checkpoints/20260507_c_fixed_quality_qmap_wide/fq_calibration.tsv"))
    parser.add_argument("--semantic-bin", type=Path, default=Path("./sorteny_semantic_qmap"))
    parser.add_argument("--encoder", type=Path, default=Path("./sorteny_compressor"))
    parser.add_argument("--decoder", type=Path, default=Path("./sorteny_decompressor"))
    parser.add_argument("--analyzer", type=Path, default=Path("src/python/analysis/analyze_block_quality.py"))
    parser.add_argument("--encoder-weights", type=Path, default=Path("weights/encoder"))
    parser.add_argument("--decoder-weights", type=Path, default=Path("weights/decoder"))
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max-files", type=int, default=0, help="0 procesa todos los RAW.")
    parser.add_argument("--smoke-index", type=int, default=-1, help="-1 elige el crop con mas bloques ROI.")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = args.output_dir
    qmap_dir = outdir / "qmaps"
    tsv_dir = outdir / "semantic_tsv"
    log_dir = outdir / "logs"
    smoke_dir = outdir / "smoke_pipeline"
    outdir.mkdir(parents=True, exist_ok=True)
    qmap_dir.mkdir(parents=True, exist_ok=True)
    tsv_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(args.dataset_dir.glob("*.raw"))
    if args.max_files > 0:
        raw_files = raw_files[: args.max_files]
    if not raw_files:
        raise FileNotFoundError(f"No hay RAW en {args.dataset_dir}")

    expected_bytes = args.bands * args.height * args.width * 2
    manifest_rows = []
    summary_rows = []
    for raw in raw_files:
        size = raw.stat().st_size
        ok = size == expected_bytes
        manifest_rows.append(
            {
                "file": str(raw),
                "size_bytes": size,
                "expected_bytes": expected_bytes,
                "bands": args.bands if ok else "",
                "height": args.height if ok else "",
                "width": args.width if ok else "",
                "status": "ok" if ok else "bad_size",
            }
        )
        if not ok:
            continue

        stem = raw.stem
        qmap = qmap_dir / f"{stem}_vegetation_focus_bgq128.bin"
        tsv = tsv_dir / f"{stem}_vegetation_focus_bgq128.tsv"
        dt = run_cmd(
            [
                executable(args.semantic_bin),
                "--input",
                str(raw),
                "--calibration",
                str(args.calibration),
                "--preset",
                "vegetation",
                "--semantic-policy",
                "focus",
                "--foreground-boost",
                "16",
                "--background-q",
                "128",
                "--threshold",
                "0.40",
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
                "--band-layout",
                "sentinel2-8",
                "--q-mean",
                "204",
                "--adaptive-strength",
                "8",
            ],
            log_path=log_dir / f"{stem}_vegetation_focus.log",
        )
        stats = read_semantic_tsv(tsv)
        summary_rows.append({"file": str(raw), "semantic_time_s": dt, "qmap": str(qmap), "summary_tsv": str(tsv), **stats})

    manifest_fields = ["file", "size_bytes", "expected_bytes", "bands", "height", "width", "status"]
    summary_fields = [
        "file",
        "semantic_time_s",
        "blocks",
        "semantic_matches",
        "missing_bands",
        "index_mean",
        "index_min",
        "index_max",
        "q_min",
        "q_max",
        "q_mean",
        "base_q_mean",
        "q_roi_mean",
        "q_background_mean",
        "qmap",
        "summary_tsv",
    ]
    write_csv(outdir / "dataset_manifest.csv", manifest_rows, manifest_fields)
    write_csv(outdir / "semantic_dataset_summary.csv", summary_rows, summary_fields)

    if args.smoke_index >= 0:
        smoke_file = raw_files[min(args.smoke_index, len(raw_files) - 1)]
    else:
        if summary_rows:
            best = max(summary_rows, key=lambda r: (int(r["semantic_matches"]), float(r["q_mean"])))
            smoke_file = Path(best["file"])
        else:
            smoke_file = raw_files[0]
    smoke_stem = smoke_file.stem
    smoke_qmap = qmap_dir / f"{smoke_stem}_vegetation_focus_bgq128.bin"
    smoke_tsv = tsv_dir / f"{smoke_stem}_vegetation_focus_bgq128.tsv"

    band_map = outdir / "sentinel2_8_band_map.tsv"
    make_band_map(band_map)
    qmap_by_map = outdir / "qmap_band_map_equivalence.bin"
    tsv_by_map = outdir / "qmap_band_map_equivalence.tsv"
    run_cmd(
        [
            executable(args.semantic_bin),
            "--input",
            str(smoke_file),
            "--calibration",
            str(args.calibration),
            "--preset",
            "vegetation",
            "--semantic-policy",
            "focus",
            "--foreground-boost",
            "16",
            "--background-q",
            "128",
            "--threshold",
            "0.40",
            "--output-qmap",
            str(qmap_by_map),
            "--summary-tsv",
            str(tsv_by_map),
            "--bands",
            str(args.bands),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--band-map",
            str(band_map),
            "--q-mean",
            "204",
            "--adaptive-strength",
            "8",
        ],
        log_path=log_dir / "band_map_equivalence.log",
    )
    band_map_equal = smoke_qmap.read_bytes() == qmap_by_map.read_bytes()

    missing_preset_rows = []
    for preset in ["water", "burned", "snow"]:
        qmap = outdir / f"qmap_{preset}_limited_8band.bin"
        tsv = outdir / f"semantic_{preset}_limited_8band.tsv"
        run_cmd(
            [
                executable(args.semantic_bin),
                "--input",
                str(smoke_file),
                "--calibration",
                str(args.calibration),
                "--preset",
                preset,
                "--semantic-policy",
                "focus",
                "--foreground-boost",
                "16",
                "--background-q",
                "128",
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
                "--band-layout",
                "sentinel2-8",
                "--q-mean",
                "204",
                "--adaptive-strength",
                "8",
            ],
            log_path=log_dir / f"{preset}_missing_bands.log",
        )
        stats = read_semantic_tsv(tsv)
        missing_preset_rows.append({"preset": preset, **stats, "qmap": str(qmap), "summary_tsv": str(tsv)})

    smoke_latent = smoke_dir / "latent_vegetation_focus_bgq128.bin"
    smoke_recon = smoke_dir / "reconstructed_vegetation_focus_bgq128.raw"
    smoke_quality_json = smoke_dir / "block_quality_vegetation_focus_bgq128.json"
    smoke_quality_csv = smoke_dir / "block_quality_vegetation_focus_bgq128.csv"
    run_cmd(
        [
            executable(args.encoder),
            str(smoke_file),
            "0.1",
            str(smoke_latent),
            str(args.encoder_weights),
            "0.125",
            str(smoke_qmap),
        ],
        log_path=log_dir / "smoke_compress.log",
    )
    run_cmd(
        [executable(args.decoder), str(smoke_latent), str(smoke_recon), str(args.decoder_weights), "0.125"],
        log_path=log_dir / "smoke_decompress.log",
    )
    run_cmd(
        [
            "python3",
            str(args.analyzer),
            str(smoke_file),
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
        log_path=log_dir / "smoke_analyze_block_quality.log",
    )

    smoke_summary = {
        "input": str(smoke_file),
        "qmap": str(smoke_qmap),
        "semantic_tsv": str(smoke_tsv),
        "latent": str(smoke_latent),
        "reconstruction": str(smoke_recon),
        "block_quality_json": str(smoke_quality_json),
        "global_quality": quality_metrics(smoke_file, smoke_recon, args.bands, args.height, args.width),
        "latent": latent_stats(smoke_latent),
    }

    checkpoint = {
        "checkpoint": str(outdir),
        "dataset_dir": str(args.dataset_dir),
        "raw_files": len(raw_files),
        "valid_raw_files": sum(1 for r in manifest_rows if r["status"] == "ok"),
        "band_order": "B02,B03,B04,B05,B06,B07,B08,B8A",
        "band_map_equivalence": band_map_equal,
        "missing_preset_checks": missing_preset_rows,
        "smoke": smoke_summary,
    }
    write_json(outdir / "semantic_dataset_summary.json", {"rows": summary_rows, "checkpoint": checkpoint})
    write_csv(
        outdir / "missing_band_preset_checks.csv",
        missing_preset_rows,
        ["preset", "blocks", "semantic_matches", "missing_bands", "q_mean", "qmap", "summary_tsv"],
    )
    write_json(outdir / "checkpoint_summary.json", checkpoint)

    print(f"[OK] Dataset smoke completado: {outdir}")
    print(f"RAW validos: {checkpoint['valid_raw_files']}/{len(raw_files)}")
    print(f"Band-map equivalente a sentinel2-8: {band_map_equal}")
    print(f"Smoke PSNR: {smoke_summary['global_quality']['psnr_db']:.4f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
