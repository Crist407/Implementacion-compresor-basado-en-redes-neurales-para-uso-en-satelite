#!/usr/bin/env python3
"""
Barrido empirico Q -> calidad para SORTENY C.

Este script no modifica el formato de bitstream. Para cada valor Q genera un
Q-map constante, ejecuta compresion/descompresion C y analiza la calidad por
bloques con analyze_block_quality.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_Q_VALUES = [160, 176, 192, 204, 216, 232]


def q_u8(value: str) -> int:
    q = int(value)
    if q < 0 or q > 255:
        raise argparse.ArgumentTypeError(f"Q debe estar en [0,255], recibido {q}")
    return q


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def run_cmd(cmd: list[str], cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    dt = time.perf_counter() - t0
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.write(f"# returncode={proc.returncode} elapsed_s={dt:.6f}\n")
        f.write("\n[stdout]\n")
        f.write(proc.stdout)
        f.write("\n[stderr]\n")
        f.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Comando fallido ({proc.returncode}): {' '.join(cmd)}; log={log_path}")
    return dt


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "q",
        "lambda_quant",
        "compress_s",
        "decompress_s",
        "analyze_s",
        "mse",
        "psnr_db",
        "mae",
        "max_abs",
        "exact_pct",
        "mean_block_mse",
        "worst_block_mse",
        "worst_block_y",
        "worst_block_x",
        "latent_size_bytes",
        "reconstructed_size_bytes",
        "latent_matches_baseline",
        "reconstructed_matches_baseline",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def files_equal(a: Path, b: Path) -> bool | None:
    if not a.exists() or not b.exists():
        return None
    if a.stat().st_size != b.stat().st_size:
        return False
    chunk = 1024 * 1024
    with a.open("rb") as fa, b.open("rb") as fb:
        while True:
            ca = fa.read(chunk)
            cb = fb.read(chunk)
            if ca != cb:
                return False
            if not ca:
                return True


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(description="Barrido Q -> calidad para SORTENY C.")
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--input", type=Path, default=root / "data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw")
    parser.add_argument("--output-dir", type=Path, default=root / "output/checkpoints/20260506_q_sweep_calibration")
    parser.add_argument("--q-values", type=q_u8, nargs="+", default=DEFAULT_Q_VALUES)
    parser.add_argument("--lambda", dest="lambda_value", type=float, default=0.1)
    parser.add_argument("--max-lambda", type=float, default=0.125)
    parser.add_argument("--encoder-weights", type=Path, default=root / "weights/encoder")
    parser.add_argument("--decoder-weights", type=Path, default=root / "weights/decoder")
    parser.add_argument("--encoder", type=Path, default=root / "sorteny_compressor")
    parser.add_argument("--decoder", type=Path, default=root / "sorteny_decompressor")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--baseline-latent", type=Path, default=root / "output/checkpoints/20260506_baseline_constant_qmap/latent.bin")
    parser.add_argument("--baseline-reconstructed", type=Path, default=root / "output/checkpoints/20260506_baseline_constant_qmap/reconstructed.raw")
    args = parser.parse_args()
    if args.max_lambda <= 0.0:
        parser.error("--max-lambda debe ser positivo")
    if args.threads <= 0:
        parser.error("--threads debe ser positivo")
    return args


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_qmap = repo_root / "src/python/utils/generate_qmap.py"
    analyze_blocks = repo_root / "src/python/analysis/analyze_block_quality.py"
    for required in [args.input, args.encoder_weights, args.decoder_weights, args.encoder, args.decoder, generate_qmap, analyze_blocks]:
        if not required.exists():
            raise FileNotFoundError(required)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)

    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    print(f"Barrido Q: {args.q_values}")
    print(f"Salida: {output_dir}")

    for q in args.q_values:
        q_dir = output_dir / f"q_{q:03d}"
        q_dir.mkdir(parents=True, exist_ok=True)
        qmap = q_dir / "qmap.bin"
        latent = q_dir / "latent.bin"
        reconstructed = q_dir / "reconstructed.raw"
        block_json = q_dir / "block_quality.json"
        block_csv = q_dir / "block_quality.csv"

        lambda_quant = (float(q) / 255.0) * args.max_lambda
        print(f"\n== Q={q} lambda={lambda_quant:.8f} ==")

        t_generate = run_cmd(
            [
                sys.executable,
                str(generate_qmap),
                str(qmap),
                "--pattern",
                "constant",
                "--q",
                str(q),
                "--summary-json",
                str(q_dir / "qmap_summary.json"),
            ],
            cwd=repo_root,
            log_path=q_dir / "generate_qmap.log",
        )

        t_compress = run_cmd(
            [
                str(args.encoder),
                str(args.input),
                str(args.lambda_value),
                str(latent),
                str(args.encoder_weights),
                str(args.max_lambda),
                str(qmap),
            ],
            cwd=repo_root,
            log_path=q_dir / "compress.log",
            env=env,
        )

        t_decompress = run_cmd(
            [
                str(args.decoder),
                str(latent),
                str(reconstructed),
                str(args.decoder_weights),
                str(args.max_lambda),
            ],
            cwd=repo_root,
            log_path=q_dir / "decompress.log",
            env=env,
        )

        t_analyze = run_cmd(
            [
                sys.executable,
                str(analyze_blocks),
                str(args.input),
                str(reconstructed),
                "--bitstream",
                str(latent),
                "--block-size",
                str(args.block_size),
                "--output-json",
                str(block_json),
                "--output-csv",
                str(block_csv),
            ],
            cwd=repo_root,
            log_path=q_dir / "analyze_block_quality.log",
        )

        quality = load_json(block_json)
        worst = quality["worst_blocks"][0] if quality["worst_blocks"] else {}
        row = {
            "q": q,
            "lambda_quant": lambda_quant,
            "generate_s": t_generate,
            "compress_s": t_compress,
            "decompress_s": t_decompress,
            "analyze_s": t_analyze,
            "mse": quality["global"]["mse"],
            "psnr_db": quality["global"]["psnr_db"],
            "mae": quality["global"]["mae"],
            "max_abs": quality["global"]["max_abs"],
            "exact_pct": quality["global"]["exact_pct"],
            "mean_block_mse": quality["by_q"][str(q)]["mean_mse"],
            "worst_block_mse": worst.get("mse"),
            "worst_block_y": worst.get("block_y"),
            "worst_block_x": worst.get("block_x"),
            "latent_size_bytes": latent.stat().st_size,
            "reconstructed_size_bytes": reconstructed.stat().st_size,
            "latent_matches_baseline": files_equal(latent, args.baseline_latent) if q == 204 else None,
            "reconstructed_matches_baseline": files_equal(reconstructed, args.baseline_reconstructed) if q == 204 else None,
        }
        rows.append(row)
        results.append(
            {
                "q": q,
                "lambda_quant": lambda_quant,
                "directory": str(q_dir),
                "timings": {
                    "generate_qmap_s": t_generate,
                    "compress_s": t_compress,
                    "decompress_s": t_decompress,
                    "analyze_s": t_analyze,
                },
                "global": quality["global"],
                "by_q": quality["by_q"].get(str(q)),
                "worst_block": worst,
                "regression": {
                    "latent_matches_baseline": row["latent_matches_baseline"],
                    "reconstructed_matches_baseline": row["reconstructed_matches_baseline"],
                },
            }
        )
        print(f"MSE={row['mse']:.4f} PSNR={row['psnr_db']:.4f} dB MAE={row['mae']:.4f}")

    q_sorted = sorted(rows, key=lambda r: r["q"])
    mse_monotonic_nonincreasing = all(
        q_sorted[i]["mse"] >= q_sorted[i + 1]["mse"] for i in range(len(q_sorted) - 1)
    )
    psnr_monotonic_nondecreasing = all(
        q_sorted[i]["psnr_db"] <= q_sorted[i + 1]["psnr_db"] for i in range(len(q_sorted) - 1)
    )
    monotonic = {
        "mse_nonincreasing_with_q": mse_monotonic_nonincreasing,
        "psnr_nondecreasing_with_q": psnr_monotonic_nondecreasing,
    }

    summary = {
        "config": {
            "repo_root": str(repo_root),
            "input": str(args.input),
            "output_dir": str(output_dir),
            "q_values": args.q_values,
            "lambda_cli": args.lambda_value,
            "max_lambda": args.max_lambda,
            "threads": args.threads,
            "block_size": args.block_size,
            "note": "Bitstream sin codificador entropico: cabecera + Q-map + latentes int32.",
        },
        "monotonicity": monotonic,
        "results": results,
    }
    (output_dir / "sweep_results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output_dir / "sweep_results.csv", rows)

    print("\nResumen:")
    for row in rows:
        print(f"Q={row['q']:3d} lambda={row['lambda_quant']:.8f} MSE={row['mse']:.4f} PSNR={row['psnr_db']:.4f} dB")
    print("Monotonia:", monotonic)
    if any(row["q"] == 204 and row["latent_matches_baseline"] is False for row in rows):
        print("WARN: Q=204 no coincide byte a byte con baseline latent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
