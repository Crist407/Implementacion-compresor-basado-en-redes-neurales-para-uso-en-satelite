#!/usr/bin/env python3
"""
Genera una calibracion TSV para sorteny_fq_qmap.

Este script es auxiliar: usa los barridos Q ya ejecutados para ajustar por
bloque un modelo simple MSE ~= c0 + c1 / M(lambda)^2. La decision final de Q
en la ruta Raspberry se hace en C leyendo el TSV resultante.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


MOD_LAMBDA_SCALE = 0.05
MOD_HIDDEN = 192
MOD_OUT = 3072


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_block_csv(path: Path) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by = int(row["block_y"])
            bx = int(row["block_x"])
            out[(by, bx)] = float(row["mse"])
    return out


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def modulator_mean(weights_dir: Path, lambda_value: float) -> float:
    k1 = np.fromfile(weights_dir / "mod_dense_1_kernel.bin", dtype=np.float32).reshape(1, MOD_HIDDEN)
    b1 = np.fromfile(weights_dir / "mod_dense_1_bias.bin", dtype=np.float32).reshape(MOD_HIDDEN)
    k2 = np.fromfile(weights_dir / "mod_dense_2_kernel.bin", dtype=np.float32).reshape(MOD_HIDDEN, MOD_OUT)
    b2 = np.fromfile(weights_dir / "mod_dense_2_bias.bin", dtype=np.float32).reshape(MOD_OUT)

    x = np.array([lambda_value / MOD_LAMBDA_SCALE], dtype=np.float32)
    hidden = relu(x @ k1 + b1)
    out = relu(hidden @ k2 + b2)
    return float(np.mean(out))


def fit_line(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    if xs.size != ys.size or xs.size < 2:
        raise ValueError("se necesitan al menos dos puntos para ajustar")
    x_mean = float(np.mean(xs))
    y_mean = float(np.mean(ys))
    denom = float(np.sum((xs - x_mean) ** 2))
    if denom == 0.0:
        raise ValueError("xs sin varianza")
    slope = float(np.sum((xs - x_mean) * (ys - y_mean)) / denom)
    intercept = y_mean - slope * x_mean
    pred = intercept + slope * xs
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - y_mean) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - (ss_res / ss_tot)
    return intercept, slope, r2


def psnr_from_mse(mse: float) -> float:
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((65535.0 * 65535.0) / mse)


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(description="Construye calibracion TSV para sorteny_fq_qmap.")
    parser.add_argument(
        "--sweep",
        type=Path,
        default=root / "output/checkpoints/20260506_q_sweep_calibration/sweep_results.json",
        help="sweep_results.json generado por sweep_q_quality.py.",
    )
    parser.add_argument("--weights", type=Path, default=root / "weights/encoder")
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--q-baseline", type=int, default=204)
    args = parser.parse_args()
    if args.q_baseline < 0 or args.q_baseline > 255:
        parser.error("--q-baseline debe estar en [0,255]")
    return args


def main() -> int:
    args = parse_args()
    sweep = load_json(args.sweep)
    max_lambda = float(sweep["config"]["max_lambda"])
    results = sorted(sweep["results"], key=lambda r: int(r["q"]))
    if len(results) < 2:
        raise ValueError("el barrido debe contener al menos dos valores Q")

    q_values = [int(r["q"]) for r in results]
    lambda_values = np.array([float(r["lambda_quant"]) for r in results], dtype=np.float64)
    mod_values = np.array([modulator_mean(args.weights, lam) for lam in lambda_values], dtype=np.float64)
    mod_b, mod_a, mod_r2 = fit_line(lambda_values, mod_values)
    inv_mod2 = 1.0 / np.maximum(mod_values * mod_values, 1e-12)

    per_q_blocks: dict[int, dict[tuple[int, int], float]] = {}
    for r in results:
        q = int(r["q"])
        q_dir = Path(r["directory"])
        block_csv = q_dir / "block_quality.csv"
        if not block_csv.exists():
            raise FileNotFoundError(block_csv)
        per_q_blocks[q] = load_block_csv(block_csv)

    block_keys = sorted(per_q_blocks[q_values[0]].keys())
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    valid_count = 0
    with args.output_tsv.open("w", newline="", encoding="utf-8") as f:
        f.write("# sorteny_fq_calibration_tsv_v1\n")
        f.write("# model\tmse=c0+c1/(mod_a*lambda+mod_b)^2\n")
        f.write(f"# sweep\t{args.sweep}\n")
        f.write(f"# weights\t{args.weights}\n")
        f.write(f"# max_lambda\t{max_lambda:.9g}\n")
        f.write(f"# q_min\t{min(q_values)}\n")
        f.write(f"# q_max\t{max(q_values)}\n")
        f.write(f"# q_baseline\t{args.q_baseline}\n")
        f.write(f"# mod_a\t{mod_a:.9g}\n")
        f.write(f"# mod_b\t{mod_b:.9g}\n")
        f.write(f"# mod_r2\t{mod_r2:.9g}\n")

        fields = [
            "block_y",
            "block_x",
            "c0",
            "c1",
            "r2",
            "valid",
            "q_baseline",
            "q_min",
            "q_max",
            "max_lambda",
            "mod_a",
            "mod_b",
            "mse_at_baseline",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()

        for by, bx in block_keys:
            ys = np.array([per_q_blocks[q][(by, bx)] for q in q_values], dtype=np.float64)
            c0, c1, r2 = fit_line(inv_mod2, ys)
            valid = int(math.isfinite(c0) and math.isfinite(c1) and c1 > 0.0 and mod_a > 0.0)
            valid_count += valid
            baseline_m = mod_a * ((args.q_baseline / 255.0) * max_lambda) + mod_b
            mse_at_baseline = c0 + (c1 / max(baseline_m * baseline_m, 1e-12))
            row = {
                "block_y": by,
                "block_x": bx,
                "c0": c0,
                "c1": c1,
                "r2": r2,
                "valid": valid,
                "q_baseline": args.q_baseline,
                "q_min": min(q_values),
                "q_max": max(q_values),
                "max_lambda": max_lambda,
                "mod_a": mod_a,
                "mod_b": mod_b,
                "mse_at_baseline": mse_at_baseline,
            }
            writer.writerow(row)
            rows.append(row)

    summary = {
        "calibration": str(args.output_tsv),
        "sweep": str(args.sweep),
        "weights": str(args.weights),
        "q_values": q_values,
        "max_lambda": max_lambda,
        "modulator_linear_fit": {
            "mod_a": mod_a,
            "mod_b": mod_b,
            "r2": mod_r2,
        },
        "blocks": len(rows),
        "valid_blocks": valid_count,
        "invalid_blocks": len(rows) - valid_count,
        "baseline": {
            "q": args.q_baseline,
            "mean_predicted_mse": float(np.mean([r["mse_at_baseline"] for r in rows])),
            "mean_predicted_psnr_db": psnr_from_mse(float(np.mean([r["mse_at_baseline"] for r in rows]))),
        },
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
