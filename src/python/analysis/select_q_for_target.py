#!/usr/bin/env python3
"""
Selecciona un Q global a partir de un barrido empirico Q -> calidad.

La salida principal es un Q-map constante uint8. Este paso es un puente de
calibracion y analisis: no forma parte de la ruta final de ejecucion en
Raspberry, donde la decision de Q-map debe hacerse en C.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def q_u8(value: int) -> int:
    value = int(value)
    if value < 0 or value > 255:
        raise argparse.ArgumentTypeError(f"Q debe estar en [0,255], recibido {value}")
    return value


def load_points(path: Path) -> tuple[list[dict[str, float]], float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    max_lambda = float(data["config"]["max_lambda"])
    points = []
    for row in data["results"]:
        points.append(
            {
                "q": int(row["q"]),
                "lambda_quant": float(row["lambda_quant"]),
                "mse": float(row["global"]["mse"]),
                "psnr_db": float(row["global"]["psnr_db"]),
                "mae": float(row["global"]["mae"]),
            }
        )
    points.sort(key=lambda p: p["q"])
    if not points:
        raise ValueError(f"{path}: no contiene puntos de barrido")
    return points, max_lambda


def interpolate_q(points: list[dict[str, float]], metric: str, target: float) -> tuple[int, dict[str, Any]]:
    values = [p[metric] for p in points]
    increasing = values[-1] >= values[0]

    if increasing:
        lower_bound = values[0]
        upper_bound = values[-1]
        below_range = target < lower_bound
        above_range = target > upper_bound
    else:
        lower_bound = values[-1]
        upper_bound = values[0]
        below_range = target < lower_bound
        above_range = target > upper_bound

    if below_range:
        q = points[0]["q"] if increasing else points[-1]["q"]
        return int(q), {"mode": "clamped", "reason": "target_below_sweep_range"}
    if above_range:
        q = points[-1]["q"] if increasing else points[0]["q"]
        return int(q), {"mode": "clamped", "reason": "target_above_sweep_range"}

    for a, b in zip(points, points[1:]):
        va = a[metric]
        vb = b[metric]
        lo = min(va, vb)
        hi = max(va, vb)
        if lo <= target <= hi:
            if va == vb:
                q_float = float(a["q"])
            else:
                t = (target - va) / (vb - va)
                q_float = float(a["q"]) + t * (float(b["q"]) - float(a["q"]))
            q = max(0, min(255, int(round(q_float))))
            return q, {
                "mode": "interpolated",
                "between": [int(a["q"]), int(b["q"])],
                "q_float": q_float,
                "metric_a": va,
                "metric_b": vb,
            }

    nearest = min(points, key=lambda p: abs(p[metric] - target))
    return int(nearest["q"]), {"mode": "fallback_nearest"}


def nearest_q(points: list[dict[str, float]], metric: str, target: float) -> tuple[int, dict[str, Any]]:
    nearest = min(points, key=lambda p: abs(p[metric] - target))
    return int(nearest["q"]), {"mode": "nearest", "nearest_metric": nearest[metric]}


def estimate_from_neighbors(points: list[dict[str, float]], q: int) -> dict[str, Any]:
    exact = [p for p in points if int(p["q"]) == q]
    if exact:
        p = exact[0]
        return {
            "source": "measured",
            "mse": p["mse"],
            "psnr_db": p["psnr_db"],
            "mae": p["mae"],
        }

    for a, b in zip(points, points[1:]):
        qa = int(a["q"])
        qb = int(b["q"])
        if qa <= q <= qb:
            t = (q - qa) / (qb - qa)
            return {
                "source": "linear_interpolation",
                "between": [qa, qb],
                "mse": a["mse"] + t * (b["mse"] - a["mse"]),
                "psnr_db": a["psnr_db"] + t * (b["psnr_db"] - a["psnr_db"]),
                "mae": a["mae"] + t * (b["mae"] - a["mae"]),
            }

    nearest = min(points, key=lambda p: abs(int(p["q"]) - q))
    return {
        "source": "nearest_outside_range",
        "nearest_q": int(nearest["q"]),
        "mse": nearest["mse"],
        "psnr_db": nearest["psnr_db"],
        "mae": nearest["mae"],
    }


def write_constant_qmap(path: Path, q: int, width: int, height: int) -> bytes:
    qmap = bytes([q]) * (width * height)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(qmap)
    return qmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selecciona Q global desde un barrido Q->calidad.")
    parser.add_argument("--sweep", type=Path, required=True, help="sweep_results.json.")
    parser.add_argument("--target-psnr", type=float, default=None)
    parser.add_argument("--target-mse", type=float, default=None)
    parser.add_argument("--method", choices=["interpolate", "nearest"], default="interpolate")
    parser.add_argument("--output-qmap", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    args = parser.parse_args()
    if (args.target_psnr is None) == (args.target_mse is None):
        parser.error("usa exactamente uno de --target-psnr o --target-mse")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width y --height deben ser positivos")
    return args


def main() -> int:
    args = parse_args()
    points, max_lambda = load_points(args.sweep)

    if args.target_psnr is not None:
        metric = "psnr_db"
        target = float(args.target_psnr)
    else:
        metric = "mse"
        target = float(args.target_mse)

    if args.method == "nearest":
        q, selection = nearest_q(points, metric, target)
    else:
        q, selection = interpolate_q(points, metric, target)

    lambda_quant = (float(q) / 255.0) * max_lambda
    estimate = estimate_from_neighbors(points, q)
    qmap = write_constant_qmap(args.output_qmap, q, args.width, args.height)

    summary = {
        "sweep": str(args.sweep),
        "target": {
            "metric": metric,
            "value": target,
        },
        "method": args.method,
        "selected": {
            "q": q,
            "lambda_quant": lambda_quant,
            "estimate": estimate,
            "selection_detail": selection,
        },
        "qmap": {
            "output": str(args.output_qmap),
            "width": args.width,
            "height": args.height,
            "size_bytes": len(qmap),
            "sha256": hashlib.sha256(qmap).hexdigest(),
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
