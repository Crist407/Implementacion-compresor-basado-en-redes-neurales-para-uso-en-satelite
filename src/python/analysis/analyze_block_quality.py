#!/usr/bin/env python3
"""
Analiza calidad local por bloques para reconstrucciones SORTENY C.

La resolucion de bloque por defecto es 16x16 pixeles, que corresponde a un
pixel del Q-map latente para imagenes 512x512 con stride total 16.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np


MAX_U16 = 65535.0


def load_raw_u16(path: Path, bands: int, height: int, width: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint16)
    expected = bands * height * width
    if data.size != expected:
        raise ValueError(f"{path}: tamano inesperado {data.size}, esperado {expected}")
    return data.reshape(bands, height, width)


def psnr_from_mse(mse: float) -> float:
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((MAX_U16 * MAX_U16) / mse)


def metrics_for_arrays(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(abs_diff))
    max_abs = float(np.max(abs_diff))
    exact_pct = float(np.mean(abs_diff == 0.0) * 100.0)
    return {
        "mse": mse,
        "psnr_db": psnr_from_mse(mse),
        "mae": mae,
        "max_abs": max_abs,
        "exact_pct": exact_pct,
    }


def load_qmap_from_file(path: Path, q_height: int, q_width: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    expected = q_height * q_width
    if data.size != expected:
        raise ValueError(f"{path}: Q-map tiene {data.size} bytes, esperado {expected}")
    return data.reshape(q_height, q_width)


def load_qmap_from_bitstream(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    with path.open("rb") as f:
        header_bytes = f.read(10)
        if len(header_bytes) != 10:
            raise ValueError(f"{path}: cabecera incompleta")
        bands, height, width, datatype, num_filters = struct.unpack("<5H", header_bytes)
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"{path}: dimensiones no divisibles por 16: {height}x{width}")
        q_height = height // 16
        q_width = width // 16
        q_size = q_height * q_width
        q_raw = f.read(q_size)
        if len(q_raw) != q_size:
            raise ValueError(f"{path}: Q-map incompleto")
    qmap = np.frombuffer(q_raw, dtype=np.uint8).copy().reshape(q_height, q_width)
    header = {
        "bands": int(bands),
        "height": int(height),
        "width": int(width),
        "datatype": int(datatype),
        "num_filters": int(num_filters),
    }
    return qmap, header


def finite_float(value: float) -> float | str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return float(value)


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}
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


def analyze_blocks(
    original: np.ndarray,
    reconstructed: np.ndarray,
    block_size: int,
    qmap: np.ndarray | None,
) -> list[dict[str, Any]]:
    bands, height, width = original.shape
    block_h = height // block_size
    block_w = width // block_size
    rows: list[dict[str, Any]] = []

    for by in range(block_h):
        y0 = by * block_size
        y1 = y0 + block_size
        for bx in range(block_w):
            x0 = bx * block_size
            x1 = x0 + block_size
            m = metrics_for_arrays(original[:, y0:y1, x0:x1], reconstructed[:, y0:y1, x0:x1])
            row: dict[str, Any] = {
                "block_y": by,
                "block_x": bx,
                "y0": y0,
                "x0": x0,
                "height": block_size,
                "width": block_size,
                "samples": bands * block_size * block_size,
                **m,
            }
            if qmap is not None:
                row["q"] = int(qmap[by, bx])
            rows.append(row)
    return rows


def group_by_q(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if "q" in row:
            grouped.setdefault(int(row["q"]), []).append(row)

    out: dict[str, dict[str, Any]] = {}
    for q, items in sorted(grouped.items()):
        mse = float(np.mean([r["mse"] for r in items]))
        mae = float(np.mean([r["mae"] for r in items]))
        exact_pct = float(np.mean([r["exact_pct"] for r in items]))
        max_abs = float(max(r["max_abs"] for r in items))
        out[str(q)] = {
            "blocks": len(items),
            "mean_mse": mse,
            "psnr_from_mean_mse_db": psnr_from_mse(mse),
            "mean_mae": mae,
            "max_abs": max_abs,
            "mean_exact_pct": exact_pct,
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "block_y",
        "block_x",
        "q",
        "mse",
        "psnr_db",
        "mae",
        "max_abs",
        "exact_pct",
        "y0",
        "x0",
        "height",
        "width",
        "samples",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analiza MSE/PSNR por bloques de una reconstruccion SORTENY.")
    parser.add_argument("original", type=Path, help="Imagen original RAW BSQ uint16.")
    parser.add_argument("reconstructed", type=Path, help="Reconstruccion RAW BSQ uint16.")
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--qmap", type=Path, default=None, help="Q-map uint8 crudo.")
    parser.add_argument("--bitstream", type=Path, default=None, help="Bitstream SORTENY del que extraer el Q-map.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10, help="Numero de peores bloques a reportar.")
    args = parser.parse_args()
    if args.block_size <= 0:
        parser.error("--block-size debe ser positivo")
    if args.height % args.block_size != 0 or args.width % args.block_size != 0:
        parser.error("--height y --width deben ser divisibles por --block-size")
    if args.qmap and args.bitstream:
        parser.error("usa --qmap o --bitstream, no ambos")
    return args


def main() -> int:
    args = parse_args()
    original = load_raw_u16(args.original, args.bands, args.height, args.width)
    reconstructed = load_raw_u16(args.reconstructed, args.bands, args.height, args.width)

    qmap = None
    bitstream_header = None
    q_height = args.height // args.block_size
    q_width = args.width // args.block_size
    if args.bitstream:
        qmap, bitstream_header = load_qmap_from_bitstream(args.bitstream)
    elif args.qmap:
        qmap = load_qmap_from_file(args.qmap, q_height, q_width)

    if qmap is not None and qmap.shape != (q_height, q_width):
        raise ValueError(f"Q-map shape {qmap.shape} no coincide con bloques {(q_height, q_width)}")

    global_metrics = metrics_for_arrays(original, reconstructed)
    rows = analyze_blocks(original, reconstructed, args.block_size, qmap)
    worst = sorted(rows, key=lambda r: r["mse"], reverse=True)[: args.top_k]

    result = {
        "config": {
            "original": str(args.original),
            "reconstructed": str(args.reconstructed),
            "bitstream": str(args.bitstream) if args.bitstream else None,
            "qmap": str(args.qmap) if args.qmap else None,
            "bands": args.bands,
            "height": args.height,
            "width": args.width,
            "block_size": args.block_size,
            "block_grid": [q_height, q_width],
        },
        "bitstream_header": bitstream_header,
        "global": global_metrics,
        "qmap": None
        if qmap is None
        else {
            "unique_count": int(len(set(int(x) for x in qmap.reshape(-1)))),
            "unique_values": sorted(int(x) for x in set(qmap.reshape(-1))),
        },
        "by_q": group_by_q(rows),
        "worst_blocks": worst,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(json_ready(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_csv:
        write_csv(args.output_csv, rows)

    print("Global:")
    print(f"  MSE={global_metrics['mse']:.4f}")
    print(f"  PSNR={global_metrics['psnr_db']:.4f} dB")
    print(f"  MAE={global_metrics['mae']:.4f}")
    print(f"  MaxAbs={global_metrics['max_abs']:.0f}")
    if qmap is not None:
        print("By Q:")
        for q, values in result["by_q"].items():
            print(
                f"  Q={q}: blocks={values['blocks']} "
                f"mean_mse={values['mean_mse']:.4f} "
                f"psnr={values['psnr_from_mean_mse_db']:.4f} dB "
                f"mean_mae={values['mean_mae']:.4f}"
            )
    print("Worst blocks:")
    for row in worst[: min(args.top_k, len(worst))]:
        q_text = f" Q={row['q']}" if "q" in row else ""
        print(f"  ({row['block_y']},{row['block_x']}){q_text}: mse={row['mse']:.4f} psnr={row['psnr_db']:.4f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
