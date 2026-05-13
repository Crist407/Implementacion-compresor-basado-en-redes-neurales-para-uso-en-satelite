#!/usr/bin/env python3
"""
Genera mapas ROI manuales para sorteny_semantic_qmap.

El mapa de salida es uint8 a resolucion de Q-map. Para imagenes 512x512 con
bloques 16x16, la salida por defecto es 32x32 = 1024 bytes. Cualquier valor
distinto de cero se interpreta como bloque ROI.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def load_tsv(path: Path, height: int, width: int) -> np.ndarray:
    roi = np.zeros((height, width), dtype=np.uint8)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            by = int(row["block_y"])
            bx = int(row["block_x"])
            if by < 0 or by >= height or bx < 0 or bx >= width:
                raise ValueError(f"{path}: bloque fuera de rango ({by}, {bx})")
            roi[by, bx] = 1
    return roi


def apply_rect(roi: np.ndarray, rect: list[int]) -> None:
    y0, x0, y1, x1 = rect
    height, width = roi.shape
    if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
        raise ValueError(f"rectangulo fuera de rango: {rect}, grid={height}x{width}")
    roi[y0:y1, x0:x1] = 1


def generate(args: argparse.Namespace) -> np.ndarray:
    if args.from_tsv:
        roi = load_tsv(args.from_tsv, args.height, args.width)
    else:
        roi = np.zeros((args.height, args.width), dtype=np.uint8)

    if args.pattern == "full":
        roi[:, :] = 1
    elif args.pattern == "center":
        y0 = args.height // 4
        y1 = args.height - y0
        x0 = args.width // 4
        x1 = args.width - x0
        roi[y0:y1, x0:x1] = 1
    elif args.pattern == "checker":
        yy, xx = np.indices((args.height, args.width))
        roi[((yy + xx) % 2) == 0] = 1
    elif args.pattern != "empty":
        raise ValueError(f"patron desconocido: {args.pattern}")

    for rect in args.rect:
        apply_rect(roi, rect)

    return roi


def write_tsv(path: Path, roi: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["block_y", "block_x"], delimiter="\t")
        writer.writeheader()
        for by, bx in np.argwhere(roi != 0):
            writer.writerow({"block_y": int(by), "block_x": int(bx)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera ROI-map uint8 para seleccion manual de bloques.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--pattern", choices=["empty", "full", "center", "checker"], default="empty")
    parser.add_argument(
        "--rect",
        type=int,
        nargs=4,
        action="append",
        default=[],
        metavar=("Y0", "X0", "Y1", "X1"),
        help="Rectangulo half-open en coordenadas de bloque. Se puede repetir.",
    )
    parser.add_argument("--from-tsv", type=Path, default=None)
    parser.add_argument("--output-tsv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("--width y --height deben ser positivos")
    return args


def main() -> int:
    args = parse_args()
    roi = generate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = roi.reshape(-1).tobytes()
    args.output.write_bytes(payload)

    if args.output_tsv:
        write_tsv(args.output_tsv, roi)

    summary = {
        "output": str(args.output),
        "width": args.width,
        "height": args.height,
        "size_bytes": len(payload),
        "roi_blocks": int(np.count_nonzero(roi)),
        "background_blocks": int(roi.size - np.count_nonzero(roi)),
        "pattern": args.pattern,
        "rectangles": args.rect,
        "from_tsv": str(args.from_tsv) if args.from_tsv else None,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(json_ready(summary), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
