#!/usr/bin/env python3
"""
Genera Q-maps uint8 reproducibles para pruebas de Fase 2.

El bitstream C actual espera un mapa de calidad a resolución latente. Para la
imagen canónica de 512x512 y stride total 16, el mapa es de 32x32 = 1024 bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def q_u8(value: int) -> int:
    value = int(value)
    if value < 0 or value > 255:
        raise argparse.ArgumentTypeError(f"Q debe estar en [0, 255], recibido {value}")
    return value


def generate_qmap(args: argparse.Namespace) -> bytearray:
    width = args.width
    height = args.height
    qmap = bytearray(width * height)

    if args.pattern == "constant":
        for i in range(width * height):
            qmap[i] = args.q
        return qmap

    if args.pattern == "horizontal-split":
        split = args.split if args.split is not None else height // 2
        if split < 0 or split > height:
            raise ValueError(f"--split debe estar entre 0 y {height}")
        for y in range(height):
            q = args.q_low if y < split else args.q_high
            for x in range(width):
                qmap[y * width + x] = q
        return qmap

    if args.pattern == "vertical-split":
        split = args.split if args.split is not None else width // 2
        if split < 0 or split > width:
            raise ValueError(f"--split debe estar entre 0 y {width}")
        for y in range(height):
            for x in range(width):
                qmap[y * width + x] = args.q_low if x < split else args.q_high
        return qmap

    if args.pattern == "checkerboard":
        tile = max(1, args.tile)
        for y in range(height):
            for x in range(width):
                qmap[y * width + x] = args.q_low if ((x // tile) + (y // tile)) % 2 == 0 else args.q_high
        return qmap

    if args.pattern in {"gradient-x", "gradient-y"}:
        denom = (width - 1) if args.pattern == "gradient-x" else (height - 1)
        if denom <= 0:
            raise ValueError(f"{args.pattern} requiere dimension mayor que 1")
        for y in range(height):
            for x in range(width):
                t = (x / denom) if args.pattern == "gradient-x" else (y / denom)
                q = round(args.q_low + t * (args.q_high - args.q_low))
                qmap[y * width + x] = q_u8(q)
        return qmap

    raise ValueError(f"Patrón no soportado: {args.pattern}")


def summarize(qmap: bytes, args: argparse.Namespace, output: Path) -> dict:
    unique = sorted(set(qmap))
    return {
        "output": str(output),
        "width": args.width,
        "height": args.height,
        "size_bytes": len(qmap),
        "pattern": args.pattern,
        "q": args.q,
        "q_low": args.q_low,
        "q_high": args.q_high,
        "split": args.split,
        "tile": args.tile,
        "unique_count": len(unique),
        "unique_values": unique,
        "sha256": hashlib.sha256(qmap).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera Q-maps uint8 para SORTENY C.")
    parser.add_argument("output", type=Path, help="Ruta del Q-map binario de salida.")
    parser.add_argument("--width", type=int, default=32, help="Anchura del Q-map latente. Default: 32.")
    parser.add_argument("--height", type=int, default=32, help="Altura del Q-map latente. Default: 32.")
    parser.add_argument(
        "--pattern",
        choices=["constant", "horizontal-split", "vertical-split", "checkerboard", "gradient-x", "gradient-y"],
        default="constant",
        help="Patrón a generar. Default: constant.",
    )
    parser.add_argument("--q", type=q_u8, default=204, help="Valor Q para patrón constant. Default: 204.")
    parser.add_argument("--q-low", type=q_u8, default=180, help="Valor bajo para patrones variables. Default: 180.")
    parser.add_argument("--q-high", type=q_u8, default=204, help="Valor alto para patrones variables. Default: 204.")
    parser.add_argument("--split", type=int, default=None, help="Fila/columna de corte para split. Default: mitad.")
    parser.add_argument("--tile", type=int, default=1, help="Tamaño de baldosa para checkerboard. Default: 1.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Ruta opcional para guardar resumen JSON.")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("--width y --height deben ser positivos")
    if args.tile <= 0:
        parser.error("--tile debe ser positivo")
    return args


def main() -> int:
    args = parse_args()
    qmap = generate_qmap(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(qmap)

    summary = summarize(qmap, args, args.output)
    text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
