#!/usr/bin/env python3
"""
Genera un paquete pre-web para seleccionar y validar una ROI manual.

La salida principal es una mascara ROI 32x32 compatible con
`sorteny_semantic_qmap --preset manual`. La herramienta tambien exporta un
quicklook RGB con rejilla, una superposicion de ROI y comandos reproducibles
para continuar con el pipeline C.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_INPUT = Path("data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw")
DEFAULT_OUTPUT_DIR = Path("output/checkpoints/20260523_manual_roi_preweb")
DEFAULT_CALIBRATION_WIDE = Path("output/checkpoints/20260507_c_fixed_quality_qmap_wide/fq_calibration.tsv")
DEFAULT_CALIBRATION = Path("output/checkpoints/20260507_c_fixed_quality_qmap/fq_calibration.tsv")

PROFILE_ARGS = {
    "conservative": ["--semantic-policy", "focus", "--foreground-boost", "8", "--background-penalty", "24"],
    "balanced": ["--semantic-policy", "focus", "--foreground-boost", "16", "--background-penalty", "48"],
    "aggressive": ["--semantic-policy", "focus", "--foreground-boost", "16", "--background-q", "128"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="RAW BSQ uint16 de 8 bandas.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--pattern", choices=["empty", "center", "full", "checker"], default="center")
    parser.add_argument(
        "--rect",
        type=int,
        nargs=4,
        action="append",
        default=[],
        metavar=("Y0", "X0", "Y1", "X1"),
        help="Rectangulo half-open en bloques. Se puede repetir.",
    )
    parser.add_argument("--from-tsv", type=Path, default=None, help="TSV con columnas block_y,block_x.")
    parser.add_argument("--profile", choices=sorted(PROFILE_ARGS), default="aggressive")
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--semantic-bin", type=Path, default=Path("./sorteny_semantic_qmap"))
    parser.add_argument("--skip-qmap", action="store_true", help="Solo genera quicklook/ROI, sin llamar a C.")
    args = parser.parse_args()

    if args.bands != 8:
        parser.error("esta utilidad visual asume Sentinel-2 de 8 bandas para RGB B04,B03,B02")
    if args.height % args.block_size != 0 or args.width % args.block_size != 0:
        parser.error("--height y --width deben ser divisibles por --block-size")
    return args


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
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_raw(path: Path, bands: int, height: int, width: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint16)
    expected = bands * height * width
    if data.size != expected:
        raise ValueError(f"{path}: esperado {expected} uint16, obtenido {data.size}")
    return data.reshape(bands, height, width)


def stretch_u16_to_u8(band: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(band.astype(np.float64), [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1.0
    scaled = (band.astype(np.float64) - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def make_rgb(raw: np.ndarray) -> Image.Image:
    # Sentinel2-8: B02,B03,B04,B05,B06,B07,B08,B8A -> RGB = B04,B03,B02
    rgb = np.stack([stretch_u16_to_u8(raw[2]), stretch_u16_to_u8(raw[1]), stretch_u16_to_u8(raw[0])], axis=-1)
    return Image.fromarray(rgb)


def draw_grid(img: Image.Image, block_size: int, color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    for x in range(0, width + 1, block_size):
        draw.line([(x, 0), (x, height)], fill=color, width=1)
    for y in range(0, height + 1, block_size):
        draw.line([(0, y), (width, y)], fill=color, width=1)
    return out


def load_tsv(path: Path, grid_h: int, grid_w: int) -> np.ndarray:
    roi = np.zeros((grid_h, grid_w), dtype=np.uint8)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            by = int(row["block_y"])
            bx = int(row["block_x"])
            if not (0 <= by < grid_h and 0 <= bx < grid_w):
                raise ValueError(f"{path}: bloque fuera de rango ({by}, {bx})")
            roi[by, bx] = 1
    return roi


def apply_rect(roi: np.ndarray, rect: list[int]) -> None:
    y0, x0, y1, x1 = rect
    grid_h, grid_w = roi.shape
    if not (0 <= y0 < y1 <= grid_h and 0 <= x0 < x1 <= grid_w):
        raise ValueError(f"rectangulo fuera de rango: {rect}, grid={grid_h}x{grid_w}")
    roi[y0:y1, x0:x1] = 1


def make_roi(args: argparse.Namespace, grid_h: int, grid_w: int) -> np.ndarray:
    roi = load_tsv(args.from_tsv, grid_h, grid_w) if args.from_tsv else np.zeros((grid_h, grid_w), dtype=np.uint8)

    if args.pattern == "center":
        roi[grid_h // 4 : grid_h - grid_h // 4, grid_w // 4 : grid_w - grid_w // 4] = 1
    elif args.pattern == "full":
        roi[:, :] = 1
    elif args.pattern == "checker":
        yy, xx = np.indices((grid_h, grid_w))
        roi[((yy + xx) % 2) == 0] = 1
    elif args.pattern != "empty":
        raise ValueError(f"patron desconocido: {args.pattern}")

    for rect in args.rect:
        apply_rect(roi, rect)
    return roi


def write_roi_tsv(path: Path, roi: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["block_y", "block_x"], delimiter="\t")
        writer.writeheader()
        for by, bx in np.argwhere(roi != 0):
            writer.writerow({"block_y": int(by), "block_x": int(bx)})


def overlay_roi(rgb: Image.Image, roi: np.ndarray, block_size: int) -> Image.Image:
    out = rgb.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for by, bx in np.argwhere(roi != 0):
        x0 = int(bx) * block_size
        y0 = int(by) * block_size
        x1 = x0 + block_size - 1
        y1 = y0 + block_size - 1
        draw.rectangle([x0, y0, x1, y1], fill=(255, 40, 40, 70), outline=(255, 0, 0, 230), width=2)
    return Image.alpha_composite(out, overlay).convert("RGB")


def qmap_preview(qmap_path: Path, output_path: Path) -> None:
    data = np.fromfile(qmap_path, dtype=np.uint8)
    if data.size != 1024:
        raise ValueError(f"{qmap_path}: Q-map inesperado de {data.size} bytes")
    qmap = data.reshape(32, 32)
    norm = np.clip((qmap.astype(np.float64) - 128.0) / (255.0 - 128.0), 0.0, 1.0)
    red = (255.0 * norm).astype(np.uint8)
    blue = (255.0 * (1.0 - norm)).astype(np.uint8)
    green = np.full_like(red, 80, dtype=np.uint8)
    rgb = np.stack([red, green, blue], axis=-1)
    Image.fromarray(rgb).resize((512, 512), Image.Resampling.NEAREST).save(output_path)


def executable(path: Path) -> str:
    if path.is_absolute():
        return str(path)
    if path.parent == Path(".") or str(path.parent) == ".":
        return f"./{path.name}"
    return str(path)


def resolve_calibration(path: Path | None) -> Path:
    if path is not None:
        return path
    if DEFAULT_CALIBRATION_WIDE.exists():
        return DEFAULT_CALIBRATION_WIDE
    return DEFAULT_CALIBRATION


def run_qmap(args: argparse.Namespace, calibration: Path, roi_map: Path, qmap_path: Path, summary_tsv: Path, log_path: Path) -> None:
    cmd = [
        executable(args.semantic_bin),
        "--calibration",
        str(calibration),
        "--preset",
        "manual",
        "--roi-map",
        str(roi_map),
        "--output-qmap",
        str(qmap_path),
        "--summary-tsv",
        str(summary_tsv),
        "--q-mean",
        "204",
        "--adaptive-strength",
        "8",
        *PROFILE_ARGS[args.profile],
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    log_path.write_text(
        "\n".join(["$ " + " ".join(cmd), f"exit_code={proc.returncode}", "", "== STDOUT ==", proc.stdout, "== STDERR ==", proc.stderr]),
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"fallo generando Q-map manual; ver {log_path}")


def write_commands(path: Path, args: argparse.Namespace, calibration: Path, roi_map: Path, qmap_path: Path) -> None:
    bitstream = args.output_dir / "manual_focus.bin"
    reconstruction = args.output_dir / "manual_focus.raw"
    metrics_json = args.output_dir / "manual_focus_quality.json"
    metrics_csv = args.output_dir / "manual_focus_block_quality.csv"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# 1) Generar Q-map manual en C",
        " ".join(
            [
                executable(args.semantic_bin),
                "--calibration",
                str(calibration),
                "--preset manual",
                "--roi-map",
                str(roi_map),
                "--output-qmap",
                str(qmap_path),
                "--summary-tsv",
                str(args.output_dir / f"qmap_manual_{args.profile}.tsv"),
                "--q-mean 204 --adaptive-strength 8",
                " ".join(PROFILE_ARGS[args.profile]),
            ]
        ),
        "",
        "# 2) Comprimir y descomprimir con SORTENY C",
        f"./sorteny_compressor {args.input} 0.1 {bitstream} weights/encoder 0.125 {qmap_path}",
        f"./sorteny_decompressor {bitstream} {reconstruction} weights/decoder 0.125",
        "",
        "# 3) Analizar calidad por bloque",
        (
            "python3 src/python/analysis/analyze_block_quality.py "
            f"{args.input} {reconstruction} --qmap {qmap_path} "
            f"--output-json {metrics_json} --output-csv {metrics_csv}"
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grid_h = args.height // args.block_size
    grid_w = args.width // args.block_size
    raw = load_raw(args.input, args.bands, args.height, args.width)
    rgb = make_rgb(raw)
    roi = make_roi(args, grid_h, grid_w)

    quicklook = args.output_dir / "quicklook_rgb.png"
    quicklook_grid = args.output_dir / "quicklook_grid_32x32.png"
    roi_overlay = args.output_dir / "quicklook_roi_overlay.png"
    roi_map = args.output_dir / "roi_map.raw"
    roi_tsv = args.output_dir / "roi.tsv"
    roi_summary = args.output_dir / "roi_summary.json"
    qmap = args.output_dir / f"qmap_manual_{args.profile}.raw"
    qmap_tsv = args.output_dir / f"qmap_manual_{args.profile}.tsv"
    qmap_png = args.output_dir / f"qmap_manual_{args.profile}.png"
    command_file = args.output_dir / "operator_commands.sh"

    rgb.save(quicklook)
    draw_grid(rgb, args.block_size).save(quicklook_grid)
    draw_grid(overlay_roi(rgb, roi, args.block_size), args.block_size, color=(255, 255, 255)).save(roi_overlay)

    payload = roi.reshape(-1).astype(np.uint8).tobytes()
    roi_map.write_bytes(payload)
    write_roi_tsv(roi_tsv, roi)

    calibration = resolve_calibration(args.calibration)
    if not args.skip_qmap:
        run_qmap(args, calibration, roi_map, qmap, qmap_tsv, args.output_dir / "qmap_generation.log")
        qmap_preview(qmap, qmap_png)

    write_commands(command_file, args, calibration, roi_map, qmap)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": args.input,
        "profile": args.profile,
        "profile_args": PROFILE_ARGS[args.profile],
        "grid": f"{grid_h}x{grid_w}",
        "block_size": args.block_size,
        "roi_blocks": int(np.count_nonzero(roi)),
        "background_blocks": int(roi.size - np.count_nonzero(roi)),
        "roi_pct": float(np.count_nonzero(roi) * 100.0 / roi.size),
        "roi_sha256": hashlib.sha256(payload).hexdigest(),
        "calibration": calibration,
        "outputs": {
            "quicklook": quicklook,
            "quicklook_grid": quicklook_grid,
            "roi_overlay": roi_overlay,
            "roi_map": roi_map,
            "roi_tsv": roi_tsv,
            "qmap": qmap if qmap.exists() else None,
            "qmap_summary": qmap_tsv if qmap_tsv.exists() else None,
            "qmap_preview": qmap_png if qmap_png.exists() else None,
            "operator_commands": command_file,
        },
    }
    roi_summary.write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    readme = args.output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# ROI Manual Pre-Web",
                "",
                "Este checkpoint genera una seleccion ROI manual sin usar aplicacion web.",
                "",
                f"- Entrada: `{args.input}`",
                f"- Perfil: `{args.profile}`",
                f"- Bloques ROI: `{summary['roi_blocks']}` de `{roi.size}` ({summary['roi_pct']:.2f}%)",
                "",
                "Archivos principales:",
                "- `quicklook_grid_32x32.png`: vista RGB con rejilla de bloques.",
                "- `quicklook_roi_overlay.png`: vista RGB con ROI superpuesta.",
                "- `roi_map.raw`: mascara `uint8` 32x32 consumida por C.",
                "- `roi.tsv`: bloques seleccionados para auditoria.",
                f"- `qmap_manual_{args.profile}.raw`: Q-map generado por `sorteny_semantic_qmap --preset manual`.",
                "- `operator_commands.sh`: comandos para continuar con compresion/descompresion C.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
