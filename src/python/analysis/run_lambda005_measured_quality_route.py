#!/usr/bin/env python3
"""Run the measured-error lambda/Q route for SORTENY lambda005.

This script is intentionally an orchestrator: it runs the existing C binaries,
measures the base reconstruction error, derives measured Q-maps, applies them
with the C codec, and writes traceable metrics. It does not replace the C codec.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


MAX_SAMPLE_VALUE = 65535.0
QMAP_SIDE = 32
QMAP_BYTES = QMAP_SIDE * QMAP_SIDE
DEFAULT_MODES = [
    "global_target_measured",
    "adaptive_s8_measured",
    "focus_bgq128_measured",
    "target_focus_bgq128_measured",
    "preserve_roi_q240_bgq128",
    "preserve_roi_q255_bgq128",
]


@dataclass(frozen=True)
class CalibrationBlock:
    by: int
    bx: int
    c0: float
    c1: float
    valid: bool
    q_baseline: int
    q_min: int
    q_max: int
    max_lambda: float
    mod_a: float
    mod_b: float
    mse_at_baseline: float


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def executable(root: Path, value: str | Path) -> str:
    p = resolve_path(root, value)
    if p.exists():
        return str(p)
    found = shutil.which(str(value))
    if found:
        return found
    return str(p)


def run_cmd(cmd: list[str], log_prefix: Path, cwd: Path, env: dict[str, str] | None = None) -> float:
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (log_prefix.with_suffix(".stdout.txt")).open("w", encoding="utf-8") as out, (
        log_prefix.with_suffix(".stderr.txt")
    ).open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=cwd, stdout=out, stderr=err, env=env)
    elapsed = time.perf_counter() - started
    meta = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "started_at": now_iso(),
    }
    log_prefix.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return elapsed


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def psnr_from_mse(mse: float) -> float:
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10((MAX_SAMPLE_VALUE * MAX_SAMPLE_VALUE) / mse)


def mse_from_psnr(psnr_db: float) -> float:
    return (MAX_SAMPLE_VALUE * MAX_SAMPLE_VALUE) / (10.0 ** (psnr_db / 10.0))


def load_raw_u16(path: Path, bands: int, height: int, width: int) -> np.ndarray:
    data = np.fromfile(path, dtype="<u2")
    expected = bands * height * width
    if data.size != expected:
        raise ValueError(f"{path}: expected {expected} uint16 samples, got {data.size}")
    return data.reshape((bands, height, width))


def load_qmap(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size != QMAP_BYTES:
        raise ValueError(f"{path}: expected {QMAP_BYTES} bytes, got {data.size}")
    return data.reshape((QMAP_SIDE, QMAP_SIDE))


def write_qmap(path: Path, qmap: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    q = np.asarray(qmap, dtype=np.uint8)
    if q.shape != (QMAP_SIDE, QMAP_SIDE):
        raise ValueError(f"qmap shape must be {(QMAP_SIDE, QMAP_SIDE)}, got {q.shape}")
    q.tofile(path)


def metrics_for_arrays(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff * diff))
    return {
        "mse": mse,
        "psnr_db": psnr_from_mse(mse),
        "mae": float(np.mean(abs_diff)),
        "max_abs": float(np.max(abs_diff)),
        "exact_pct": float(np.mean(abs_diff == 0.0) * 100.0),
    }


def block_mse(original: np.ndarray, reconstructed: np.ndarray, block_size: int) -> np.ndarray:
    bands, height, width = original.shape
    qh = height // block_size
    qw = width // block_size
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    out = np.zeros((qh, qw), dtype=np.float64)
    for by in range(qh):
        y0 = by * block_size
        y1 = y0 + block_size
        for bx in range(qw):
            x0 = bx * block_size
            x1 = x0 + block_size
            d = diff[:, y0:y1, x0:x1]
            out[by, bx] = float(np.mean(d * d))
    return out


def group_summary(mse_map: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if not np.any(mask):
        return {"blocks": 0, "mse": math.nan, "psnr_db": math.nan}
    mse = float(np.mean(mse_map[mask]))
    return {"blocks": int(np.sum(mask)), "mse": mse, "psnr_db": psnr_from_mse(mse)}


def q_summary(qmap: np.ndarray, roi: np.ndarray | None = None) -> dict[str, Any]:
    values = qmap.astype(np.float64)
    out: dict[str, Any] = {
        "q_min": int(np.min(qmap)),
        "q_max": int(np.max(qmap)),
        "q_mean": float(np.mean(values)),
        "q_unique": int(np.unique(qmap).size),
        "qmap_bytes": int(qmap.size),
    }
    if roi is not None:
        bg = ~roi
        out["q_roi_mean"] = float(np.mean(qmap[roi])) if np.any(roi) else math.nan
        out["q_background_mean"] = float(np.mean(qmap[bg])) if np.any(bg) else math.nan
    return out


def load_calibration(path: Path) -> list[CalibrationBlock]:
    rows: list[CalibrationBlock] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t")
        for row in reader:
            rows.append(
                CalibrationBlock(
                    by=int(row["block_y"]),
                    bx=int(row["block_x"]),
                    c0=float(row["c0"]),
                    c1=float(row["c1"]),
                    valid=(int(row["valid"]) != 0),
                    q_baseline=int(row["q_baseline"]),
                    q_min=int(row["q_min"]),
                    q_max=int(row["q_max"]),
                    max_lambda=float(row["max_lambda"]),
                    mod_a=float(row["mod_a"]),
                    mod_b=float(row["mod_b"]),
                    mse_at_baseline=float(row["mse_at_baseline"]),
                )
            )
    if len(rows) != QMAP_BYTES:
        raise ValueError(f"{path}: expected {QMAP_BYTES} calibration rows, got {len(rows)}")
    rows.sort(key=lambda r: (r.by, r.bx))
    return rows


def mod_value(block: CalibrationBlock, q: int, max_lambda: float) -> float:
    lam = (float(q) / 255.0) * max_lambda
    return block.mod_a * lam + block.mod_b


def clamp_q(q: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, q))


def select_q_from_target(
    block: CalibrationBlock,
    measured_mse0: float,
    target_mse: float,
    q_baseline: int,
    q_min: int,
    q_max: int,
    max_lambda: float,
) -> tuple[int, str, float]:
    """Select Q using a measured base error and the fixed-quality model shape."""
    c0 = max(0.0, block.c0)
    m_base = max(mod_value(block, q_baseline, max_lambda), 1e-12)
    if not block.valid or measured_mse0 <= c0:
        # Fallback keeps the measured route conservative when the floor is not usable.
        c1_measured = max(block.c1, 1e-12)
        reason_prefix = "fallback_calibrated_c1"
    else:
        c1_measured = max((measured_mse0 - c0) * (m_base * m_base), 1e-12)
        reason_prefix = "measured_c1"

    if target_mse <= c0:
        return q_max, f"{reason_prefix}_target_below_floor", c1_measured

    target_mod = math.sqrt(c1_measured / max(target_mse - c0, 1e-12))
    lam = (target_mod - block.mod_b) / max(block.mod_a, 1e-12)
    q_float = (lam / max_lambda) * 255.0
    if not math.isfinite(q_float):
        return q_baseline, f"{reason_prefix}_nonfinite", c1_measured

    q_raw = int(round(q_float))
    q = clamp_q(q_raw, q_min, q_max)
    if q != q_raw:
        suffix = "clamped_low" if q_raw < q_min else "clamped_high"
        return q, f"{reason_prefix}_{suffix}", c1_measured
    return q, reason_prefix, c1_measured


def read_roi_from_semantic_tsv(path: Path) -> np.ndarray:
    rows: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(int(row["semantic_match"]))
    if len(rows) != QMAP_BYTES:
        raise ValueError(f"{path}: expected {QMAP_BYTES} rows, got {len(rows)}")
    return np.array(rows, dtype=bool).reshape((QMAP_SIDE, QMAP_SIDE))


def write_summary_tsv(
    path: Path,
    mode: str,
    qmap: np.ndarray,
    measured_mse0: np.ndarray,
    target_mse_map: np.ndarray | None,
    roi: np.ndarray | None,
    reasons: np.ndarray | None,
    c1_measured: np.ndarray | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "block_y",
            "block_x",
            "mode",
            "measured_mse0",
            "target_mse",
            "final_q",
            "roi",
            "reason",
            "c1_measured",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for by in range(QMAP_SIDE):
            for bx in range(QMAP_SIDE):
                writer.writerow(
                    {
                        "block_y": by,
                        "block_x": bx,
                        "mode": mode,
                        "measured_mse0": f"{measured_mse0[by, bx]:.9g}",
                        "target_mse": ""
                        if target_mse_map is None
                        else f"{target_mse_map[by, bx]:.9g}",
                        "final_q": int(qmap[by, bx]),
                        "roi": "" if roi is None else int(bool(roi[by, bx])),
                        "reason": "" if reasons is None else str(reasons[by, bx]),
                        "c1_measured": ""
                        if c1_measured is None
                        else f"{c1_measured[by, bx]:.9g}",
                    }
                )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_measured_qmaps(
    args: argparse.Namespace,
    blocks: list[CalibrationBlock],
    measured_mse0: np.ndarray,
    roi: np.ndarray,
) -> dict[str, dict[str, Any]]:
    q_min = args.operational_q_min
    q_max = args.operational_q_max
    target_mse = args.target_mse if args.target_mse is not None else mse_from_psnr(args.target_psnr)

    measured: dict[str, dict[str, Any]] = {}
    block_array = np.array(blocks, dtype=object).reshape((QMAP_SIDE, QMAP_SIDE))

    if "global_target_measured" in args.modes:
        c0 = float(np.mean([max(b.c0, 0.0) for b in blocks]))
        template = blocks[0]
        m_base = max(mod_value(template, args.q_baseline, args.max_lambda), 1e-12)
        global_mse0 = float(np.mean(measured_mse0))
        c1 = max((global_mse0 - c0) * (m_base * m_base), 1e-12)
        pseudo = CalibrationBlock(
            0,
            0,
            c0,
            c1,
            True,
            args.q_baseline,
            q_min,
            q_max,
            args.max_lambda,
            template.mod_a,
            template.mod_b,
            global_mse0,
        )
        q, reason, c1_measured = select_q_from_target(
            pseudo,
            global_mse0,
            target_mse,
            args.q_baseline,
            q_min,
            q_max,
            args.max_lambda,
        )
        measured["global_target_measured"] = {
            "qmap": np.full((QMAP_SIDE, QMAP_SIDE), q, dtype=np.uint8),
            "target_mse": np.full((QMAP_SIDE, QMAP_SIDE), target_mse, dtype=np.float64),
            "reasons": np.full((QMAP_SIDE, QMAP_SIDE), reason, dtype=object),
            "c1_measured": np.full((QMAP_SIDE, QMAP_SIDE), c1_measured, dtype=np.float64),
        }

    adaptive_qmap: np.ndarray | None = None
    if any(m in args.modes for m in ["adaptive_s8_measured", "focus_bgq128_measured"]):
        valid = measured_mse0 > 0.0
        logs = np.log(np.where(valid, measured_mse0, np.nan))
        log_mean = float(np.nanmean(logs))
        log_std = float(np.nanstd(logs))
        if not math.isfinite(log_std) or log_std <= 0.0:
            log_std = 1.0
        z = (np.log(np.maximum(measured_mse0, 1e-12)) - log_mean) / log_std
        adaptive_qmap = np.rint(args.q_baseline + args.adaptive_strength * z).astype(np.int32)
        adaptive_qmap = np.clip(adaptive_qmap, q_min, q_max).astype(np.uint8)
        if "adaptive_s8_measured" in args.modes:
            measured["adaptive_s8_measured"] = {
                "qmap": adaptive_qmap,
                "target_mse": measured_mse0.copy(),
                "reasons": np.full((QMAP_SIDE, QMAP_SIDE), "relative_measured_mse0", dtype=object),
                "c1_measured": None,
            }

    if "focus_bgq128_measured" in args.modes:
        if adaptive_qmap is None:
            raise ValueError("focus_bgq128_measured requires adaptive base")
        focus = np.where(
            roi,
            np.clip(adaptive_qmap.astype(np.int32) + args.foreground_boost, q_min, q_max),
            args.background_q,
        ).astype(np.uint8)
        measured["focus_bgq128_measured"] = {
            "qmap": focus,
            "target_mse": None,
            "reasons": np.where(roi, "roi_adaptive_boost", "background_fixed_q128"),
            "c1_measured": None,
        }

    if "target_focus_bgq128_measured" in args.modes:
        q = np.full((QMAP_SIDE, QMAP_SIDE), args.background_q, dtype=np.uint8)
        reasons = np.full((QMAP_SIDE, QMAP_SIDE), "background_fixed_q128", dtype=object)
        c1_map = np.full((QMAP_SIDE, QMAP_SIDE), np.nan, dtype=np.float64)
        target_map = np.full((QMAP_SIDE, QMAP_SIDE), target_mse, dtype=np.float64)
        for by in range(QMAP_SIDE):
            for bx in range(QMAP_SIDE):
                if not roi[by, bx]:
                    continue
                block = block_array[by, bx]
                qq, reason, c1_measured = select_q_from_target(
                    block,
                    float(measured_mse0[by, bx]),
                    target_mse,
                    args.q_baseline,
                    q_min,
                    q_max,
                    args.max_lambda,
                )
                q[by, bx] = qq
                reasons[by, bx] = reason
                c1_map[by, bx] = c1_measured
        measured["target_focus_bgq128_measured"] = {
            "qmap": q,
            "target_mse": target_map,
            "reasons": reasons,
            "c1_measured": c1_map,
        }

    for fg in args.preserve_q_values:
        mode = f"preserve_roi_q{fg}_bgq{args.background_q}"
        if mode not in args.modes:
            continue
        q = np.where(roi, fg, args.background_q).astype(np.uint8)
        measured[mode] = {
            "qmap": q,
            "target_mse": None,
            "reasons": np.where(roi, f"roi_fixed_q{fg}", f"background_fixed_q{args.background_q}"),
            "c1_measured": None,
        }
    return measured


def qmap_diff(measured: np.ndarray, precalibrated: np.ndarray) -> dict[str, Any]:
    diff = measured.astype(np.int16) - precalibrated.astype(np.int16)
    return {
        "equal_blocks": int(np.sum(diff == 0)),
        "equal_pct": float(np.mean(diff == 0) * 100.0),
        "mean_delta_q": float(np.mean(diff)),
        "max_abs_delta_q": int(np.max(np.abs(diff))),
        "measured_q_mean": float(np.mean(measured)),
        "precalibrated_q_mean": float(np.mean(precalibrated)),
    }


def run_codec_case(
    args: argparse.Namespace,
    root: Path,
    outdir: Path,
    mode: str,
    qmap_path: Path,
    original: np.ndarray,
    roi: np.ndarray,
    measured_mse0: np.ndarray,
) -> dict[str, Any]:
    bitstream = outdir / "bitstreams" / f"{mode}.bin"
    recon = outdir / "reconstructions" / f"{mode}.raw"
    logs = outdir / "logs"
    bitstream.parent.mkdir(parents=True, exist_ok=True)
    recon.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    compress_s = run_cmd(
        [
            executable(root, args.compressor),
            str(args.input),
            f"{args.lambda_value:.8g}",
            str(bitstream),
            str(args.encoder_weights),
            f"{args.max_lambda:.8g}",
            str(qmap_path),
        ],
        logs / f"{mode}_compress",
        root,
        env=env,
    )
    decompress_s = run_cmd(
        [
            executable(root, args.decompressor),
            str(bitstream),
            str(recon),
            str(args.decoder_weights),
            f"{args.max_lambda:.8g}",
        ],
        logs / f"{mode}_decompress",
        root,
        env=env,
    )

    reconstructed = load_raw_u16(recon, args.bands, args.height, args.width)
    global_metrics = metrics_for_arrays(original, reconstructed)
    mse_map = block_mse(original, reconstructed, args.block_size)
    roi_metrics = group_summary(mse_map, roi)
    bg_metrics = group_summary(mse_map, ~roi)
    qmap = load_qmap(qmap_path)
    qstats = q_summary(qmap, roi)
    base_mse_ref = float(np.mean(measured_mse0))
    delta_global = (
        global_metrics["psnr_db"] - psnr_from_mse(base_mse_ref)
        if base_mse_ref > 0.0
        else math.nan
    )

    return {
        "mode": mode,
        "global_mse": global_metrics["mse"],
        "global_psnr_db": global_metrics["psnr_db"],
        "global_mae": global_metrics["mae"],
        "roi_blocks": roi_metrics["blocks"],
        "roi_mse": roi_metrics["mse"],
        "roi_psnr_db": roi_metrics["psnr_db"],
        "background_blocks": bg_metrics["blocks"],
        "background_mse": bg_metrics["mse"],
        "background_psnr_db": bg_metrics["psnr_db"],
        **qstats,
        "compress_elapsed_s": compress_s,
        "decompress_elapsed_s": decompress_s,
        "total_elapsed_s": compress_s + decompress_s,
        "bitstream_bytes": bitstream.stat().st_size,
        "reconstruction_bytes": recon.stat().st_size,
        "qmap_sha256": sha256_file(qmap_path),
        "bitstream_sha256": sha256_file(bitstream),
        "reconstruction_sha256": sha256_file(recon),
        "delta_global_psnr_vs_base": delta_global,
    }


def generate_precalibrated_qmap(
    args: argparse.Namespace,
    root: Path,
    outdir: Path,
    mode: str,
) -> Path | None:
    qmap = outdir / "qmaps" / "precalibrated" / f"{mode}.raw"
    tsv = outdir / "summary_tsv" / "precalibrated" / f"{mode}.tsv"
    qmap.parent.mkdir(parents=True, exist_ok=True)
    tsv.parent.mkdir(parents=True, exist_ok=True)

    if mode == "global_target_measured":
        cmd = [
            executable(root, args.fq_qmap),
            "--calibration",
            str(args.calibration),
            "--target-psnr",
            f"{args.target_psnr:.8g}" if args.target_mse is None else f"{psnr_from_mse(args.target_mse):.8g}",
            "--output-qmap",
            str(qmap),
            "--summary-tsv",
            str(tsv),
        ]
    elif mode == "adaptive_s8_measured":
        cmd = [
            executable(root, args.fq_qmap),
            "--calibration",
            str(args.calibration),
            "--adaptive-difficulty",
            "--q-mean",
            str(args.q_baseline),
            "--adaptive-strength",
            f"{args.adaptive_strength:.8g}",
            "--output-qmap",
            str(qmap),
            "--summary-tsv",
            str(tsv),
        ]
    elif mode == "focus_bgq128_measured":
        cmd = semantic_cmd(args, root, qmap, tsv, "focus", ["--foreground-boost", str(args.foreground_boost)])
    elif mode == "target_focus_bgq128_measured":
        target_arg = ["--roi-target-mse", f"{args.target_mse:.9g}"] if args.target_mse is not None else [
            "--roi-target-psnr",
            f"{args.target_psnr:.8g}",
        ]
        cmd = semantic_cmd(args, root, qmap, tsv, "target-focus", target_arg)
    elif mode.startswith("preserve_roi_q"):
        fg = int(mode.split("_q", 1)[1].split("_", 1)[0])
        cmd = semantic_cmd(args, root, qmap, tsv, "preserve-roi", ["--foreground-q", str(fg)])
    else:
        return None

    run_cmd(cmd, outdir / "logs" / f"precalibrated_{mode}", root)
    return qmap


def semantic_cmd(
    args: argparse.Namespace,
    root: Path,
    qmap: Path,
    tsv: Path,
    policy: str,
    extra: list[str],
) -> list[str]:
    cmd = [
        executable(root, args.semantic_qmap),
        "--input",
        str(args.input),
        "--calibration",
        str(args.calibration),
        "--preset",
        args.preset,
        "--threshold",
        f"{args.threshold:.8g}",
        "--semantic-policy",
        policy,
        "--background-q",
        str(args.background_q),
        "--output-qmap",
        str(qmap),
        "--summary-tsv",
        str(tsv),
    ]
    return cmd + extra


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_out = root / "output" / "checkpoints" / f"lambda005_measured_quality_route_{now_stamp()}"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=root / "data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw")
    p.add_argument(
        "--calibration",
        type=Path,
        default=root / "config/fq_calibration_lambda005.tsv",
    )
    p.add_argument("--output-dir", type=Path, default=default_out)
    p.add_argument("--compressor", default="./sorteny_compressor")
    p.add_argument("--decompressor", default="./sorteny_decompressor")
    p.add_argument("--fq-qmap", default="./sorteny_fq_qmap")
    p.add_argument("--semantic-qmap", default="./sorteny_semantic_qmap")
    p.add_argument("--encoder-weights", type=Path, default=root / "weights/encoder")
    p.add_argument("--decoder-weights", type=Path, default=root / "weights/decoder")
    p.add_argument("--bands", type=int, default=8)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--lambda-value", type=float, default=0.05)
    p.add_argument("--max-lambda", type=float, default=0.05)
    p.add_argument("--target-psnr", type=float, default=76.8)
    p.add_argument("--target-mse", type=float, default=None)
    p.add_argument("--preset", default="cloud_avoid")
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--q-baseline", type=int, default=204)
    p.add_argument("--adaptive-strength", type=float, default=8.0)
    p.add_argument("--foreground-boost", type=int, default=16)
    p.add_argument("--background-q", type=int, default=128)
    p.add_argument("--preserve-q-values", type=int, nargs="+", default=[240, 255])
    p.add_argument("--operational-q-min", type=int, default=128)
    p.add_argument("--operational-q-max", type=int, default=255)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    p.add_argument("--no-precalibrated-compare", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    args.input = resolve_path(root, args.input)
    args.calibration = resolve_path(root, args.calibration)
    args.output_dir = resolve_path(root, args.output_dir)
    args.encoder_weights = resolve_path(root, args.encoder_weights)
    args.decoder_weights = resolve_path(root, args.decoder_weights)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["qmaps", "summary_tsv", "bitstreams", "reconstructions", "logs"]:
        (args.output_dir / sub).mkdir(parents=True, exist_ok=True)

    if args.height // args.block_size != QMAP_SIDE or args.width // args.block_size != QMAP_SIDE:
        raise ValueError("This script currently expects 512x512 with 16x16 blocks, i.e. a 32x32 Q-map")

    blocks = load_calibration(args.calibration)
    original = load_raw_u16(args.input, args.bands, args.height, args.width)

    # Base q204 route.
    q204 = args.output_dir / "qmaps" / "q204.raw"
    q204_tsv = args.output_dir / "summary_tsv" / "q204.tsv"
    run_cmd(
        [
            executable(root, args.fq_qmap),
            "--calibration",
            str(args.calibration),
            "--target-from-q",
            str(args.q_baseline),
            "--output-qmap",
            str(q204),
            "--summary-tsv",
            str(q204_tsv),
        ],
        args.output_dir / "logs" / "qmap_q204",
        root,
    )
    base_row = run_codec_case(args, root, args.output_dir, "base_q204", q204, original, np.ones((QMAP_SIDE, QMAP_SIDE), dtype=bool), np.zeros((QMAP_SIDE, QMAP_SIDE)))
    base_recon = load_raw_u16(args.output_dir / "reconstructions" / "base_q204.raw", args.bands, args.height, args.width)
    measured_mse0 = block_mse(original, base_recon, args.block_size)

    # ROI probe using the same C preset implementation.
    roi_probe_qmap = args.output_dir / "qmaps" / "roi_probe.raw"
    roi_probe_tsv = args.output_dir / "summary_tsv" / "roi_probe.tsv"
    run_cmd(
        semantic_cmd(args, root, roi_probe_qmap, roi_probe_tsv, "focus", ["--foreground-boost", str(args.foreground_boost)]),
        args.output_dir / "logs" / "qmap_roi_probe",
        root,
    )
    roi = read_roi_from_semantic_tsv(roi_probe_tsv)

    measured_qmaps = build_measured_qmaps(args, blocks, measured_mse0, roi)
    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    rows.append(base_row | {"route": "base", "preset": "_none", "roi_pct": 100.0})

    for mode, data in measured_qmaps.items():
        qmap = data["qmap"]
        qmap_path = args.output_dir / "qmaps" / f"{mode}.raw"
        tsv_path = args.output_dir / "summary_tsv" / f"{mode}.tsv"
        write_qmap(qmap_path, qmap)
        write_summary_tsv(
            tsv_path,
            mode,
            qmap,
            measured_mse0,
            data["target_mse"],
            roi,
            data["reasons"],
            data["c1_measured"],
        )

        row = run_codec_case(args, root, args.output_dir, mode, qmap_path, original, roi, measured_mse0)
        row["route"] = "measured_error"
        row["preset"] = args.preset
        row["threshold"] = args.threshold
        row["roi_pct"] = float(np.mean(roi) * 100.0)
        rows.append(row)

        if not args.no_precalibrated_compare:
            precal_path = generate_precalibrated_qmap(args, root, args.output_dir, mode)
            if precal_path is not None:
                cmp_row = {
                    "mode": mode,
                    **qmap_diff(qmap, load_qmap(precal_path)),
                    "measured_qmap": str(qmap_path),
                    "precalibrated_qmap": str(precal_path),
                }
                comparisons.append(cmp_row)

    write_csv(args.output_dir / "metrics.csv", rows)
    (args.output_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "qmap_precalibrated_comparison.csv", comparisons)
    (args.output_dir / "qmap_precalibrated_comparison.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )

    run_meta = {
        "script": str(Path(__file__).relative_to(root)),
        "created_at": now_iso(),
        "input": str(args.input),
        "calibration": str(args.calibration),
        "output_dir": str(args.output_dir),
        "modes": args.modes,
        "preset": args.preset,
        "threshold": args.threshold,
        "target_psnr": args.target_psnr,
        "target_mse": args.target_mse if args.target_mse is not None else mse_from_psnr(args.target_psnr),
        "max_lambda": args.max_lambda,
        "q_baseline": args.q_baseline,
        "operational_q_range": [args.operational_q_min, args.operational_q_max],
        "threads": args.threads,
        "input_sha256": sha256_file(args.input),
        "calibration_sha256": sha256_file(args.calibration),
        "roi_blocks": int(np.sum(roi)),
        "roi_pct": float(np.mean(roi) * 100.0),
        "metrics_csv": str(args.output_dir / "metrics.csv"),
        "qmap_comparison_csv": str(args.output_dir / "qmap_precalibrated_comparison.csv"),
    }
    (args.output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"[OK] measured quality route written to {args.output_dir}")
    print(f"[OK] rows={len(rows)} comparisons={len(comparisons)} roi_pct={run_meta['roi_pct']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
