#!/usr/bin/env python3
"""
Run CSMR bitrate audit for preserve-roi policy variants.

Input is the C-only preserve-roi checkpoint. The Q-maps are already generated
by C; this script only snapshots them, converts Q to lambda with lambda_max
0.05, runs SORTENY.py/CSMR, and reports real .tfci bitrate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import audit_csmr_dataset_smoke_bitrate as smoke
import audit_csmr_real_bitrate as csmr_base
import build_global_adaptive_focus_case_study as base


DEFAULT_PRESERVE_CHECKPOINT = Path("output/checkpoints/20260627_lambda005_preserve_roi_policy_audit")
DEFAULT_OUTPUT_DIR = Path("output/checkpoints/20260628_lambda005_csmr_preserve_roi_policy")
DEFAULT_THRESHOLD_CHECKPOINT = Path("output/checkpoints/20260625_lambda005_threshold_optimization")
DEFAULT_FULL_CHECKPOINT = Path("output/checkpoints/20260619_lambda005_semantic_presets_full_auto_modes")
DEFAULT_MODEL_PATH = Path("models/SORTENY_Sentinel2_model")
DEFAULT_SORTENY_PY = Path("src/python/reference/SORTENY.py")
CONTROL_POLICIES = ["q204", "adaptive_s8"]
FOCUS_POLICY = "focus_bgq128"
PRESERVE_POLICIES = ["preserve_roi_q255_bgq128", "preserve_roi_q240_bgq128"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preserve-checkpoint", type=Path, default=DEFAULT_PRESERVE_CHECKPOINT)
    parser.add_argument("--threshold-checkpoint", type=Path, default=DEFAULT_THRESHOLD_CHECKPOINT)
    parser.add_argument("--full-checkpoint", type=Path, default=DEFAULT_FULL_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sorteny-py", type=Path, default=DEFAULT_SORTENY_PY)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--lambda-min", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=0.05)
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all selected C-only cases.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    smoke.write_csv(path, rows)


def write_json(path: Path, data: Any) -> None:
    smoke.write_json(path, data)


def finite_float(value: Any) -> float:
    return smoke.finite_float(value)


def fmt(value: Any, digits: int = 4) -> str:
    return smoke.fmt(value, digits)


def require_inputs(root: Path, args: argparse.Namespace, preserve_dir: Path) -> dict[str, Any]:
    required = [
        preserve_dir / "preserve_roi_policy_summary.csv",
        preserve_dir / "preserve_roi_group_summary.csv",
        preserve_dir / "run_manifest.json",
        resolve(root, args.model_path) / "saved_model.pb",
        resolve(root, args.sorteny_py),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(f"- {p}" for p in missing))
    return {
        "preserve_manifest": read_json(preserve_dir / "run_manifest.json"),
        "tensorflow_probe": csmr_base.require_preflight(root, args),
    }


def control_case_path(threshold_dir: Path, full_dir: Path, crop: str, policy: str) -> Path:
    candidates = [
        threshold_dir / "cases" / crop / "_controls" / policy / "case_result.json",
        full_dir / "cases" / crop / "_controls" / policy / "case_result.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No control case_result found for {crop}/{policy}")


def control_source_row(case_result: dict[str, Any], raw_path: Path) -> dict[str, Any]:
    metrics = case_result.get("metrics", {})
    global_metrics = metrics.get("global", {})
    latent = metrics.get("latent", {})
    artifacts = case_result.get("artifacts", {})
    return {
        "crop": case_result.get("crop", ""),
        "preset": "_control",
        "policy": case_result.get("policy", ""),
        "status": "valid" if case_result.get("case_status") == "done" else case_result.get("case_status", ""),
        "roi_group": "_control",
        "file": raw_path,
        "qmap_path": artifacts.get("qmap_path", ""),
        "summary_tsv": "",
        "global_psnr_db": global_metrics.get("psnr_db", ""),
        "global_mse": global_metrics.get("mse", ""),
        "global_mae": global_metrics.get("mae", ""),
        "latent_zero_pct": latent.get("zero_pct", ""),
        "latent_entropy_bits_per_symbol": latent.get("entropy_bits_per_symbol", ""),
        "latent_zlib_bps_per_input_sample": latent.get("zlib_bps_per_input_sample", ""),
    }


def snapshot_qmap(source: Path, target: Path, force: bool) -> Path:
    if not source.exists():
        raise FileNotFoundError(source)
    if source.stat().st_size != base.Q_BYTES:
        raise ValueError(f"{source}: expected {base.Q_BYTES} bytes, got {source.stat().st_size}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if force or not target.exists():
        shutil.copy2(source, target)
    if target.stat().st_size != base.Q_BYTES:
        raise ValueError(f"{target}: expected {base.Q_BYTES} bytes, got {target.stat().st_size}")
    return target


def resolve_summary_tsv(path: Path) -> Path:
    if path.exists():
        return path
    text = str(path)
    if "/qmaps/" in text:
        candidate = Path(text.replace("/qmaps/", "/summary_tsv/"))
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


def clean_incomplete(outdir: Path, crop: str, preset: str, policy: str) -> None:
    result_path = smoke.case_result_path(outdir, crop, preset, policy)
    if result_path.exists():
        try:
            if read_json(result_path).get("case_status") == "done":
                return
        except json.JSONDecodeError:
            pass
    work_dir = smoke.case_work_dir(outdir, crop, preset, policy)
    input_copy = work_dir / "input.8_512_512_2_1.raw"
    tfci = Path(str(input_copy) + ".tfci")
    recon = Path(str(tfci) + ".raw")
    tfci.unlink(missing_ok=True)
    recon.unlink(missing_ok=True)


def process_control(
    root: Path,
    args: argparse.Namespace,
    outdir: Path,
    progress: Path,
    threshold_dir: Path,
    full_dir: Path,
    crop: str,
    raw_path: Path,
    policy: str,
) -> dict[str, Any]:
    case_path = control_case_path(threshold_dir, full_dir, crop, policy)
    case_result = read_json(case_path)
    source_row = control_source_row(case_result, raw_path)
    source_qmap = Path(str(case_result["artifacts"]["qmap_path"]))
    qmap = snapshot_qmap(source_qmap, outdir / "qmap_snapshots" / crop / "_controls" / f"{policy}.bin", args.force)
    clean_incomplete(outdir, crop, "_control", policy)
    row = smoke.process_one(
        root=root,
        args=args,
        outdir=outdir,
        progress=progress,
        crop=crop,
        preset="_control",
        policy=policy,
        source_kind="control",
        source_row=source_row,
        raw_path=raw_path,
        qmap_path=qmap,
        summary_tsv=None,
        adaptive_blocks=None,
        adaptive_global=None,
        adaptive_tfci_bps=math.nan,
    )
    row["source_c_case_result_path"] = str(case_path)
    row["source_c_qmap_path"] = str(source_qmap)
    return row


def adaptive_cache_for(cache: dict[str, dict[str, Any]], crop: str, raw_path: Path, row: dict[str, Any]) -> dict[str, Any]:
    if crop in cache:
        return cache[crop]
    recon_path = Path(str(row["reconstruction_path"]))
    original = base.load_raw(raw_path)
    reconstruction = base.load_raw(recon_path)
    cache[crop] = {
        "blocks": base.block_metrics(original, reconstruction),
        "global": base.metrics_for_arrays(original, reconstruction),
        "tfci_bps": float(row["tfci_bps_per_input_sample"]),
    }
    return cache[crop]


def process_semantic(
    root: Path,
    args: argparse.Namespace,
    outdir: Path,
    progress: Path,
    source_row: dict[str, Any],
    adaptive: dict[str, Any],
) -> dict[str, Any]:
    crop = str(source_row["crop"])
    preset = str(source_row["preset"])
    policy = str(source_row["policy"])
    raw_path = Path(str(source_row["file"]))
    source_qmap = Path(str(source_row["qmap_path"]))
    summary_tsv = resolve_summary_tsv(Path(str(source_row["summary_tsv"])))
    qmap = snapshot_qmap(source_qmap, outdir / "qmap_snapshots" / crop / preset / f"{policy}.bin", args.force)
    clean_incomplete(outdir, crop, preset, policy)
    row = smoke.process_one(
        root=root,
        args=args,
        outdir=outdir,
        progress=progress,
        crop=crop,
        preset=preset,
        policy=policy,
        source_kind=str(source_row.get("policy_kind", "semantic")),
        source_row=source_row,
        raw_path=raw_path,
        qmap_path=qmap,
        summary_tsv=summary_tsv,
        adaptive_blocks=adaptive["blocks"],
        adaptive_global=adaptive["global"],
        adaptive_tfci_bps=adaptive["tfci_bps"],
    )
    row["source_c_qmap_path"] = str(source_qmap)
    row["source_c_summary_tsv"] = str(summary_tsv)
    row["foreground_q"] = source_row.get("foreground_q", "")
    row["background_q"] = source_row.get("background_q", "")
    return row


def selected_source_rows(rows: list[dict[str, str]], max_cases: int) -> list[dict[str, Any]]:
    focus = [dict(r) for r in rows if r.get("policy") == FOCUS_POLICY]
    preserve = [dict(r) for r in rows if r.get("policy") in PRESERVE_POLICIES]
    keys = []
    seen = set()
    for row in focus:
        key = (row["crop"], row["preset"])
        if key not in seen:
            keys.append(key)
            seen.add(key)
    if max_cases > 0:
        keys = keys[:max_cases]
    keyset = set(keys)
    out = [r for r in focus + preserve if (r["crop"], r["preset"]) in keyset]
    return out


def summarize_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("preset") == "_control":
            continue
        groups.setdefault((row["preset"], row.get("roi_group", ""), row["policy"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for (preset, roi_group, policy), items in sorted(groups.items()):
        def avg(key: str) -> float:
            vals = [finite_float(row.get(key)) for row in items]
            vals = [v for v in vals if math.isfinite(v)]
            return float(np.mean(vals)) if vals else math.nan

        summary.append(
            {
                "preset": preset,
                "roi_group": roi_group,
                "policy": policy,
                "runs": len(items),
                "tfci_bps_mean": avg("tfci_bps_per_input_sample"),
                "delta_tfci_bps_vs_adaptive_mean": avg("delta_tfci_bps_vs_adaptive_csmr"),
                "delta_roi_psnr_vs_adaptive_mean": avg("delta_roi_psnr_vs_adaptive_csmr"),
                "delta_background_psnr_vs_adaptive_mean": avg("delta_background_psnr_vs_adaptive_csmr"),
                "q_mean": avg("q_mean"),
                "packed_l_matches_qmap_pct": avg("packed_l_matches_qmap") * 100.0,
            }
        )
    return summary


def write_report(path: Path, rows: list[dict[str, Any]], groups: list[dict[str, Any]]) -> None:
    lines = [
        "# CSMR preserve-roi",
        "",
        "Esta auditoria mide bitrate real `.tfci` para `preserve-roi`, usando Q-maps C y `lambda = Q/255*0.05`.",
        "",
        "## Resumen mid_roi",
        "",
        "| Preset | Politica | Runs | bps CSMR | Delta bps vs adaptive | Delta ROI PSNR | Delta fondo PSNR | L=Q % |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in groups:
        if row["roi_group"] != "mid_roi":
            continue
        lines.append(
            f"| {row['preset']} | {row['policy']} | {row['runs']} | {fmt(row['tfci_bps_mean'], 5)} | "
            f"{fmt(row['delta_tfci_bps_vs_adaptive_mean'], 5)} | {fmt(row['delta_roi_psnr_vs_adaptive_mean'])} | "
            f"{fmt(row['delta_background_psnr_vs_adaptive_mean'])} | {fmt(row['packed_l_matches_qmap_pct'], 1)} |"
        )
    matches = sum(int(row.get("packed_l_matches_qmap", 0)) for row in rows)
    lines.extend(
        [
            "",
            "## Validacion",
            "",
            f"- Subcasos CSMR: `{len(rows)}`.",
            f"- `packed_l_matches_qmap`: `{matches}/{len(rows)}`.",
            "- Si `preserve-roi` baja menos bitrate que `focus_bgq128`, se interpreta como modo de maxima conservacion de ROI, no como modo principal de ahorro.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = repo_root()
    preserve_dir = resolve(root, args.preserve_checkpoint)
    threshold_dir = resolve(root, args.threshold_checkpoint)
    full_dir = resolve(root, args.full_checkpoint)
    outdir = resolve(root, args.output_dir)
    args.model_path = resolve(root, args.model_path)
    args.sorteny_py = resolve(root, args.sorteny_py)
    args.python_bin = resolve(root, args.python_bin)

    preflight = require_inputs(root, args, preserve_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    progress = outdir / "progress.csv"
    source_rows = selected_source_rows(read_csv(preserve_dir / "preserve_roi_policy_summary.csv"), args.max_cases)
    write_csv(outdir / "csmr_preserve_source_cases.csv", source_rows)

    rows: list[dict[str, Any]] = []
    control_cache: dict[tuple[str, str], dict[str, Any]] = {}
    adaptive_cache: dict[str, dict[str, Any]] = {}
    processed_focus: set[tuple[str, str]] = set()
    for row in source_rows:
        crop = row["crop"]
        preset = row["preset"]
        raw_path = Path(str(row["file"]))
        for policy in CONTROL_POLICIES:
            key = (crop, policy)
            if key not in control_cache:
                print(f"{crop}/_control/{policy}", flush=True)
                control_cache[key] = process_control(root, args, outdir, progress, threshold_dir, full_dir, crop, raw_path, policy)
                rows.append(control_cache[key])
        adaptive = adaptive_cache_for(adaptive_cache, crop, raw_path, control_cache[(crop, "adaptive_s8")])
        if row["policy"] == FOCUS_POLICY:
            key = (crop, preset)
            if key in processed_focus:
                continue
            processed_focus.add(key)
        print(f"{crop}/{preset}/{row['policy']}", flush=True)
        rows.append(process_semantic(root, args, outdir, progress, row, adaptive))

    groups = summarize_groups(rows)
    write_csv(outdir / "csmr_preserve_roi_summary.csv", rows)
    write_json(outdir / "csmr_preserve_roi_summary.json", {"rows": rows, "groups": groups})
    write_csv(outdir / "csmr_preserve_roi_group_summary.csv", groups)
    write_csv(outdir / "csmr_vs_c_proxy.csv", smoke.csmr_vs_c_proxy(rows))
    write_report(outdir / "csmr_preserve_roi_report.md", rows, groups)
    write_json(
        outdir / "run_manifest.json",
        {
            "created_at": now_iso(),
            "preserve_checkpoint": str(preserve_dir),
            "threshold_checkpoint": str(threshold_dir),
            "full_checkpoint": str(full_dir),
            "output_dir": str(outdir),
            "lambda_min": args.lambda_min,
            "lambda_max": args.lambda_max,
            "source_rows": len(source_rows),
            "csmr_rows": len(rows),
            "preflight": preflight,
        },
    )
    print(f"Wrote preserve-roi CSMR audit to {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
