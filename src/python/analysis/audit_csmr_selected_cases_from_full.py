#!/usr/bin/env python3
"""
Run CSMR bitrate audit for the selected cases from the lambda005 full run.

The selection checkpoint is produced by build_csmr_case_selection_from_full.py.
This script does not run SORTENY C compression/decompression and does not
modify the full C-only checkpoint. It snapshots existing C Q-maps, converts
them to CSMR quality arrays, runs SORTENY.py, and measures real .tfci bitrate.
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


DEFAULT_SELECTION_CHECKPOINT = Path("output/checkpoints/20260620_lambda005_full_csmr_case_selection")
DEFAULT_OUTPUT_DIR = Path("output/checkpoints/20260621_lambda005_csmr_selected_cases_bitrate")
DEFAULT_MODEL_PATH = Path("models/SORTENY_Sentinel2_model")
DEFAULT_SORTENY_PY = Path("src/python/reference/SORTENY.py")
CONTROL_POLICIES = ["q204", "adaptive_s8"]
FOCUS_POLICY = "focus_bgq128"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-checkpoint", type=Path, default=DEFAULT_SELECTION_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sorteny-py", type=Path, default=DEFAULT_SORTENY_PY)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--lambda-min", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-partial-selection", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    smoke.write_json(path, data)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    smoke.write_csv(path, rows)


def finite_float(value: Any) -> float:
    return smoke.finite_float(value)


def fmt(value: Any, digits: int = 4) -> str:
    return smoke.fmt(value, digits)


def require_inputs(root: Path, args: argparse.Namespace, selection_dir: Path) -> dict[str, Any]:
    selection_csv = selection_dir / "csmr_selected_cases.csv"
    selection_manifest = selection_dir / "run_manifest.json"
    required = [
        selection_csv,
        selection_manifest,
        resolve(root, args.model_path) / "saved_model.pb",
        resolve(root, args.sorteny_py),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(f"- {p}" for p in missing))

    selection_meta = read_json(selection_manifest)
    if selection_meta.get("partial") and not args.allow_partial_selection:
        raise RuntimeError(
            "The CSMR case selection is marked partial=true. "
            "Use --allow-partial-selection for provisional CSMR results, or regenerate the selection after the full run finishes."
        )

    probe = csmr_base.require_preflight(root, args)
    return {
        "selection_manifest": selection_meta,
        "tensorflow_probe": probe,
    }


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


def source_qmap_from_case(case_result_path: Path) -> Path:
    data = read_json(case_result_path)
    if data.get("case_status") != "done":
        raise RuntimeError(f"{case_result_path}: case_status is not done")
    value = data.get("artifacts", {}).get("qmap_path")
    if not value:
        raise ValueError(f"{case_result_path}: missing artifacts.qmap_path")
    return Path(str(value))


def case_selection_rows(selection_dir: Path) -> list[dict[str, str]]:
    rows = read_csv(selection_dir / "csmr_selected_cases.csv")
    if not rows:
        raise RuntimeError(f"No selected cases found in {selection_dir / 'csmr_selected_cases.csv'}")
    return rows


def adaptive_cache_for(
    cache: dict[str, dict[str, Any]], crop: str, raw_path: Path, adaptive_row: dict[str, Any]
) -> dict[str, Any]:
    if crop in cache:
        return cache[crop]
    recon_path = Path(str(adaptive_row["reconstruction_path"]))
    original = base.load_raw(raw_path)
    reconstruction = base.load_raw(recon_path)
    cache[crop] = {
        "blocks": base.block_metrics(original, reconstruction),
        "global": base.metrics_for_arrays(original, reconstruction),
        "tfci_bps": float(adaptive_row["tfci_bps_per_input_sample"]),
    }
    return cache[crop]


def add_traceability(row: dict[str, Any], selected: dict[str, str], source_qmap: Path) -> dict[str, Any]:
    row["selection_reason"] = selected.get("selection_reason", "")
    row["selection_score"] = selected.get("selection_score", "")
    row["source_c_qmap_path"] = source_qmap
    row["source_c_selection_row"] = selected.get("semantic_case_result_path", "")
    row["selection_partial_warning"] = selected.get("selection_partial_warning", "")
    return row


def result_is_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = read_json(path)
    except json.JSONDecodeError:
        return False
    return data.get("case_status") == "done" and isinstance(data.get("row"), dict)


def clean_incomplete_csmr_outputs(outdir: Path, crop: str, preset: str, policy: str) -> None:
    """Remove outputs that can be left half-written by an interrupted CSMR run."""
    result_path = smoke.case_result_path(outdir, crop, preset, policy)
    if result_is_done(result_path):
        return
    work_dir = smoke.case_work_dir(outdir, crop, preset, policy)
    input_copy = work_dir / "input.8_512_512_2_1.raw"
    tfci = Path(str(input_copy) + ".tfci")
    recon = Path(str(tfci) + ".raw")
    tfci.unlink(missing_ok=True)
    recon.unlink(missing_ok=True)


def process_control(
    *,
    root: Path,
    args: argparse.Namespace,
    outdir: Path,
    progress: Path,
    selected: dict[str, str],
    crop: str,
    policy: str,
) -> dict[str, Any]:
    raw_path = Path(selected["raw_path"])
    case_key = "adaptive_case_result_path" if policy == "adaptive_s8" else f"{policy}_case_result_path"
    case_result_path = Path(selected[case_key])
    case_result = read_json(case_result_path)
    source_row = control_source_row(case_result, raw_path)
    source_qmap = source_qmap_from_case(case_result_path)
    qmap = snapshot_qmap(source_qmap, outdir / "qmap_snapshots" / crop / "_controls" / f"{policy}.bin", args.force)
    clean_incomplete_csmr_outputs(outdir, crop, "_control", policy)
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
    row["source_c_case_result_path"] = case_result_path
    row["source_c_qmap_path"] = source_qmap
    row["qmap_snapshot_path"] = qmap
    return row


def process_focus(
    *,
    root: Path,
    args: argparse.Namespace,
    outdir: Path,
    progress: Path,
    selected: dict[str, str],
    adaptive: dict[str, Any],
) -> dict[str, Any]:
    crop = selected["crop"]
    preset = selected["preset"]
    raw_path = Path(selected["raw_path"])
    source_qmap = Path(selected["qmap_path"])
    qmap = snapshot_qmap(source_qmap, outdir / "qmap_snapshots" / crop / preset / f"{FOCUS_POLICY}.bin", args.force)
    summary_tsv = Path(selected["summary_tsv"])
    clean_incomplete_csmr_outputs(outdir, crop, preset, FOCUS_POLICY)
    row = smoke.process_one(
        root=root,
        args=args,
        outdir=outdir,
        progress=progress,
        crop=crop,
        preset=preset,
        policy=FOCUS_POLICY,
        source_kind="semantic_focus",
        source_row=selected,
        raw_path=raw_path,
        qmap_path=qmap,
        summary_tsv=summary_tsv,
        adaptive_blocks=adaptive["blocks"],
        adaptive_global=adaptive["global"],
        adaptive_tfci_bps=adaptive["tfci_bps"],
    )
    row["source_c_case_result_path"] = selected.get("semantic_case_result_path", "")
    row["source_c_qmap_path"] = source_qmap
    row["qmap_snapshot_path"] = qmap
    row["selection_reason"] = selected.get("selection_reason", "")
    row["selection_score"] = selected.get("selection_score", "")
    return row


def collect_selection_group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return smoke.collect_group_summary(rows)


def selected_proxy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return smoke.csmr_vs_c_proxy(rows)


def avg(rows: list[dict[str, Any]], key: str) -> float:
    values = [finite_float(row.get(key)) for row in rows]
    values = [v for v in values if math.isfinite(v)]
    return float(np.mean(values)) if values else math.nan


def write_table_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Tabla CSMR: casos seleccionados del full lambda005",
        "",
        "| Crop | Preset | Policy | Reason | ROI % | PSNR global | PSNR ROI | PSNR fondo | tfci bps | Delta tfci | L=Q | Entropia C | zlib C |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['crop']} | {row['preset']} | {row['policy']} | {row.get('selection_reason', '')} | "
            f"{fmt(row.get('roi_pct'))} | {fmt(row.get('global_psnr_db'))} | {fmt(row.get('roi_psnr_db'))} | "
            f"{fmt(row.get('background_psnr_db'))} | {fmt(row.get('tfci_bps_per_input_sample'), 5)} | "
            f"{fmt(row.get('delta_tfci_bps_vs_adaptive_csmr'), 5)} | {row.get('packed_l_matches_qmap', '')} | "
            f"{fmt(row.get('c_latent_entropy_bits_per_symbol'))} | {fmt(row.get('c_latent_zlib_bps_per_input_sample'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    selection_partial: bool,
    args: argparse.Namespace,
    selection_dir: Path,
) -> None:
    focus_rows = [row for row in rows if row["policy"] == FOCUS_POLICY and row["preset"] != "_control"]
    control_rows = [row for row in rows if row["preset"] == "_control"]
    matches = sum(int(row.get("packed_l_matches_qmap", 0)) for row in rows)
    provisional = "Si" if selection_partial else "No"
    lines = [
        "# CSMR selected cases from full lambda005",
        "",
        f"Generated: {now_iso()}",
        "",
        f"Selection checkpoint: `{selection_dir}`.",
        f"Output checkpoint: `{args.output_dir}`.",
        f"Seleccion parcial/provisional: `{provisional}`.",
        "",
        "Esta auditoria no relanza compresion C. Usa Q-maps ya generados por C, los copia al checkpoint CSMR y ejecuta `SORTENY.py` con:",
        "",
        "```text",
        f"lambda = Q / 255 * {args.lambda_max}",
        "```",
        "",
        "## Lectura rapida",
        "",
        f"- Filas CSMR completadas: `{sum(1 for row in rows if row.get('case_status') == 'done')}/{len(rows)}`.",
        f"- Casos focus seleccionados: `{len(focus_rows)}`.",
        f"- Controles ejecutados: `{len(control_rows)}`.",
        f"- `packed_l_matches_qmap`: `{matches}/{len(rows)}`.",
        "",
        "## Promedios",
        "",
        "| Alcance | tfci bps medio | PSNR global | Delta tfci vs adaptive | Delta ROI PSNR | Delta fondo PSNR |",
        "|---|---:|---:|---:|---:|---:|",
        f"| controles | {fmt(avg(control_rows, 'tfci_bps_per_input_sample'), 5)} | {fmt(avg(control_rows, 'global_psnr_db'))} |  |  |  |",
        f"| focus_bgq128 seleccionado | {fmt(avg(focus_rows, 'tfci_bps_per_input_sample'), 5)} | {fmt(avg(focus_rows, 'global_psnr_db'))} | {fmt(avg(focus_rows, 'delta_tfci_bps_vs_adaptive_csmr'), 5)} | {fmt(avg(focus_rows, 'delta_roi_psnr_vs_adaptive_csmr'))} | {fmt(avg(focus_rows, 'delta_background_psnr_vs_adaptive_csmr'))} |",
        "",
        "## Resumen por preset",
        "",
        "| Preset | ROI group | runs | ROI % | Delta tfci | Delta ROI PSNR | Delta fondo PSNR | Delta entropia C | Delta zlib C | L=Q % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in groups:
        lines.append(
            f"| {row['preset']} | {row['roi_group']} | {row['runs']} | {fmt(row['roi_pct_mean'])} | "
            f"{fmt(row['delta_tfci_bps_vs_adaptive_mean'], 5)} | {fmt(row['delta_roi_psnr_vs_adaptive_mean'])} | "
            f"{fmt(row['delta_background_psnr_vs_adaptive_mean'])} | {fmt(row['c_delta_entropy_vs_adaptive_mean'])} | "
            f"{fmt(row['c_delta_zlib_bps_vs_adaptive_mean'])} | {fmt(row['packed_l_matches_qmap_pct'], 1)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretacion",
            "",
            "- La metrica principal nueva es `tfci bps`: bitrate real del codificador CSMR.",
            "- Las columnas de entropia/zlib siguen siendo proxies de la ruta C y se usan para comprobar si correlacionan con CSMR.",
            "- Si `delta_tfci_bps_vs_adaptive_csmr < 0`, el preset `focus_bgq128` produce menos bitrate real que `adaptive_s8` para el mismo crop.",
            "- Si la seleccion era parcial, estos datos son utiles para avanzar, pero no deben presentarse como conclusion final hasta regenerar la seleccion sin `--allow-partial`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(outdir: Path, rows: list[dict[str, Any]], proxy_rows: list[dict[str, Any]]) -> None:
    fig_dir = outdir / "figures"
    smoke.svg_bar(fig_dir / "real_bps_by_policy.svg", rows, "tfci_bps_per_input_sample", "Bitrate real CSMR por caso/politica")
    smoke.svg_scatter(
        fig_dir / "proxy_entropy_vs_real_bps.svg",
        rows,
        "c_latent_entropy_bits_per_symbol",
        "tfci_bps_per_input_sample",
        "Proxy entropia C vs bitrate real CSMR",
        "entropia latente C (bits/simbolo)",
        "tfci bps/sample",
    )
    smoke.svg_scatter(
        fig_dir / "psnr_roi_vs_real_bps.svg",
        proxy_rows,
        "csmr_delta_tfci_bps_vs_adaptive",
        "csmr_delta_roi_psnr_vs_adaptive",
        "Delta ROI PSNR vs delta bitrate real",
        "delta tfci bps vs adaptive",
        "delta ROI PSNR CSMR",
    )
    focus_mid = [row for row in proxy_rows if row.get("roi_group") == "mid_roi"]
    smoke.svg_bar(
        fig_dir / "adaptive_vs_focus_bps_delta.svg",
        focus_mid,
        "csmr_delta_tfci_bps_vs_adaptive",
        "Mid ROI: bitrate focus_bgq128 - adaptive_s8",
    )


def main() -> None:
    args = parse_args()
    root = repo_root()
    selection_dir = resolve(root, args.selection_checkpoint)
    outdir = resolve(root, args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    progress = outdir / "progress.csv"
    if args.force and progress.exists():
        progress.unlink()

    preflight = require_inputs(root, args, selection_dir)
    selection_manifest = preflight["selection_manifest"]
    selection_partial = bool(selection_manifest.get("partial"))
    selected_rows = case_selection_rows(selection_dir)
    if selection_partial:
        for row in selected_rows:
            row["selection_partial_warning"] = "partial_selection"

    rows: list[dict[str, Any]] = []
    adaptive_cache: dict[str, dict[str, Any]] = {}

    for idx, selected in enumerate(selected_rows, start=1):
        crop = selected["crop"]
        print(f"[{idx}/{len(selected_rows)}] {crop}/{selected['preset']}", flush=True)
        raw_path = Path(selected["raw_path"])
        control_outputs: dict[str, dict[str, Any]] = {}
        for policy in CONTROL_POLICIES:
            row = process_control(
                root=root,
                args=args,
                outdir=outdir,
                progress=progress,
                selected=selected,
                crop=crop,
                policy=policy,
            )
            rows.append(row)
            control_outputs[policy] = row
        adaptive = adaptive_cache_for(adaptive_cache, crop, raw_path, control_outputs["adaptive_s8"])
        focus_row = process_focus(
            root=root,
            args=args,
            outdir=outdir,
            progress=progress,
            selected=selected,
            adaptive=adaptive,
        )
        rows.append(focus_row)

    groups = collect_selection_group_summary(rows)
    proxy_rows = selected_proxy_rows(rows)
    write_csv(outdir / "csmr_selected_bitrate_summary.csv", rows)
    write_json(outdir / "csmr_selected_bitrate_summary.json", {"rows": rows})
    write_csv(outdir / "csmr_vs_c_proxy.csv", proxy_rows)
    write_table_md(outdir / "csmr_selected_bitrate_table.md", rows)

    if not args.skip_figures:
        write_figures(outdir, rows, proxy_rows)

    write_report(outdir / "csmr_selected_bitrate_report.md", rows, groups, selection_partial, args, selection_dir)
    write_json(
        outdir / "run_manifest.json",
        {
            "generated_at": now_iso(),
            "script": Path(__file__).relative_to(root),
            "selection_checkpoint": selection_dir,
            "output_dir": outdir,
            "selection_partial": selection_partial,
            "allow_partial_selection": bool(args.allow_partial_selection),
            "lambda_min": args.lambda_min,
            "lambda_max": args.lambda_max,
            "quality_mapping": f"lambda = Q / 255 * {args.lambda_max}",
            "model_path": args.model_path,
            "sorteny_py": args.sorteny_py,
            "controls": CONTROL_POLICIES,
            "focus_policy": FOCUS_POLICY,
            "selected_cases": len(selected_rows),
            "csmr_rows": len(rows),
            "preflight": preflight,
        },
    )
    print(f"Wrote selected-case CSMR bitrate audit to {outdir}", flush=True)


if __name__ == "__main__":
    main()
