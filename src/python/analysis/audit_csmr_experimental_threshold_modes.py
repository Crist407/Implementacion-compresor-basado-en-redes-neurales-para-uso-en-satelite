#!/usr/bin/env python3
"""
Run CSMR bitrate audit for experimental automatic modes at optimized thresholds.

This script consumes the C-only threshold optimization checkpoint. It does not
run SORTENY C compression/decompression and does not modify the source
checkpoint. It snapshots existing C Q-maps, converts them to SORTENY.py quality
arrays, runs CSMR, and measures the real .tfci bitrate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import audit_csmr_dataset_smoke_bitrate as smoke
import audit_csmr_real_bitrate as csmr_base
import build_global_adaptive_focus_case_study as base


DEFAULT_THRESHOLD_CHECKPOINT = Path("output/checkpoints/20260625_lambda005_threshold_optimization")
DEFAULT_OUTPUT_DIR = Path("output/checkpoints/20260626_lambda005_csmr_experimental_threshold_modes")
DEFAULT_MODEL_PATH = Path("models/SORTENY_Sentinel2_model")
DEFAULT_SORTENY_PY = Path("src/python/reference/SORTENY.py")
PREVIOUS_CSMR_SUMMARY = Path("output/checkpoints/20260624_lambda005_final_evidence_pack/final_csmr_preset_summary.csv")
EXPERIMENTAL_PRESETS = ["low_ndvi", "high_ndvi", "dark_regions", "water_body"]
CONTROL_POLICIES = ["q204", "adaptive_s8"]
FOCUS_POLICY = "focus_bgq128"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-checkpoint", type=Path, default=DEFAULT_THRESHOLD_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sorteny-py", type=Path, default=DEFAULT_SORTENY_PY)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--lambda-min", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    smoke.write_csv(path, rows)


def write_json(path: Path, data: Any) -> None:
    smoke.write_json(path, data)


def finite_float(value: Any) -> float:
    return smoke.finite_float(value)


def fmt(value: Any, digits: int = 4) -> str:
    return smoke.fmt(value, digits)


def threshold_label(value: float) -> str:
    return f"t{str(value).replace('.', 'p')}"


def close_float(a: Any, b: Any, tol: float = 1e-9) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def require_inputs(root: Path, args: argparse.Namespace, threshold_dir: Path) -> dict[str, Any]:
    required = [
        threshold_dir / "selected_thresholds.csv",
        threshold_dir / "threshold_sweep_results.csv",
        threshold_dir / "checkpoint_summary.json",
        threshold_dir / "run_manifest.json",
        resolve(root, args.model_path) / "saved_model.pb",
        resolve(root, args.sorteny_py),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(f"- {p}" for p in missing))
    probe = csmr_base.require_preflight(root, args)
    return {
        "threshold_manifest": read_json(threshold_dir / "run_manifest.json"),
        "threshold_summary": read_json(threshold_dir / "checkpoint_summary.json"),
        "tensorflow_probe": probe,
    }


def selected_thresholds(threshold_dir: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv(threshold_dir / "selected_thresholds.csv")
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        preset = row.get("preset", "")
        if preset not in EXPERIMENTAL_PRESETS:
            continue
        selected[preset] = {
            "preset": preset,
            "selected_threshold": float(row["selected_threshold"]),
            "default_threshold": finite_float(row.get("default_threshold")),
            "decision": row.get("decision", ""),
            "reason": row.get("reason", ""),
            "recommendation_basis": row.get("recommendation_basis", ""),
            "selected_score": finite_float(row.get("selected_score")),
            "selected_mid_runs": finite_float(row.get("selected_mid_runs")),
            "selected_roi_pct_mean": finite_float(row.get("selected_roi_pct_mean")),
            "selected_entropy_delta_mean": finite_float(row.get("selected_entropy_delta_mean")),
            "selected_zlib_delta_mean": finite_float(row.get("selected_zlib_delta_mean")),
        }
    missing = [preset for preset in EXPERIMENTAL_PRESETS if preset not in selected]
    if missing:
        raise RuntimeError(f"Missing selected thresholds for experimental presets: {', '.join(missing)}")
    return selected


def selected_focus_rows(threshold_dir: Path, thresholds: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = read_csv(threshold_dir / "threshold_sweep_results.csv")
    out: list[dict[str, Any]] = []
    for row in rows:
        preset = row.get("preset", "")
        if preset not in thresholds:
            continue
        if row.get("policy") != FOCUS_POLICY or row.get("status") != "valid":
            continue
        selected = thresholds[preset]["selected_threshold"]
        if not close_float(row.get("threshold"), selected):
            continue
        enriched: dict[str, Any] = dict(row)
        enriched["selected_threshold"] = selected
        enriched["threshold_decision"] = thresholds[preset]["decision"]
        enriched["threshold_reason"] = thresholds[preset]["reason"]
        enriched["threshold_recommendation_basis"] = thresholds[preset]["recommendation_basis"]
        enriched["threshold_selected_score"] = thresholds[preset]["selected_score"]
        enriched["summary_tsv"] = str(resolve_threshold_summary_tsv(threshold_dir, enriched))
        out.append(enriched)
    if not out:
        raise RuntimeError("No experimental threshold rows selected from threshold_sweep_results.csv")
    out.sort(key=lambda row: (str(row["preset"]), str(row["crop"])))
    return out


def resolve_threshold_summary_tsv(threshold_dir: Path, row: dict[str, Any]) -> Path:
    """Return the real semantic TSV path for threshold-optimization artifacts.

    The threshold CSV keeps qmap paths under qmaps/focus_bgq128/... and older
    rows may point summary_tsv to qmap_path.with_suffix(".tsv"). The actual TSVs
    produced by the threshold auditor live under summary_tsv/focus_bgq128/...
    so derive the path from preset, threshold label, and crop.
    """
    explicit = Path(str(row.get("summary_tsv", "")))
    if explicit.exists():
        return explicit
    crop = str(row["crop"])
    preset = str(row["preset"])
    label = str(row.get("threshold_label") or threshold_label(float(row["selected_threshold"])))
    derived = threshold_dir / "summary_tsv" / FOCUS_POLICY / preset / label / f"{crop}.tsv"
    if not derived.exists():
        raise FileNotFoundError(
            f"Missing semantic TSV for {crop}/{preset}/{label}. Tried {explicit} and {derived}."
        )
    return derived


def control_case_result_path(threshold_dir: Path, crop: str, policy: str) -> Path:
    return threshold_dir / "cases" / crop / "_controls" / policy / "case_result.json"


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
        "file": str(raw_path),
        "qmap_path": artifacts.get("qmap_path", ""),
        "summary_tsv": "",
        "global_psnr_db": global_metrics.get("psnr_db", ""),
        "global_mse": global_metrics.get("mse", ""),
        "global_mae": global_metrics.get("mae", ""),
        "latent_zero_pct": latent.get("zero_pct", ""),
        "latent_entropy_bits_per_symbol": latent.get("entropy_bits_per_symbol", ""),
        "latent_zlib_bps_per_input_sample": latent.get("zlib_bps_per_input_sample", ""),
    }


def source_qmap_from_case(case_result_path: Path) -> Path:
    data = read_json(case_result_path)
    if data.get("case_status") != "done":
        raise RuntimeError(f"{case_result_path}: case_status is not done")
    value = data.get("artifacts", {}).get("qmap_path")
    if not value:
        raise ValueError(f"{case_result_path}: missing artifacts.qmap_path")
    return Path(str(value))


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


def result_is_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = read_json(path)
    except json.JSONDecodeError:
        return False
    return data.get("case_status") == "done" and isinstance(data.get("row"), dict)


def clean_incomplete_csmr_outputs(outdir: Path, crop: str, preset: str, policy: str) -> None:
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
    threshold_dir: Path,
    outdir: Path,
    progress: Path,
    crop: str,
    raw_path: Path,
    policy: str,
) -> dict[str, Any]:
    case_result_path = control_case_result_path(threshold_dir, crop, policy)
    if not case_result_path.exists():
        raise FileNotFoundError(case_result_path)
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
    row["source_c_case_result_path"] = str(case_result_path)
    row["source_c_qmap_path"] = str(source_qmap)
    row["qmap_snapshot_path"] = str(qmap)
    row["threshold_audit_role"] = "control"
    return row


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


def focus_case_result_path(threshold_dir: Path, row: dict[str, Any]) -> Path:
    crop = str(row["crop"])
    preset = str(row["preset"])
    label = str(row.get("threshold_label") or threshold_label(float(row["selected_threshold"])))
    return threshold_dir / "cases" / crop / preset / label / FOCUS_POLICY / "case_result.json"


def process_focus(
    *,
    root: Path,
    args: argparse.Namespace,
    threshold_dir: Path,
    outdir: Path,
    progress: Path,
    selected: dict[str, Any],
    adaptive: dict[str, Any],
) -> dict[str, Any]:
    crop = str(selected["crop"])
    preset = str(selected["preset"])
    raw_path = Path(str(selected["file"]))
    source_qmap = Path(str(selected["qmap_path"]))
    qmap = snapshot_qmap(source_qmap, outdir / "qmap_snapshots" / crop / preset / f"{FOCUS_POLICY}.bin", args.force)
    summary_tsv = Path(str(selected["summary_tsv"]))
    clean_incomplete_csmr_outputs(outdir, crop, preset, FOCUS_POLICY)
    row = smoke.process_one(
        root=root,
        args=args,
        outdir=outdir,
        progress=progress,
        crop=crop,
        preset=preset,
        policy=FOCUS_POLICY,
        source_kind="experimental_threshold_focus",
        source_row=selected,
        raw_path=raw_path,
        qmap_path=qmap,
        summary_tsv=summary_tsv,
        adaptive_blocks=adaptive["blocks"],
        adaptive_global=adaptive["global"],
        adaptive_tfci_bps=adaptive["tfci_bps"],
    )
    row["threshold"] = selected.get("selected_threshold", selected.get("threshold", ""))
    row["threshold_label"] = selected.get("threshold_label", "")
    row["threshold_decision"] = selected.get("threshold_decision", "")
    row["threshold_reason"] = selected.get("threshold_reason", "")
    row["threshold_recommendation_basis"] = selected.get("threshold_recommendation_basis", "")
    row["threshold_selected_score"] = selected.get("threshold_selected_score", "")
    row["source_c_case_result_path"] = str(focus_case_result_path(threshold_dir, selected))
    row["source_c_qmap_path"] = str(source_qmap)
    row["qmap_snapshot_path"] = str(qmap)
    row["threshold_audit_role"] = "experimental_focus"
    return row


def avg(rows: list[dict[str, Any]], key: str) -> float:
    values = [finite_float(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else math.nan


def count_good_mid(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        if row.get("roi_group") != "mid_roi":
            continue
        if finite_float(row.get("delta_tfci_bps_vs_adaptive_csmr")) >= 0:
            continue
        if finite_float(row.get("delta_roi_psnr_vs_adaptive_csmr")) < -0.05:
            continue
        if finite_float(row.get("delta_background_psnr_vs_adaptive_csmr")) > -0.30:
            continue
        count += 1
    return count


def group_decision(preset: str, rows: list[dict[str, Any]]) -> tuple[str, str]:
    good_mid = count_good_mid(rows)
    delta_bps = avg(rows, "delta_tfci_bps_vs_adaptive_csmr")
    delta_roi = avg(rows, "delta_roi_psnr_vs_adaptive_csmr")
    roi_pct = avg(rows, "roi_pct")
    if good_mid >= 2 and math.isfinite(delta_bps) and delta_bps < 0 and delta_roi >= -0.05:
        return "raspberry_candidate", "CSMR confirms multiple useful mid_roi cases."
    if good_mid == 1 and math.isfinite(delta_bps) and delta_bps < 0:
        return "limited_evidence", "CSMR confirms one useful mid_roi case; evidence is not broad enough."
    if math.isfinite(delta_bps) and delta_bps < 0 and (roi_pct < 5.0 or roi_pct > 40.0):
        return "keep_experimental", "Bitrate improves, but ROI coverage is outside the preferred operational range."
    if math.isfinite(delta_roi) and delta_roi < -0.05:
        return "keep_experimental", "Bitrate/quality tradeoff degrades ROI too much."
    return "discard_for_now", "No robust CSMR evidence over adaptive_s8."


def collect_experimental_group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    focus_rows = [row for row in rows if row.get("policy") == FOCUS_POLICY and row.get("preset") != "_control"]
    by_preset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in focus_rows:
        by_preset[str(row["preset"])].append(row)

    out: list[dict[str, Any]] = []
    for preset in sorted(by_preset):
        items = by_preset[preset]
        decision, reason = group_decision(preset, items)
        out.append(
            {
                "preset": preset,
                "threshold": avg(items, "threshold"),
                "runs": len(items),
                "done_runs": sum(1 for row in items if row.get("case_status") == "done"),
                "mid_roi_runs": sum(1 for row in items if row.get("roi_group") == "mid_roi"),
                "good_mid_roi_runs": count_good_mid(items),
                "roi_pct_mean": avg(items, "roi_pct"),
                "tfci_bps_mean": avg(items, "tfci_bps_per_input_sample"),
                "delta_tfci_bps_vs_adaptive_mean": avg(items, "delta_tfci_bps_vs_adaptive_csmr"),
                "delta_roi_psnr_vs_adaptive_mean": avg(items, "delta_roi_psnr_vs_adaptive_csmr"),
                "delta_background_psnr_vs_adaptive_mean": avg(items, "delta_background_psnr_vs_adaptive_csmr"),
                "c_delta_entropy_vs_adaptive_mean": avg(items, "c_delta_entropy_vs_adaptive"),
                "c_delta_zlib_bps_vs_adaptive_mean": avg(items, "c_delta_zlib_bps_vs_adaptive"),
                "packed_l_matches_qmap_pct": avg(items, "packed_l_matches_qmap") * 100.0,
                "raspberry_decision": decision,
                "decision_reason": reason,
            }
        )
    return out


def previous_comparison_rows(out_groups: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_path = resolve(root, PREVIOUS_CSMR_SUMMARY)
    if previous_path.exists():
        for row in read_csv(previous_path):
            rows.append(
                {
                    "source": "previous_validated_main_modes",
                    "preset": row.get("preset", ""),
                    "runs": row.get("runs", ""),
                    "roi_pct_mean": row.get("roi_pct_mean", ""),
                    "delta_tfci_bps_vs_adaptive_mean": row.get("delta_tfci_bps_vs_adaptive_mean", ""),
                    "delta_roi_psnr_vs_adaptive_mean": row.get("delta_roi_psnr_vs_adaptive_mean", ""),
                    "delta_background_psnr_vs_adaptive_mean": row.get("delta_background_psnr_vs_adaptive_mean", ""),
                    "decision": "reference",
                }
            )
    for row in out_groups:
        rows.append(
            {
                "source": "experimental_threshold_modes",
                "preset": row.get("preset", ""),
                "runs": row.get("runs", ""),
                "roi_pct_mean": row.get("roi_pct_mean", ""),
                "delta_tfci_bps_vs_adaptive_mean": row.get("delta_tfci_bps_vs_adaptive_mean", ""),
                "delta_roi_psnr_vs_adaptive_mean": row.get("delta_roi_psnr_vs_adaptive_mean", ""),
                "delta_background_psnr_vs_adaptive_mean": row.get("delta_background_psnr_vs_adaptive_mean", ""),
                "decision": row.get("raspberry_decision", ""),
            }
        )
    return rows


def write_table_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Tabla CSMR: modos experimentales con thresholds optimizados",
        "",
        "| Preset | Crop | ROI group | ROI % | threshold | tfci bps | delta tfci | delta ROI PSNR | delta fondo PSNR | L=Q | entropia C delta | zlib C delta |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("policy") != FOCUS_POLICY or row.get("preset") == "_control":
            continue
        lines.append(
            f"| {row['preset']} | {row['crop']} | {row.get('roi_group', '')} | {fmt(row.get('roi_pct'))} | "
            f"{fmt(row.get('threshold'))} | {fmt(row.get('tfci_bps_per_input_sample'), 5)} | "
            f"{fmt(row.get('delta_tfci_bps_vs_adaptive_csmr'), 5)} | "
            f"{fmt(row.get('delta_roi_psnr_vs_adaptive_csmr'))} | "
            f"{fmt(row.get('delta_background_psnr_vs_adaptive_csmr'))} | "
            f"{row.get('packed_l_matches_qmap', '')} | {fmt(row.get('c_delta_entropy_vs_adaptive'))} | "
            f"{fmt(row.get('c_delta_zlib_bps_vs_adaptive'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    threshold_dir: Path,
) -> None:
    focus_rows = [row for row in rows if row.get("policy") == FOCUS_POLICY and row.get("preset") != "_control"]
    matches = sum(int(row.get("packed_l_matches_qmap", 0)) for row in rows)
    done = sum(1 for row in rows if row.get("case_status") == "done")
    lines = [
        "# CSMR modos experimentales con thresholds optimizados",
        "",
        f"Generado: {now_iso()}",
        "",
        f"Checkpoint thresholds C-only: `{threshold_dir}`.",
        f"Checkpoint CSMR salida: `{args.output_dir}`.",
        "",
        "Esta auditoria no relanza compresion C. Usa Q-maps ya generados por `sorteny_semantic_qmap`, los convierte a `quality_array` y ejecuta `SORTENY.py`/CSMR con:",
        "",
        "```text",
        f"lambda = Q / 255 * {args.lambda_max}",
        "```",
        "",
        "## Umbrales evaluados",
        "",
        "| Preset | Threshold | Decision previa | Base |",
        "|---|---:|---|---|",
    ]
    for preset in EXPERIMENTAL_PRESETS:
        item = selected[preset]
        lines.append(
            f"| {preset} | {fmt(item['selected_threshold'])} | {item['decision']} | {item['recommendation_basis']} |"
        )
    lines.extend(
        [
            "",
            "## Lectura rapida",
            "",
            f"- Filas CSMR completadas: `{done}/{len(rows)}`.",
            f"- Casos focus experimentales: `{len(focus_rows)}`.",
            f"- `packed_l_matches_qmap`: `{matches}/{len(rows)}`.",
            "",
            "## Resumen por preset experimental",
            "",
            "| Preset | Threshold | Runs | Mid ROI | Good mid ROI | ROI % | Delta tfci | Delta ROI PSNR | Delta fondo PSNR | Decision Raspberry |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in groups:
        lines.append(
            f"| {row['preset']} | {fmt(row['threshold'])} | {row['runs']} | {row['mid_roi_runs']} | "
            f"{row['good_mid_roi_runs']} | {fmt(row['roi_pct_mean'])} | "
            f"{fmt(row['delta_tfci_bps_vs_adaptive_mean'], 5)} | "
            f"{fmt(row['delta_roi_psnr_vs_adaptive_mean'])} | "
            f"{fmt(row['delta_background_psnr_vs_adaptive_mean'])} | {row['raspberry_decision']} |"
        )
    lines.extend(
        [
            "",
            "## Criterio de decision",
            "",
            "- `raspberry_candidate`: al menos dos casos `mid_roi` utiles, menor bitrate CSMR que `adaptive_s8`, ROI no degradada y fondo degradado.",
            "- `limited_evidence`: hay senal positiva, pero depende de un unico caso util.",
            "- `keep_experimental`: mejora algun proxy o bitrate, pero la ROI es demasiado pequena/grande o falta robustez.",
            "- `discard_for_now`: no hay evidencia suficiente para meterlo como modo principal en Raspberry.",
            "",
            "## Interpretacion",
            "",
            "Si un modo reduce `tfci bps` frente a `adaptive_s8`, la mejora ya no es solo proxy de entropia: se traduce en tamano real CSMR. Aun asi, para entrar como modo principal en Raspberry necesitamos que la mejora no dependa de un unico crop y que la ROI sea interpretable.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(outdir: Path, rows: list[dict[str, Any]], proxy_rows: list[dict[str, Any]]) -> None:
    fig_dir = outdir / "figures"
    focus_rows = [row for row in rows if row.get("policy") == FOCUS_POLICY and row.get("preset") != "_control"]
    smoke.svg_bar(
        fig_dir / "real_bps_by_experimental_preset.svg",
        focus_rows,
        "tfci_bps_per_input_sample",
        "Bitrate real CSMR por preset experimental",
    )
    smoke.svg_scatter(
        fig_dir / "roi_psnr_vs_real_bps.svg",
        focus_rows,
        "tfci_bps_per_input_sample",
        "roi_psnr_db",
        "PSNR ROI frente a bitrate real CSMR",
        "tfci bps/sample",
        "PSNR ROI (dB)",
    )
    smoke.svg_scatter(
        fig_dir / "delta_roi_vs_delta_bps.svg",
        focus_rows,
        "delta_tfci_bps_vs_adaptive_csmr",
        "delta_roi_psnr_vs_adaptive_csmr",
        "Delta ROI PSNR vs delta bitrate real",
        "delta tfci bps vs adaptive",
        "delta ROI PSNR CSMR",
    )
    smoke.svg_scatter(
        fig_dir / "proxy_zlib_vs_real_bps_delta.svg",
        proxy_rows,
        "c_delta_zlib_bps_vs_adaptive",
        "csmr_delta_tfci_bps_vs_adaptive",
        "Proxy zlib C vs delta bitrate CSMR",
        "delta zlib proxy C",
        "delta tfci bps CSMR",
    )


def main() -> None:
    args = parse_args()
    root = repo_root()
    threshold_dir = resolve(root, args.threshold_checkpoint)
    outdir = resolve(root, args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    progress = outdir / "progress.csv"
    if args.force and progress.exists():
        progress.unlink()

    preflight = require_inputs(root, args, threshold_dir)
    thresholds = selected_thresholds(threshold_dir)
    selected_rows = selected_focus_rows(threshold_dir, thresholds)

    rows: list[dict[str, Any]] = []
    control_cache: dict[tuple[str, str], dict[str, Any]] = {}
    adaptive_cache: dict[str, dict[str, Any]] = {}

    for idx, selected in enumerate(selected_rows, start=1):
        crop = str(selected["crop"])
        preset = str(selected["preset"])
        raw_path = Path(str(selected["file"]))
        print(f"[{idx}/{len(selected_rows)}] {crop}/{preset}/t={selected['selected_threshold']}", flush=True)

        for policy in CONTROL_POLICIES:
            key = (crop, policy)
            if key not in control_cache:
                control_cache[key] = process_control(
                    root=root,
                    args=args,
                    threshold_dir=threshold_dir,
                    outdir=outdir,
                    progress=progress,
                    crop=crop,
                    raw_path=raw_path,
                    policy=policy,
                )
                rows.append(control_cache[key])

        adaptive = adaptive_cache_for(adaptive_cache, crop, raw_path, control_cache[(crop, "adaptive_s8")])
        focus_row = process_focus(
            root=root,
            args=args,
            threshold_dir=threshold_dir,
            outdir=outdir,
            progress=progress,
            selected=selected,
            adaptive=adaptive,
        )
        rows.append(focus_row)

    groups = collect_experimental_group_summary(rows)
    proxy_rows = smoke.csmr_vs_c_proxy(rows)
    previous = previous_comparison_rows(groups, root)

    write_csv(outdir / "csmr_experimental_threshold_summary.csv", rows)
    write_json(outdir / "csmr_experimental_threshold_summary.json", {"rows": rows, "groups": groups})
    write_csv(outdir / "csmr_experimental_threshold_group_summary.csv", groups)
    write_csv(outdir / "csmr_experimental_vs_previous.csv", previous)
    write_csv(outdir / "csmr_vs_c_proxy.csv", proxy_rows)
    write_table_md(outdir / "csmr_experimental_threshold_table.md", rows)
    if not args.skip_figures:
        write_figures(outdir, rows, proxy_rows)
    write_report(
        outdir / "csmr_experimental_threshold_report.md",
        rows,
        groups,
        thresholds,
        args,
        threshold_dir,
    )
    write_json(
        outdir / "run_manifest.json",
        {
            "generated_at": now_iso(),
            "script": Path(__file__).relative_to(root),
            "threshold_checkpoint": threshold_dir,
            "output_dir": outdir,
            "experimental_presets": EXPERIMENTAL_PRESETS,
            "selected_thresholds": thresholds,
            "selected_focus_cases": len(selected_rows),
            "unique_crops": len({row["crop"] for row in selected_rows}),
            "csmr_rows": len(rows),
            "controls": CONTROL_POLICIES,
            "focus_policy": FOCUS_POLICY,
            "lambda_min": args.lambda_min,
            "lambda_max": args.lambda_max,
            "quality_mapping": f"lambda = Q / 255 * {args.lambda_max}",
            "model_path": args.model_path,
            "sorteny_py": args.sorteny_py,
            "preflight": preflight,
        },
    )
    print(f"Wrote experimental-threshold CSMR audit to {outdir}", flush=True)


if __name__ == "__main__":
    main()
