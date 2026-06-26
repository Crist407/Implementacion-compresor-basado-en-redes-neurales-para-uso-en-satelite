#!/usr/bin/env python3
"""Generate final-report raster figures from existing analysis checkpoints."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "informe_final" / "figures"
REFRAMED = ROOT / "output" / "checkpoints" / "20260631_lambda005_report_reframed_evidence"
FINAL = ROOT / "output" / "checkpoints" / "20260624_lambda005_final_evidence_pack"
EXPERIMENTAL = ROOT / "output" / "checkpoints" / "20260626_lambda005_csmr_experimental_threshold_modes"
PRESERVE = ROOT / "output" / "checkpoints" / "20260628_lambda005_csmr_preserve_roi_policy"
RPI = ROOT / "output" / "checkpoints" / "20260614_raspberry_lambda005_optimization_comparison"
QMAP = ROOT / "output" / "checkpoints" / "20260701_raspberry_lambda005_qmap_cost_report"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close()


def copy_visuals() -> None:
    for source in sorted((REFRAMED / "figures").glob("0*.png")):
        shutil.copy2(source, OUT / source.name)


def policy_bitrate() -> None:
    data = rows(FINAL / "final_csmr_policy_summary.csv")
    labels = [r["policy"].replace("_", "\n") for r in data]
    values = [float(r["tfci_bps_mean"]) for r in data]
    colors = ["#5b6770", "#2f6f8f", "#2f8f62"]
    plt.figure(figsize=(7.4, 4.2))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel("Bitrate real (.tfci), bps")
    plt.ylim(0, max(values) * 1.20)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.3f}",
                 ha="center", va="bottom", fontsize=9)
    plt.grid(axis="y", alpha=0.25)
    save("csmr_policy_bitrate.png")


def preset_savings() -> None:
    data = rows(REFRAMED / "preset_policy_q204_ranking.csv")
    selected = [
        r for r in data
        if r["source_set"] == "main" and r["policy"] == "focus_bgq128"
    ]
    selected.sort(key=lambda r: float(r["saving_pct_vs_q204_mean"]), reverse=True)
    labels = [r["preset"].replace("_", "\n") for r in selected]
    values = [float(r["saving_pct_vs_q204_mean"]) for r in selected]
    runs = [int(float(r["runs"])) for r in selected]
    plt.figure(figsize=(8.4, 4.6))
    bars = plt.bar(labels, values, color="#2f8f62")
    plt.ylabel("Estalvi respecte q204 (%)")
    plt.ylim(0, max(values) * 1.30)
    for bar, value, n in zip(bars, values, runs):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.2,
                 f"{value:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    save("csmr_savings_main_presets.png")


def rd_plots() -> None:
    data = [
        r for r in rows(REFRAMED / "q204_referenced_csmr_summary.csv")
        if r["source_set"] == "main" and r["policy"] == "focus_bgq128"
    ]
    for region, title, out_name in [
        ("roi", "Regió d'interès", "rd_roi_q204_focus.png"),
        ("background", "Fons", "rd_background_q204_focus.png"),
    ]:
        plt.figure(figsize=(7.4, 5.0))
        for idx, r in enumerate(data):
            qx = float(r["q204_tfci_bps"])
            fx = float(r["tfci_bps"])
            qy = float(r[f"same_mask_q204_{region}_psnr_db"])
            fy = float(r[f"same_mask_candidate_{region}_psnr_db"])
            plt.plot([qx, fx], [qy, fy], color="#aab2b8", linewidth=0.7, alpha=0.7)
            if idx == 0:
                plt.scatter(qx, qy, color="#5b6770", marker="o", label="q204", s=24)
                plt.scatter(fx, fy, color="#2f8f62", marker="^", label="focus", s=30)
            else:
                plt.scatter(qx, qy, color="#5b6770", marker="o", s=24)
                plt.scatter(fx, fy, color="#2f8f62", marker="^", s=30)
        plt.xlabel("Bitrate real (.tfci), bps")
        plt.ylabel(f"PSNR {title.lower()} (dB)")
        plt.title(f"Rate-distortion: {title}")
        plt.grid(alpha=0.25)
        plt.legend()
        save(out_name)


def experimental_savings() -> None:
    rank = rows(REFRAMED / "preset_policy_q204_ranking.csv")
    names = ["high_ndvi", "low_ndvi", "dark_regions", "water_body"]
    selected = []
    for name in names:
        candidates = [
            r for r in rank
            if r["source_set"] == "experimental"
            and r["preset"] == name
            and r["policy"] == "focus_bgq128"
        ]
        if candidates:
            selected.append(candidates[0])
    labels = [r["preset"].replace("_", "\n") for r in selected]
    values = [float(r["saving_pct_vs_q204_mean"]) for r in selected]
    runs = [int(float(r["runs"])) for r in selected]
    plt.figure(figsize=(7.2, 4.4))
    bars = plt.bar(labels, values, color=["#2f8f62", "#d6a238", "#8a6f52", "#3c7fa3"])
    plt.ylabel("Estalvi respecte q204 (%)")
    plt.ylim(0, max(values) * 1.35)
    for bar, value, n in zip(bars, values, runs):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.2,
                 f"{value:.1f}%\n(n={n})", ha="center", fontsize=9)
    plt.grid(axis="y", alpha=0.25)
    save("csmr_experimental_savings.png")


def preserve_tradeoff() -> None:
    rank = rows(REFRAMED / "preset_policy_q204_ranking.csv")
    selected = [
        r for r in rank
        if r["source_set"] == "preserve"
        and r["preset"] == "cloud_avoid"
        and r["policy"] in {
            "focus_bgq128",
            "preserve_roi_q240_bgq128",
            "preserve_roi_q255_bgq128",
        }
    ]
    labels = {
        "focus_bgq128": "focus",
        "preserve_roi_q240_bgq128": "preserve Q240",
        "preserve_roi_q255_bgq128": "preserve Q255",
    }
    colors = {"focus_bgq128": "#2f8f62", "preserve_roi_q240_bgq128": "#d6a238",
              "preserve_roi_q255_bgq128": "#b34b4b"}
    plt.figure(figsize=(7.2, 4.8))
    for r in selected:
        x = float(r["saving_pct_vs_q204_mean"])
        y = float(r["delta_roi_psnr_vs_q204_same_mask_mean"])
        policy = r["policy"]
        plt.scatter(x, y, s=75, color=colors[policy], label=labels[policy])
        plt.annotate(labels[policy], (x, y), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Estalvi respecte q204 (%)")
    plt.ylabel(r"$\Delta$PSNR ROI respecte q204 (dB)")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    save("preserve_roi_tradeoff.png")


def raspberry_speedup() -> None:
    data = rows(RPI / "raspberry_optimization_speedup.csv")
    # The file contains paired rows; aggregate by case with arithmetic mean.
    grouped: dict[str, list[float]] = {}
    for r in data:
        case = r.get("case") or r.get("policy") or r.get("mode") or "case"
        value = r.get("total_speedup")
        if value:
            grouped.setdefault(case, []).append(float(value))
    labels = list(grouped)
    values = [sum(grouped[k]) / len(grouped[k]) for k in labels]
    order = sorted(range(len(labels)), key=lambda i: values[i], reverse=True)
    labels = [labels[i].replace("_focus_bgq128", "").replace("_", "\n") for i in order]
    values = [values[i] for i in order]
    plt.figure(figsize=(9.0, 4.8))
    plt.bar(labels, values, color="#2f6f8f")
    plt.axhline(1.0, color="#444444", linewidth=1)
    plt.ylabel("Speedup total (baseline / optimitzat)")
    plt.xticks(fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    save("raspberry_speedup_by_mode.png")


def qmap_cost() -> None:
    data = rows(QMAP / "raspberry_qmap_cost_by_case.csv")
    labels = [r["case"].replace("_focus_bgq128", "").replace("_", "\n") for r in data]
    values = [float(r["qmap_elapsed_mean_s"]) for r in data]
    plt.figure(figsize=(9.0, 4.8))
    bars = plt.bar(labels, values, color="#735d8c")
    plt.ylabel("Temps de generació Q-map (s)")
    plt.xticks(fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.004,
                 f"{value:.3f}", ha="center", fontsize=7, rotation=90)
    plt.ylim(0, max(values) * 1.35)
    save("raspberry_qmap_cost.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    copy_visuals()
    policy_bitrate()
    preset_savings()
    rd_plots()
    experimental_savings()
    preserve_tradeoff()
    raspberry_speedup()
    qmap_cost()
    print(f"Wrote final-report figures to {OUT}")


if __name__ == "__main__":
    main()
