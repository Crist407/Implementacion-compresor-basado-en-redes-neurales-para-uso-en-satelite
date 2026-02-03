#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
from typing import Tuple, Dict, Any, List

MODEL_DIR = "models/SORTENY_Sentinel2_model"
WEIGHTS_DIR = "weights/pesos_bin"
INDEX_TSV = os.path.join(WEIGHTS_DIR, "weights_index.tsv")

np.set_printoptions(suppress=True, linewidth=120)

# Some Lambda layers in the SavedModel refer to this global variable.
# Keep it consistent with the training/export code.
bit_length = 16


def parse_shape(shape_str: str) -> Tuple[int, ...]:
    # shapes like "5x5x1x128" or "192x3072" or "1x192"
    parts = shape_str.strip().split("x")
    return tuple(int(p) for p in parts if p)


def load_index() -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(INDEX_TSV):
        raise FileNotFoundError(f"No existe índice TSV: {INDEX_TSV}")
    with open(INDEX_TSV, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 5:
                continue
            filename, dtype, size_bytes, shape, sha256 = cols[:5]
            idx[filename] = {
                "dtype": dtype,
                "size_bytes": int(size_bytes),
                "shape": parse_shape(shape),
                "sha256": sha256,
                "path": os.path.join(WEIGHTS_DIR, filename),
            }
    return idx


def load_bin(path: str, shape: Tuple[int, ...]) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"Tamaño inesperado en {path}: {arr.size} vs {expected} para shape {shape}")
    return arr.reshape(shape)


def max_mean_abs(a: np.ndarray) -> Tuple[float, float]:
    diff = np.abs(a)
    return float(diff.max()), float(diff.mean())


def compare_dense(name_tf: str, tf_w: np.ndarray, bin_w: np.ndarray) -> Dict[str, Any]:
    # TF Dense kernel is (in_dim, out_dim)
    results = {}
    diff_direct = tf_w - bin_w
    results["direct_max"], results["direct_mean"] = max_mean_abs(diff_direct)
    if bin_w.shape[::-1] == tf_w.shape:
        diff_T = tf_w - bin_w.T
        results["transpose_max"], results["transpose_mean"] = max_mean_abs(diff_T)
    else:
        results["transpose_max"], results["transpose_mean"] = float("inf"), float("inf")
    # Choose best
    best = min(("direct", results["direct_max"]), ("transpose", results["transpose_max"]), key=lambda x: x[1])
    results["best"] = best[0]
    results["name"] = name_tf
    results["tf_shape"] = list(tf_w.shape)
    results["bin_shape"] = list(bin_w.shape)
    return results


def compare_conv(name_tf: str, tf_w: np.ndarray, bin_w: np.ndarray) -> Dict[str, Any]:
    # TF SignalConv2D kernel: (H, W, in_ch, out_ch)
    results = {}
    def _rec(label: str, candidate: np.ndarray):
        if candidate.shape != tf_w.shape:
            results[f"{label}_max"] = float("inf")
            results[f"{label}_mean"] = float("inf")
            return
        d = tf_w - candidate
        mx, mn = max_mean_abs(d)
        results[f"{label}_max"] = mx
        results[f"{label}_mean"] = mn

    # candidates
    _rec("direct", bin_w)
    # Try alternative orientations; record only if shapes align
    swapped = np.transpose(bin_w, (0, 1, 3, 2))  # swap in/out
    _rec("swap_in_out", swapped)
    flipped = np.flip(bin_w, axis=(0, 1))       # flip spatial
    _rec("flip_hw", flipped)
    flipped_swapped = np.flip(swapped, axis=(0, 1))
    _rec("flip_hw_and_swap", flipped_swapped)

    # choose best by max error
    best_key = min([(k, v) for k, v in results.items() if k.endswith("_max")], key=lambda kv: kv[1])[0]
    results["best"] = best_key.replace("_max", "")
    results["name"] = name_tf
    results["tf_shape"] = list(tf_w.shape)
    results["bin_shape"] = list(bin_w.shape)
    return results

def get_gdn_epsilons() -> Dict[str, float]:
    eps: Dict[str, float] = {}
    try:
        model = tf.keras.models.load_model(MODEL_DIR, compile=False)
        analysis = getattr(model, "analysis_transform", None) or None
        if analysis is not None and getattr(analysis, "layers", None) and len(analysis.layers) >= 5:
            for i in range(3):
                layer = analysis.layers[1 + i]
                gdn = getattr(layer, "activation", None)
                val = getattr(gdn, "epsilon", None)
                if val is not None:
                    eps[f"gdn_{i}_epsilon"] = float(val)
    except Exception:
        pass
    return eps


def load_model_and_refs() -> Dict[str, np.ndarray]:
    model = tf.keras.models.load_model(MODEL_DIR, compile=False)

    refs: Dict[str, np.ndarray] = {}
    # Spectral Dense
    spec = getattr(model, "spectral_analysis_transform", None)
    if spec is not None and getattr(spec, "layers", None):
        dense = next((l for l in spec.layers if isinstance(l, tf.keras.layers.Dense)), None)
        if dense is not None:
            refs["spectral_analysis_kernel.bin"] = dense.kernel.numpy()

    # Analysis convs
    analysis = getattr(model, "analysis_transform", None) or None
    if analysis is not None and getattr(analysis, "layers", None) and len(analysis.layers) >= 5:
        for i, base in enumerate(["analysis_conv_0_kernel.bin", "analysis_conv_1_kernel.bin", "analysis_conv_2_kernel.bin", "analysis_conv_3_kernel.bin"]):
            layer = analysis.layers[1 + i]  # 0: Lambda, 1..4: convs
            k = getattr(layer, "kernel", None)
            if k is not None:
                refs[base] = k.numpy()

    # Modulating dense
    mod = getattr(model, "modulating_transform", None)
    if mod is not None and getattr(mod, "layers", None):
        # layers: [Lambda, Dense(192), Dense(3072)]
        d0 = mod.layers[1]
        d1 = mod.layers[2]
        if isinstance(d0, tf.keras.layers.Dense):
            refs["mod_dense_0_kernel.bin"] = d0.kernel.numpy()
        if isinstance(d1, tf.keras.layers.Dense):
            refs["mod_dense_1_kernel.bin"] = d1.kernel.numpy()

    return refs


def main():
    idx = load_index()
    refs = load_model_and_refs()

    targets: List[str] = [
        "mod_dense_0_kernel.bin",
        "mod_dense_1_kernel.bin",
        "analysis_conv_0_kernel.bin",
        "analysis_conv_1_kernel.bin",
        "analysis_conv_2_kernel.bin",
        "analysis_conv_3_kernel.bin",
        "spectral_analysis_kernel.bin",
    ]

    print("\n== Peso a peso: comparación de orientaciones ==\n")
    summary = {}
    for fname in targets:
        ref = refs.get(fname)
        meta = idx.get(fname)
        if ref is None:
            print(f"[WARN] No se encontró referencia TF para {fname}")
            continue
        if meta is None:
            print(f"[WARN] No hay metadatos TSV para {fname}")
            continue
        bin_w = load_bin(meta["path"], meta["shape"])

        if ref.ndim == 2:
            res = compare_dense(fname, ref, bin_w)
        elif ref.ndim == 4:
            res = compare_conv(fname, ref, bin_w)
        else:
            print(f"[INFO] Omitiendo {fname}: ndim={ref.ndim}")
            continue

        summary[fname] = res
        print(f"{fname} -> best={res['best']} | direct_max={res['direct_max']:.6g} mean={res['direct_mean']:.6g}", end="")
        # print alternative if present
        if 'transpose_max' in res:
            print(f" | transpose_max={res['transpose_max']:.6g} mean={res['transpose_mean']:.6g}", end="")
        if 'swap_in_out_max' in res:
            print(f" | swap_in_out_max={res['swap_in_out_max']:.6g} mean={res['swap_in_out_mean']:.6g}", end="")
        if 'flip_hw_max' in res:
            print(f" | flip_hw_max={res['flip_hw_max']:.6g} mean={res['flip_hw_mean']:.6g}", end="")
        if 'flip_hw_and_swap_max' in res:
            print(f" | flip+swap_max={res['flip_hw_and_swap_max']:.6g} mean={res['flip_hw_and_swap_mean']:.6g}", end="")
        print()

    # Recommendation summary
    print("\n== Recomendaciones ==")
    for fname, res in summary.items():
        if res['best'] in ("direct",):
            action = "Orientación OK (no cambiar)"
        elif res['best'] in ("transpose", "swap_in_out"):
            action = "Transponer ejes de entrada/salida (ajustar loader o loops)"
        elif res['best'] in ("flip_hw", "flip_hw_and_swap"):
            action = "Posible correlación/convolución: invertir H/W (revise apply_conv2d)"
        else:
            action = "Revisar manualmente"
        print(f"- {fname}: {action} | mejor={res['best']}")

    # Save JSON summary for traceability
    with open("weight_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nResumen guardado en weight_comparison_summary.json")

    # Print GDN epsilons (useful for matching C implementation)
    eps = get_gdn_epsilons()
    if eps:
        print("\n== GDN epsilons (Analysis Transform) ==")
        for k, v in eps.items():
            print(f"- {k}: {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
