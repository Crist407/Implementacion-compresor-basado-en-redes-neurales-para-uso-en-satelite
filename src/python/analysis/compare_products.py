#!/usr/bin/env python3
import os
import argparse
import numpy as np

def load_planar(path: str, shape_flat: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != shape_flat:
        raise ValueError(f"Unexpected size for {path}: got {arr.size}, expected {shape_flat}")
    return arr


def stats(name: str, a: np.ndarray, b: np.ndarray) -> None:
    d = np.abs(a - b)
    print(f"{name}: max={d.max():.6g} mean={d.mean():.6g}")


def per_channel_diffs(name: str, a: np.ndarray, b: np.ndarray, C: int, H: int, W: int, topk: int = 5) -> None:
    A = a.reshape(C, H, W)
    B = b.reshape(C, H, W)
    diffs = np.max(np.abs(A - B), axis=(1, 2))
    ord_idx = np.argsort(-diffs)
    print(f"Top-{topk} channels with largest max-abs-diff ({name}):")
    for i in ord_idx[:topk]:
        print(f"  ch {i:4d}: max={diffs[i]:.6g}")


def main():
    p = argparse.ArgumentParser(description="Compare C vs Python products and outputs")
    p.add_argument("--C", dest="c_dir", default=".", help="Directory with C dumps (debug_dumps)")
    p.add_argument("--PY", dest="py_dir", default=".", help="Directory with Python dumps (debug_dumps)")
    p.add_argument("--height", type=int, default=32)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--channels", type=int, default=3072)
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    # Cargamos dimensiones pasadas por argumentos
    H, W, C = args.height, args.width, args.channels
    N = H * W * C

    # Paths de archivos
    y_pre_c = os.path.join(args.c_dir, "Y_pre_c.bin")
    m_c = os.path.join(args.c_dir, "M_c.bin")
    y_float_c = os.path.join(args.c_dir, "Y_float_c.bin")
    y_hat_c = os.path.join(args.c_dir, "Y_hat_c_even.bin")

    conv3_py = os.path.join(args.py_dir, "conv3_py.bin")
    m_py = os.path.join(args.py_dir, "M_py.bin")
    y_float_py = os.path.join(args.py_dir, "Y_float_py.bin")
    y_hat_py = os.path.join(args.py_dir, "python_ground_truth.bin")

    # Load
    YpreC = load_planar(y_pre_c, N)
    MC = np.fromfile(m_c, dtype=np.float32)
    YfloatC = load_planar(y_float_c, N)
    YhatC = load_planar(y_hat_c, N)

    Conv3PY = load_planar(conv3_py, N)
    MPY = np.fromfile(m_py, dtype=np.float32)
    YfloatPY = load_planar(y_float_py, N)
    YhatPY = load_planar(y_hat_py, N)

    # Recompute products to double-check broadcasting
    MC_r = MC.reshape(C, 1, 1)
    MPY_r = MPY.reshape(C, 1, 1)
    prodC = (YpreC.reshape(C, H, W) * MC_r).reshape(-1)
    prodPY = (Conv3PY.reshape(C, H, W) * MPY_r).reshape(-1)

    # Stats
    print("== Global diffs ==")
    stats("Y_pre*C vs Y_float_c", prodC, YfloatC)
    stats("conv3*PY vs Y_float_py", prodPY, YfloatPY)
    stats("Y_float_c vs Y_float_py", YfloatC, YfloatPY)
    stats("Y_hat_c_even vs py_hat", YhatC, YhatPY)

    print("\n== Per-channel (max abs diff)==")
    per_channel_diffs("Y_float C vs PY", YfloatC, YfloatPY, C, H, W, args.topk)
    per_channel_diffs("Y_hat C vs PY", YhatC, YhatPY, C, H, W, args.topk)

if __name__ == "__main__":
    main()
