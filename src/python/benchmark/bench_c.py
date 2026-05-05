#!/usr/bin/env python3
"""Benchmark simple del encoder C actual.

Este script mide el binario `sorteny_compressor` con las rutas vigentes del
repo. Los perfiles oficiales de Raspberry Pi 3B+ se obtuvieron con el harness
externo descrito en Progreso 1; este script queda como utilidad local ligera.
"""

import argparse
import os
import subprocess
import time
from statistics import mean, stdev

DEF_BIN = "./sorteny_compressor"
DEF_INPUT = "data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw"
DEF_WEIGHTS = "weights/encoder"
DEF_LAMBDA = 0.1
DEF_MAXL = 0.125
DEF_OUT = "debug_dumps/Y_hat_bench.bin"


def run_once(bin_path, input_raw, lam, out_path, weights, max_lambda, strict=False, threads=0):
    env = os.environ.copy()
    for key in ("DEBUG_DUMP", "DUMP_Y_PRE", "DUMP_Y_FLOAT", "DUMP_M", "DUMP_STAGES", "DUMP_SPECTRAL"):
        env.pop(key, None)
    if strict:
        env["STRICT_PARITY"] = "1"
    else:
        env.pop("STRICT_PARITY", None)
    if threads > 0:
        env["OMP_NUM_THREADS"] = str(threads)

    cmd = [bin_path, input_raw, str(lam), out_path, weights, str(max_lambda)]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, env=env)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description="Benchmark local del encoder C SORTENY.")
    ap.add_argument("--bin", default=DEF_BIN)
    ap.add_argument("--input", default=DEF_INPUT)
    ap.add_argument("--weights", default=DEF_WEIGHTS)
    ap.add_argument("--lambda", dest="lam", type=float, default=DEF_LAMBDA)
    ap.add_argument("--max-lambda", dest="maxl", type=float, default=DEF_MAXL)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0, help="OMP_NUM_THREADS; 0 deja el valor del entorno")
    ap.add_argument("--strict", action="store_true", help="Usar STRICT_PARITY=1")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for _ in range(args.warmup):
        run_once(args.bin, args.input, args.lam, args.out, args.weights, args.maxl, strict=args.strict, threads=args.threads)

    times = [
        run_once(args.bin, args.input, args.lam, args.out, args.weights, args.maxl, strict=args.strict, threads=args.threads)
        for _ in range(args.repeats)
    ]

    avg = mean(times)
    sd = stdev(times) if len(times) > 1 else 0.0
    print("== Benchmark C encoder ==")
    print(f"times={times}")
    print(f"mean={avg:.4f}s std={sd:.4f}s repeats={len(times)}")


if __name__ == "__main__":
    raise SystemExit(main())
