#!/usr/bin/env python3
import argparse
import os
import subprocess
import time
from statistics import mean, stdev

DEF_BIN = "./sorteny_compressor"
DEF_INPUT = "data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw"
DEF_WEIGHTS = "weights/pesos_bin_minimal"
DEF_LAMBDA = 0.01
DEF_MAXL = 0.05
DEF_OUT = "debug_dumps/Y_hat_bench.bin"


def run_once(bin_path, input_raw, lam, out_path, weights, max_lambda, use_opt, strict=False, dumps=False):
    env = os.environ.copy()
    env["USE_OPT_CONV"] = "1" if use_opt else "0"
    if strict:
        env["STRICT_PARITY"] = "1"
    else:
        env.pop("STRICT_PARITY", None)
    if dumps:
        env["DEBUG_DUMP"] = "1"
        env["DUMP_Y_PRE"] = "1"
        env["DUMP_Y_FLOAT"] = "1"
        env["DUMP_M"] = "1"
        env["DUMP_STAGES"] = "1"
    else:
        # Asegurar que no contamos I/O de dumps
        env.pop("DEBUG_DUMP", None)
        env.pop("DUMP_Y_PRE", None)
        env.pop("DUMP_Y_FLOAT", None)
        env.pop("DUMP_M", None)
        env.pop("DUMP_STAGES", None)
    cmd = [bin_path, input_raw, str(lam), out_path, weights, str(max_lambda)]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, env=env)
    t1 = time.perf_counter()
    return t1 - t0


def main():
    ap = argparse.ArgumentParser(description="Benchmark del encoder C con/sin optimización de conv 5x5.")
    ap.add_argument("--bin", default=DEF_BIN)
    ap.add_argument("--input", default=DEF_INPUT)
    ap.add_argument("--weights", default=DEF_WEIGHTS)
    ap.add_argument("--lambda", dest="lam", type=float, default=DEF_LAMBDA)
    ap.add_argument("--max-lambda", dest="maxl", type=float, default=DEF_MAXL)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--strict", action="store_true", help="Usar STRICT_PARITY=1 (1 hilo, redondeo tie-to-even)")
    ap.add_argument("--dumps", action="store_true", help="Mantener volcados (afecta tiempos)")
    args = ap.parse_args()

    # Warmup sin medir
    try:
        run_once(args.bin, args.input, args.lam, args.out, args.weights, args.maxl, use_opt=False, strict=args.strict, dumps=False)
    except Exception:
        pass

    def measure(use_opt: bool):
        times = []
        for _ in range(args.repeats):
            dt = run_once(args.bin, args.input, args.lam, args.out, args.weights, args.maxl, use_opt=use_opt, strict=args.strict, dumps=args.dumps)
            times.append(dt)
        m = mean(times)
        s = stdev(times) if len(times) > 1 else 0.0
        return times, m, s

    t0s, m0, s0 = measure(False)
    t1s, m1, s1 = measure(True)

    print("\n== Benchmark resultados ==")
    print(f"sin opt (USE_OPT_CONV=0): times={t0s}  mean={m0:.4f}s  std={s0:.4f}s")
    print(f"con opt (USE_OPT_CONV=1): times={t1s}  mean={m1:.4f}s  std={s1:.4f}s")
    if m1 > 0:
        print(f"speedup ≈ {m0/m1:.3f}x")
    else:
        print("speedup: n/a")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
