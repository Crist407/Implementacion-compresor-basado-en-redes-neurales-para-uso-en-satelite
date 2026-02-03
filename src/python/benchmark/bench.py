#!/usr/bin/env python3
import argparse
import os
import shlex
import statistics
import subprocess
import sys
import time
from typing import List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY_EXE_DEFAULT = os.path.join(ROOT, ".venv", "bin", "python")
PY_VALIDATOR = os.path.join(ROOT, "src", "python", "validar_python.py")
C_BIN = os.path.join(ROOT, "sorteny_compressor")
DATA_RAW = os.path.join(ROOT, "data", "T31TCG_20230907T104629_5.8_512_512_2_1_0.raw")
WEIGHTS_DIR = os.path.join(ROOT, "weights", "pesos_bin")
OUT_DIR = os.path.join(ROOT, "debug_dumps")


def run_cmd(cmd: List[str], env=None, quiet=False) -> Tuple[int, float, str, str]:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    dt = time.perf_counter() - t0
    out = p.stdout.decode(errors="ignore")
    err = p.stderr.decode(errors="ignore")
    if not quiet:
        sys.stdout.write(out)
        sys.stderr.write(err)
    return p.returncode, dt, out, err


def bench_python(py: str, repeats: int, warmup: int, quiet: bool) -> List[float]:
    times: List[float] = []
    env = os.environ.copy()
    # Disable dumps and reduce TF logs
    for k in ["DUMP_SPECTRAL","DUMP_STAGES","DUMP_Y_PRE","DUMP_M","DUMP_Y_FLOAT"]:
        env[k] = "0"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # Optional: leave oneDNN on by default (fast CPU path)
    # Warmups
    for _ in range(warmup):
        rc, dt, _, _ = run_cmd([py, PY_VALIDATOR], env=env, quiet=True)
        if rc != 0:
            raise RuntimeError(f"Python validator failed during warmup (rc={rc}).")
    # Measured runs
    for _ in range(repeats):
        rc, dt, _, _ = run_cmd([py, PY_VALIDATOR], env=env, quiet=quiet)
        if rc != 0:
            raise RuntimeError(f"Python validator failed (rc={rc}).")
        times.append(dt)
    return times


def bench_c(cbin: str, repeats: int, warmup: int, lambda_val: float, omp_threads: int, use_half_even: bool, quiet: bool) -> List[float]:
    times: List[float] = []
    env = os.environ.copy()
    # Disable dumps
    for k in ["DUMP_SPECTRAL","DUMP_STAGES","DUMP_Y_PRE","DUMP_M","DUMP_Y_FLOAT"]:
        env[k] = "0"
    if use_half_even:
        env["USE_HALF_EVEN"] = "1"
    if omp_threads > 0:
        env["OMP_NUM_THREADS"] = str(omp_threads)
    # Warmups
    for _ in range(warmup):
        rc, dt, _, _ = run_cmd([cbin, DATA_RAW, f"{lambda_val}", os.path.join(OUT_DIR, "Y_hat_c_bench.bin"), WEIGHTS_DIR], env=env, quiet=True)
        if rc != 0:
            raise RuntimeError(f"C encoder failed during warmup (rc={rc}).")
    # Measured runs
    for _ in range(repeats):
        rc, dt, _, _ = run_cmd([cbin, DATA_RAW, f"{lambda_val}", os.path.join(OUT_DIR, "Y_hat_c_bench.bin"), WEIGHTS_DIR], env=env, quiet=quiet)
        if rc != 0:
            raise RuntimeError(f"C encoder failed (rc={rc}).")
        times.append(dt)
    return times


def summarize(name: str, times: List[float]) -> str:
    avg = statistics.mean(times)
    best = min(times)
    worst = max(times)
    sd = statistics.pstdev(times) if len(times) > 1 else 0.0
    return f"{name}: avg={avg:.3f}s best={best:.3f}s worst={worst:.3f}s sd={sd:.3f}s (n={len(times)})"


def main():
    ap = argparse.ArgumentParser(description="Benchmark Python vs C pipelines (wall-clock time)")
    ap.add_argument("--py", default=PY_EXE_DEFAULT, help="Python executable (default: .venv/bin/python)")
    ap.add_argument("--cbin", default=C_BIN, help="C binary path (default: ./sorteny_compressor)")
    ap.add_argument("--lambda", dest="lambda_val", type=float, default=0.01, help="Lambda value for C run")
    ap.add_argument("--repeats", type=int, default=3, help="Number of measured runs per pipeline")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup runs per pipeline (not measured)")
    ap.add_argument("--omp", dest="omp_threads", type=int, default=0, help="OMP_NUM_THREADS for C (0=leave default)")
    ap.add_argument("--half-even", action="store_true", help="Set USE_HALF_EVEN=1 for C to match Python rounding")
    ap.add_argument("--quiet", action="store_true", help="Suppress child process stdout/stderr during timing runs")
    args = ap.parse_args()

    # Basic checks
    if not os.path.isfile(args.cbin):
        print(f"ERROR: C binary not found: {args.cbin}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(PY_VALIDATOR):
        print(f"ERROR: Python validator not found: {PY_VALIDATOR}", file=sys.stderr)
        sys.exit(2)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Running benchmarks...\n")

    py_times = bench_python(args.py, args.repeats, args.warmup, args.quiet)
    c_times = bench_c(args.cbin, args.repeats, args.warmup, args.lambda_val, args.omp_threads, args.half_even, args.quiet)

    print("\nSummary:")
    print(summarize("Python", py_times))
    print(summarize("C", c_times))
    # Speedup C over Python (lower is faster → speedup = py_avg / c_avg)
    py_avg = statistics.mean(py_times)
    c_avg = statistics.mean(c_times)
    print(f"\nSpeedup (C vs Python): {py_avg / c_avg:.2f}x")


if __name__ == "__main__":
    main()
