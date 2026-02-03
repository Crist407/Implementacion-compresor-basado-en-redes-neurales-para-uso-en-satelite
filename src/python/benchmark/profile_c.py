#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEF_BIN = "./sorteny_compressor"
DEF_INPUT = "data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw"
DEF_WEIGHTS = "weights/pesos_bin_minimal"
DEF_LAMBDA = 0.01
DEF_MAXL = 0.05
DEF_OUT_TIMED = "debug_dumps/Y_hat_profile_tmp.bin"
DEF_OUT_DUMPS = "debug_dumps/Y_hat_c_even.bin"
DEF_DUMPS_DIR = "debug_dumps"

# Tensor dimensions for compare script
H4, W4, C3xBANDS = 32, 32, 3072

@dataclass
class Config:
    name: str
    use_opt_conv: bool
    strict_parity: bool = True

    def env(self) -> Dict[str, str]:
        e = {}
        e["USE_OPT_CONV"] = "1" if self.use_opt_conv else "0"
        if self.strict_parity:
            e["STRICT_PARITY"] = "1"
        return e


def which_time() -> Optional[str]:
    for path in ("/usr/bin/time", "/bin/time"):
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    return None


def parse_time_v(stderr: str) -> Tuple[Optional[int], Optional[float]]:
    rss_kb = None
    # Some time -v variants include an Elapsed time line; we also measure Python-side wall time
    for line in stderr.splitlines():
        if "Maximum resident set size (kbytes)" in line:
            try:
                rss_kb = int(re.search(r"(\d+)$", line).group(1))
            except Exception:
                pass
    # time -v may print Elapsed in various formats; we don't rely on it for now
    return rss_kb, None


def run_cmd(cmd: List[str], env: Optional[Dict[str, str]] = None, capture_output=False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, env=env, capture_output=capture_output, text=True)


def run_with_time(bin_path: str, args: List[str], env_extra: Dict[str, str], dry_run=False) -> Tuple[float, Optional[int]]:
    env = os.environ.copy()
    env.update(env_extra)
    # Ensure dumps are disabled for timing
    for k in ("DEBUG_DUMP","DUMP_Y_PRE","DUMP_Y_FLOAT","DUMP_M","DUMP_STAGES"):
        env.pop(k, None)
    time_path = which_time()
    cmd = [bin_path] + args
    if time_path is not None:
        cmd = [time_path, "-v"] + cmd
    if dry_run:
        print("DRY-RUN:", shlex.join([f"{k}={v}" for k,v in env_extra.items()]), shlex.join(cmd))
        return 0.0, None
    t0 = time.perf_counter()
    if time_path is not None:
        proc = subprocess.run(cmd, env=env, check=True, text=True, capture_output=True)
        t1 = time.perf_counter()
        rss_kb, _ = parse_time_v(proc.stderr)
        return t1 - t0, rss_kb
    else:
        # Fallback: no time -v available
        subprocess.run(cmd, env=env, check=True)
        t1 = time.perf_counter()
        return t1 - t0, None


def ensure_py_ground_truth(max_lambda: float, py_bin: str, dry_run=False) -> None:
    # validar_python.py uses env MAX_LAMBDA and hardcoded IMAGE_FILE/LAMBDA
    env = os.environ.copy()
    env["MAX_LAMBDA"] = str(max_lambda)
    # Ensure required dumps for comparator
    env["DUMP_STAGES"] = "1"
    env["DUMP_Y_FLOAT"] = "1"
    env["DUMP_M"] = "1"
    env["DUMP_Y_PRE"] = "1"
    cmd = [py_bin, "src/python/validar_python.py"]
    if dry_run:
        print("DRY-RUN:", "MAX_LAMBDA=", max_lambda, shlex.join(cmd))
        return
    print("[py] Generando dumps y verdad absoluta...")
    run_cmd(cmd, env)


def run_c_with_dumps(bin_path: str, args: List[str], env_extra: Dict[str, str], dry_run=False) -> None:
    env = os.environ.copy()
    env.update(env_extra)
    env["DEBUG_DUMP"] = "1"
    env["DUMP_Y_PRE"] = "1"
    env["DUMP_Y_FLOAT"] = "1"
    env["DUMP_M"] = "1"
    env["DUMP_STAGES"] = "1"
    cmd = [bin_path] + args
    if dry_run:
        print("DRY-RUN:", shlex.join([f"{k}={v}" for k,v in env_extra.items()]), shlex.join(cmd), "(with dumps)")
        return
    run_cmd(cmd, env)


def run_compare(dumps_dir: str, py_bin: str, dry_run=False) -> Dict[str, Tuple[float, float]]:
    cmd = [py_bin, "src/python/compare_products.py", "--C", dumps_dir, "--PY", dumps_dir,
           "--height", str(H4), "--width", str(W4), "--channels", str(C3xBANDS)]
    if dry_run:
        print("DRY-RUN:", shlex.join(cmd))
        return {}
    proc = run_cmd(cmd, capture_output=True)
    # Parse stats lines
    metrics: Dict[str, Tuple[float,float]] = {}
    lines = (proc.stdout or "").splitlines()
    pat = re.compile(r"^(Y_.*|conv3\*PY.*): max=([0-9.eE+-]+) mean=([0-9.eE+-]+)")
    for ln in lines:
        m = pat.match(ln.strip())
        if m:
            key = m.group(1)
            metrics[key] = (float(m.group(2)), float(m.group(3)))
    return metrics


def write_results_md(csv_path: Path, md_path: Path, rows: List[Dict[str, str]]) -> None:
    headers = ["config","use_opt","strict","time_s","peak_rss_kb","yf_max","yf_mean","yhat_max","yhat_mean"]
    csv_lines = [",".join(headers)]
    for r in rows:
        csv_lines.append(",".join(r.get(h, "") for h in headers))
    csv_path.write_text("\n".join(csv_lines))
    # Simple Markdown table
    md_lines = [
        "| Config | USE_OPT_CONV | STRICT_PARITY | Time (s) | Peak RSS (KB) | Y_float max | Y_float mean | Y_hat max | Y_hat mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['config']} | {r['use_opt']} | {r['strict']} | {r['time_s']} | {r['peak_rss_kb']} | {r['yf_max']} | {r['yf_mean']} | {r['yhat_max']} | {r['yhat_mean']} |"
        )
    md_path.write_text("\n".join(md_lines))


def main():
    ap = argparse.ArgumentParser(description="Perfilado de memoria/tiempo y diffs C vs Python")
    ap.add_argument("--bin", default=DEF_BIN)
    ap.add_argument("--input", default=DEF_INPUT)
    ap.add_argument("--weights", default=DEF_WEIGHTS)
    ap.add_argument("--lambda", dest="lam", type=float, default=DEF_LAMBDA)
    ap.add_argument("--max-lambda", dest="maxl", type=float, default=DEF_MAXL)
    ap.add_argument("--out", default=DEF_OUT_TIMED)
    ap.add_argument("--dry-run", action="store_true", help="Solo mostrar comandos, no ejecutar")
    ap.add_argument("--configs", nargs="*", default=["baseline","opt_conv"],
                    help="Conjuntos: baseline, opt_conv")
    ap.add_argument("--python", dest="py_bin", default=None,
                    help="Intérprete de Python para los scripts de referencia (por defecto: .venv/bin/python si existe, si no sys.executable)")
    args = ap.parse_args()

    Path(DEF_DUMPS_DIR).mkdir(exist_ok=True)

    # Resolver intérprete de Python preferido
    py_bin = args.py_bin
    if not py_bin:
        venv_python = Path('.venv/bin/python')
        if venv_python.exists() and os.access(venv_python, os.X_OK):
            py_bin = str(venv_python)
        else:
            py_bin = sys.executable

    # Ensure Python dumps/ground truth first (once)
    ensure_py_ground_truth(args.maxl, py_bin=py_bin, dry_run=args.dry_run)

    # Build commands args for the C encoder
    # Build command args for timed and dumps runs (use different outputs to avoid confusion)
    out_timed = args.out if args.out else DEF_OUT_TIMED
    out_dumps = DEF_OUT_DUMPS
    c_args_timed = [args.input, str(args.lam), out_timed, args.weights, str(args.maxl)]
    c_args_dumps = [args.input, str(args.lam), out_dumps, args.weights, str(args.maxl)]

    name_to_cfg = {
        "baseline": Config(name="baseline", use_opt_conv=False, strict_parity=True),
        "opt_conv": Config(name="opt_conv", use_opt_conv=True, strict_parity=True),
    }
    selected: List[Config] = []
    for n in args.configs:
        if n not in name_to_cfg:
            print(f"[WARN] Config desconocida: {n}")
            continue
        selected.append(name_to_cfg[n])

    results: List[Dict[str, str]] = []
    for cfg in selected:
        print(f"\n== Running config: {cfg.name} ==")
        # Timed run without dumps
        dt, rss_kb = run_with_time(args.bin, c_args_timed, cfg.env(), dry_run=args.dry_run)
        # Dump run for diffs
        run_c_with_dumps(args.bin, c_args_dumps, cfg.env(), dry_run=args.dry_run)
        # Compare
        metrics = run_compare(DEF_DUMPS_DIR, py_bin=py_bin, dry_run=args.dry_run)
        yf = metrics.get("Y_float_c vs Y_float_py", (float('nan'), float('nan')))
        yh = metrics.get("Y_hat_c_even vs py_hat", (float('nan'), float('nan')))
        row = {
            "config": cfg.name,
            "use_opt": "1" if cfg.use_opt_conv else "0",
            "strict": "1" if cfg.strict_parity else "0",
            "time_s": f"{dt:.4f}" if not args.dry_run else "-",
            "peak_rss_kb": str(rss_kb) if (rss_kb is not None and not args.dry_run) else "-",
            "yf_max": f"{yf[0]:.6g}" if not args.dry_run else "-",
            "yf_mean": f"{yf[1]:.6g}" if not args.dry_run else "-",
            "yhat_max": f"{yh[0]:.6g}" if not args.dry_run else "-",
            "yhat_mean": f"{yh[1]:.6g}" if not args.dry_run else "-",
        }
        results.append(row)

    # Write outputs
    csv_path = Path(DEF_DUMPS_DIR) / "profile_results.csv"
    md_path = Path(DEF_DUMPS_DIR) / "profile_results.md"
    if args.dry_run:
        print("(dry-run) Results would be written to:", csv_path, md_path)
    else:
        write_results_md(csv_path, md_path, results)
        print("\nResumen guardado en:")
        print(" -", csv_path)
        print(" -", md_path)
        print("\nSugerencia: abre profile_results.md para una vista rápida.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
