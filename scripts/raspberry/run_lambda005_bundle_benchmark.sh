#!/usr/bin/env bash
set -euo pipefail

# Ejecuta el benchmark lambda005 sobre todas las imagenes representativas del bundle.
# No instala dependencias y no usa Python/NumPy/TensorFlow en Raspberry.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-output/checkpoints/rpi_lambda005_bundle_benchmark_$(date +%Y%m%d_%H%M%S)}"
THREADS_LIST="${THREADS_LIST:-1 2 4}"
MAX_INPUTS="${MAX_INPUTS:-0}"
MAX_CASES="${MAX_CASES:-0}"
OMP_DYNAMIC_SETTING="${OMP_DYNAMIC_SETTING:-FALSE}"
OMP_PROC_BIND_SETTING="${OMP_PROC_BIND_SETTING:-close}"
OMP_PLACES_SETTING="${OMP_PLACES_SETTING:-cores}"

mkdir -p "$OUT_ROOT"

mapfile -t INPUTS < <(find data/representative -maxdepth 1 -name '*.raw' | sort)
if [[ "${#INPUTS[@]}" -eq 0 ]]; then
  echo "[ERROR] No RAW files found under data/representative" >&2
  exit 1
fi

{
  echo "# SORTENY Raspberry lambda005 bundle benchmark"
  echo
  echo "- started_at: $(date -Is)"
  echo "- root: $ROOT"
  echo "- out_root: $OUT_ROOT"
  echo "- threads_list: $THREADS_LIST"
  echo "- omp_dynamic: $OMP_DYNAMIC_SETTING"
  echo "- omp_proc_bind: $OMP_PROC_BIND_SETTING"
  echo "- omp_places: $OMP_PLACES_SETTING"
  echo "- max_inputs: $MAX_INPUTS"
  echo "- max_cases: $MAX_CASES"
  echo
  echo "## Inputs"
  printf -- "- %s\n" "${INPUTS[@]}"
} > "$OUT_ROOT/README.md"

idx=0
for input in "${INPUTS[@]}"; do
  idx=$((idx + 1))
  if [[ "$MAX_INPUTS" != "0" && "$idx" -gt "$MAX_INPUTS" ]]; then
    break
  fi
  stem="$(basename "$input" .raw)"
  echo "==> Bundle input [$idx/${#INPUTS[@]}]: $stem"
  OUT="$OUT_ROOT/$stem" \
  INPUT="$input" \
  THREADS_LIST="$THREADS_LIST" \
  MAX_CASES="$MAX_CASES" \
  OMP_DYNAMIC_SETTING="$OMP_DYNAMIC_SETTING" \
  OMP_PROC_BIND_SETTING="$OMP_PROC_BIND_SETTING" \
  OMP_PLACES_SETTING="$OMP_PLACES_SETTING" \
  RESUME=1 \
    scripts/raspberry/run_lambda005_auto_benchmark.sh
done

{
  echo
  echo "- finished_at: $(date -Is)"
  echo "- result_archive: ${OUT_ROOT}.tar.gz"
} >> "$OUT_ROOT/README.md"

tar -czf "${OUT_ROOT}.tar.gz" -C "$(dirname "$OUT_ROOT")" "$(basename "$OUT_ROOT")"

echo "[OK] Bundle benchmark complete: $OUT_ROOT"
echo "[OK] Archive ready: ${OUT_ROOT}.tar.gz"
