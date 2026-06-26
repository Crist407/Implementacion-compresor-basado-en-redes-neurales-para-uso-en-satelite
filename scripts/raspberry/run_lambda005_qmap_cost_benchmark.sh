#!/usr/bin/env bash
set -euo pipefail

# Benchmark C-only del coste de generar Q-maps en Raspberry.
# No ejecuta sorteny_compressor ni sorteny_decompressor.
# El analisis se hace despues en PC/WSL con
# src/python/analysis/build_raspberry_qmap_cost_report.py.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-output/checkpoints/rpi_lambda005_qmap_cost_benchmark_$(date +%Y%m%d_%H%M%S)}"
INPUT_DIR="${INPUT_DIR:-data/representative}"
CALIBRATION="${CALIBRATION:-output/checkpoints/20260613_lambda005_recalibration_audit/calibrations/canonical/fq_calibration_lambda005.tsv}"
THRESHOLD_CONFIG="${THRESHOLD_CONFIG:-config/auto_thresholds_lambda005.tsv}"
INCLUDE_EXPERIMENTAL="${INCLUDE_EXPERIMENTAL:-1}"
REPEATS="${REPEATS:-3}"
MAX_INPUTS="${MAX_INPUTS:-0}"
MAX_CASES="${MAX_CASES:-0}"
MODE="${MODE:-release}"
RPI_ARCH="${RPI_ARCH:-rpi3}"
RESUME="${RESUME:-1}"
OMP_NUM_THREADS_VALUE="${OMP_NUM_THREADS_VALUE:-4}"
OMP_DYNAMIC_SETTING="${OMP_DYNAMIC_SETTING:-FALSE}"
OMP_PROC_BIND_SETTING="${OMP_PROC_BIND_SETTING:-close}"
OMP_PLACES_SETTING="${OMP_PLACES_SETTING:-cores}"

EXPECTED_RAW_BYTES=$((8 * 512 * 512 * 2))
EXPECTED_QMAP_BYTES=$((32 * 32))

mkdir -p "$OUT_ROOT"/{logs,thermal}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

file_size() {
  stat -c '%s' "$1" 2>/dev/null || echo 0
}

sha_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    echo ""
  fi
}

require_file() {
  [[ -f "$1" ]] || die "Missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "Missing directory: $1"
}

check_size() {
  local path="$1"
  local expected="$2"
  local got
  got="$(file_size "$path")"
  [[ "$got" == "$expected" ]] || die "$path has $got bytes, expected $expected"
}

log_snapshot() {
  local base="$1"
  local tag="$2"
  local path="$base/thermal/${tag}.txt"
  mkdir -p "$base/thermal"
  {
    echo "tag=$tag"
    date -Is
    hostname || true
    uname -a || true
    uptime || true
    vcgencmd measure_temp 2>/dev/null || true
    vcgencmd get_throttled 2>/dev/null || true
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || true
    free -h || true
    df -h . || true
  } > "$path"
}

run_timed() {
  local base="$1"
  local tag="$2"
  shift 2
  mkdir -p "$base/logs"
  log_snapshot "$base" "${tag}_before"
  if command -v /usr/bin/time >/dev/null 2>&1; then
    /usr/bin/time -v -o "$base/logs/${tag}.time.txt" "$@" > "$base/logs/${tag}.stdout.txt" 2> "$base/logs/${tag}.stderr.txt"
  else
    "$@" > "$base/logs/${tag}.stdout.txt" 2> "$base/logs/${tag}.stderr.txt"
  fi
  log_snapshot "$base" "${tag}_after"
}

write_kv() {
  local path="$1"
  shift
  : > "$path"
  while [[ "$#" -gt 0 ]]; do
    printf '%s=%s\n' "$1" "$2" >> "$path"
    shift 2
  done
}

should_include_tier() {
  local tier="$1"
  if [[ "$tier" == "experimental" && "$INCLUDE_EXPERIMENTAL" != "1" ]]; then
    return 1
  fi
  return 0
}

semantic_case_name() {
  printf '%s_focus_bgq128' "$1"
}

preflight() {
  require_file Makefile
  require_file "$CALIBRATION"
  require_file "$THRESHOLD_CONFIG"
  require_dir "$INPUT_DIR"
  mapfile -t INPUTS < <(find "$INPUT_DIR" -maxdepth 1 -name '*.raw' | sort)
  [[ "${#INPUTS[@]}" -gt 0 ]] || die "No RAW files found under $INPUT_DIR"

  {
    echo "created_at=$(date -Is)"
    echo "root=$ROOT"
    echo "out_root=$OUT_ROOT"
    echo "input_dir=$INPUT_DIR"
    echo "calibration=$CALIBRATION"
    echo "threshold_config=$THRESHOLD_CONFIG"
    echo "include_experimental=$INCLUDE_EXPERIMENTAL"
    echo "repeats=$REPEATS"
    echo "max_inputs=$MAX_INPUTS"
    echo "max_cases=$MAX_CASES"
    echo "mode=$MODE"
    echo "rpi_arch=$RPI_ARCH"
    echo "resume=$RESUME"
    echo "omp_num_threads=$OMP_NUM_THREADS_VALUE"
    echo "omp_dynamic=$OMP_DYNAMIC_SETTING"
    echo "omp_proc_bind=$OMP_PROC_BIND_SETTING"
    echo "omp_places=$OMP_PLACES_SETTING"
    echo "uname=$(uname -a)"
    echo "arch=$(uname -m)"
    echo "nproc=$(nproc)"
    echo "gcc=$(gcc --version 2>/dev/null | head -n 1 || true)"
    echo "make=$(make --version 2>/dev/null | head -n 1 || true)"
    echo "vcgencmd=$(command -v vcgencmd || true)"
    vcgencmd measure_temp 2>/dev/null || true
    vcgencmd get_throttled 2>/dev/null || true
    free -h || true
    df -h . || true
  } > "$OUT_ROOT/run_manifest.txt"
}

build_cases() {
  CASES=(
    "q204|fq_q204||||Q fixed at 204"
    "adaptive_s8|fq_adaptive||||Adaptive difficulty with q_mean=204 and strength=8"
  )

  while IFS=$'\t' read -r preset threshold tier description; do
    [[ -z "${preset:-}" || "$preset" == "preset" || "$preset" == \#* ]] && continue
    [[ -n "${threshold:-}" ]] || die "Missing threshold for preset $preset in $THRESHOLD_CONFIG"
    tier="${tier:-operational}"
    if ! should_include_tier "$tier"; then
      continue
    fi
    CASES+=("$(semantic_case_name "$preset")|semantic|$preset|$threshold|$tier|$description")
  done < "$THRESHOLD_CONFIG"
}

run_qmap_case() {
  local input="$1"
  local input_out="$2"
  local case_name="$3"
  local command_type="$4"
  local preset="$5"
  local threshold="$6"
  local tier="$7"
  local description="$8"
  local repeat="$9"
  local run_dir="$input_out/runs/${case_name}/repeat_${repeat}"
  local done_marker="$run_dir/DONE"
  local qmap="$run_dir/qmap.raw"
  local summary="$run_dir/summary.tsv"
  local tag="${case_name}_r${repeat}"
  local -a omp_env=(
    env
    "OMP_NUM_THREADS=$OMP_NUM_THREADS_VALUE"
    "OMP_DYNAMIC=$OMP_DYNAMIC_SETTING"
  )
  if [[ -n "$OMP_PROC_BIND_SETTING" ]]; then
    omp_env+=("OMP_PROC_BIND=$OMP_PROC_BIND_SETTING")
  fi
  if [[ -n "$OMP_PLACES_SETTING" ]]; then
    omp_env+=("OMP_PLACES=$OMP_PLACES_SETTING")
  fi

  mkdir -p "$run_dir"
  if [[ "$RESUME" == "1" && -f "$done_marker" ]]; then
    echo "  RESUME $case_name repeat=$repeat"
    return
  fi

  rm -f "$qmap" "$summary" "$done_marker"
  echo "==> Q-map $case_name repeat=$repeat input=$(basename "$input")"

  if [[ "$command_type" == "fq_q204" ]]; then
    run_timed "$input_out" "$tag" "${omp_env[@]}" ./sorteny_fq_qmap \
      --calibration "$CALIBRATION" \
      --target-from-q 204 \
      --output-qmap "$qmap" \
      --summary-tsv "$summary"
  elif [[ "$command_type" == "fq_adaptive" ]]; then
    run_timed "$input_out" "$tag" "${omp_env[@]}" ./sorteny_fq_qmap \
      --calibration "$CALIBRATION" \
      --adaptive-difficulty \
      --q-mean 204 \
      --adaptive-strength 8 \
      --output-qmap "$qmap" \
      --summary-tsv "$summary"
  elif [[ "$command_type" == "semantic" ]]; then
    run_timed "$input_out" "$tag" "${omp_env[@]}" ./sorteny_semantic_qmap \
      --input "$input" \
      --calibration "$CALIBRATION" \
      --preset "$preset" \
      --semantic-policy focus \
      --foreground-boost 16 \
      --background-q 128 \
      --threshold "$threshold" \
      --band-layout sentinel2-8 \
      --output-qmap "$qmap" \
      --summary-tsv "$summary"
  else
    die "Unknown command_type: $command_type"
  fi

  check_size "$qmap" "$EXPECTED_QMAP_BYTES"
  [[ -s "$summary" ]] || die "Missing or empty summary TSV: $summary"

  {
    sha256sum "$input" "$qmap" "$summary" 2>/dev/null || true
  } > "$run_dir/sha256.txt"

  write_kv "$run_dir/run_meta.tsv" \
    input_id "$(basename "$input" .raw)" \
    case "$case_name" \
    repeat "$repeat" \
    threads "$OMP_NUM_THREADS_VALUE" \
    command_type "$command_type" \
    preset "$preset" \
    threshold "$threshold" \
    tier "$tier" \
    description "$description" \
    input "$input_out/input/$(basename "$input")" \
    calibration "$CALIBRATION" \
    threshold_config "$THRESHOLD_CONFIG" \
    qmap "$qmap" \
    summary_tsv "$summary" \
    qmap_time_log "$input_out/logs/${tag}.time.txt" \
    qmap_stdout "$input_out/logs/${tag}.stdout.txt" \
    qmap_stderr "$input_out/logs/${tag}.stderr.txt" \
    thermal_qmap_before "$input_out/thermal/${tag}_before.txt" \
    thermal_qmap_after "$input_out/thermal/${tag}_after.txt" \
    input_bytes "$(file_size "$input")" \
    qmap_bytes "$(file_size "$qmap")" \
    summary_tsv_bytes "$(file_size "$summary")" \
    input_sha256 "$(sha_file "$input")" \
    qmap_sha256 "$(sha_file "$qmap")" \
    summary_tsv_sha256 "$(sha_file "$summary")" \
    omp_num_threads "$OMP_NUM_THREADS_VALUE" \
    omp_dynamic "$OMP_DYNAMIC_SETTING" \
    omp_proc_bind "$OMP_PROC_BIND_SETTING" \
    omp_places "$OMP_PLACES_SETTING"

  date -Is > "$done_marker"
}

preflight
build_cases

echo "==> Build C binaries"
make MODE="$MODE" OMP=1 RPI_ARCH="$RPI_ARCH" print_config > "$OUT_ROOT/logs/make_print_config.txt"
make MODE="$MODE" OMP=1 RPI_ARCH="$RPI_ARCH"
make MODE="$MODE" OMP=1 RPI_ARCH="$RPI_ARCH" test_ops

log_snapshot "$OUT_ROOT" "run_start"

idx=0
for input in "${INPUTS[@]}"; do
  idx=$((idx + 1))
  if [[ "$MAX_INPUTS" != "0" && "$idx" -gt "$MAX_INPUTS" ]]; then
    break
  fi
  [[ "$(file_size "$input")" == "$EXPECTED_RAW_BYTES" ]] || die "$input is not an 8x512x512 uint16 RAW"

  stem="$(basename "$input" .raw)"
  input_out="$OUT_ROOT/$stem"
  mkdir -p "$input_out"/{input,logs,thermal,runs}
  input_copy="$input_out/input/$(basename "$input")"
  if [[ ! -f "$input_copy" ]]; then
    cp -p "$input" "$input_copy"
  fi

  echo "==> Input [$idx/${#INPUTS[@]}]: $stem"
  case_index=0
  for case_spec in "${CASES[@]}"; do
    case_index=$((case_index + 1))
    if [[ "$MAX_CASES" != "0" && "$case_index" -gt "$MAX_CASES" ]]; then
      break
    fi
    IFS='|' read -r case_name command_type preset threshold tier description <<< "$case_spec"
    for repeat in $(seq 1 "$REPEATS"); do
      run_qmap_case "$input" "$input_out" "$case_name" "$command_type" "$preset" "$threshold" "$tier" "$description" "$repeat"
    done
  done
done

log_snapshot "$OUT_ROOT" "run_end"

{
  echo "# Raspberry lambda005 Q-map cost benchmark"
  echo
  echo "C-only benchmark. No sorteny_compressor, sorteny_decompressor, pip, apt, NumPy, TensorFlow or SORTENY.py was used."
  echo
  echo "- input_dir: $INPUT_DIR"
  echo "- calibration: $CALIBRATION"
  echo "- threshold_config: $THRESHOLD_CONFIG"
  echo "- include_experimental: $INCLUDE_EXPERIMENTAL"
  echo "- repeats: $REPEATS"
  echo "- omp_num_threads: $OMP_NUM_THREADS_VALUE"
  echo "- mode: $MODE"
  echo "- rpi_arch: $RPI_ARCH"
  echo
  echo "Copy this checkpoint back to PC/WSL and run:"
  echo
  echo '```bash'
  echo "python3 src/python/analysis/build_raspberry_qmap_cost_report.py --qmap-checkpoint $OUT_ROOT --raspberry-report output/checkpoints/20260614_raspberry_lambda005_optimized_benchmark_report --output-dir output/checkpoints/20260701_raspberry_lambda005_qmap_cost_report"
  echo '```'
} > "$OUT_ROOT/README.md"

tar -czf "${OUT_ROOT}.tar.gz" -C "$(dirname "$OUT_ROOT")" "$(basename "$OUT_ROOT")"

echo "[OK] Raspberry Q-map cost benchmark complete: $OUT_ROOT"
echo "[OK] Archive ready: ${OUT_ROOT}.tar.gz"
