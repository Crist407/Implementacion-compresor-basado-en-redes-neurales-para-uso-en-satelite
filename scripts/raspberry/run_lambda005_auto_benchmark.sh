#!/usr/bin/env bash
set -euo pipefail

# Benchmark C-only para Raspberry, sin instalar dependencias y sin usar Python.
# El analisis numerico detallado se realiza despues en PC/WSL con
# src/python/analysis/build_raspberry_benchmark_report.py.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${OUT:-output/checkpoints/rpi_lambda005_auto_benchmark_$(date +%Y%m%d_%H%M%S)}"
if [[ -z "${INPUT:-}" ]]; then
  if compgen -G "data/representative/*.raw" >/dev/null; then
    INPUT="$(find data/representative -maxdepth 1 -name '*.raw' | sort | head -n 1)"
  else
    INPUT="data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw"
  fi
fi
CALIBRATION="${CALIBRATION:-output/checkpoints/20260613_lambda005_recalibration_audit/calibrations/canonical/fq_calibration_lambda005.tsv}"
THRESHOLD_CONFIG="${THRESHOLD_CONFIG:-config/auto_thresholds_lambda005.tsv}"
INCLUDE_EXPERIMENTAL="${INCLUDE_EXPERIMENTAL:-0}"
LAMBDA_VALUE="${LAMBDA_VALUE:-0.05}"
MAX_LAMBDA="${MAX_LAMBDA:-0.05}"
THREADS_LIST="${THREADS_LIST:-1 2 4}"
MAX_CASES="${MAX_CASES:-0}"
MODE="${MODE:-release}"
RPI_ARCH="${RPI_ARCH:-rpi3}"
RESUME="${RESUME:-1}"
OMP_DYNAMIC_SETTING="${OMP_DYNAMIC_SETTING:-FALSE}"
OMP_PROC_BIND_SETTING="${OMP_PROC_BIND_SETTING:-close}"
OMP_PLACES_SETTING="${OMP_PLACES_SETTING:-cores}"

EXPECTED_RAW_BYTES=$((8 * 512 * 512 * 2))
EXPECTED_QMAP_BYTES=$((32 * 32))

mkdir -p "$OUT"/{input,qmaps,semantic_tsv,logs,thermal,runs}

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
  local tag="$1"
  local path="$OUT/thermal/${tag}.txt"
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
  local tag="$1"
  shift
  log_snapshot "${tag}_before"
  if command -v /usr/bin/time >/dev/null 2>&1; then
    /usr/bin/time -v -o "$OUT/logs/${tag}.time.txt" "$@" > "$OUT/logs/${tag}.stdout.txt" 2> "$OUT/logs/${tag}.stderr.txt"
  else
    "$@" > "$OUT/logs/${tag}.stdout.txt" 2> "$OUT/logs/${tag}.stderr.txt"
  fi
  log_snapshot "${tag}_after"
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

preflight() {
  require_file Makefile
  require_file "$INPUT"
  require_file "$CALIBRATION"
  require_file "$THRESHOLD_CONFIG"
  require_dir weights/encoder
  require_dir weights/decoder
  [[ "$(file_size "$INPUT")" == "$EXPECTED_RAW_BYTES" ]] || die "$INPUT is not an 8x512x512 uint16 RAW"

  {
    echo "created_at=$(date -Is)"
    echo "root=$ROOT"
    echo "out=$OUT"
    echo "input=$INPUT"
    echo "calibration=$CALIBRATION"
    echo "threshold_config=$THRESHOLD_CONFIG"
    echo "include_experimental=$INCLUDE_EXPERIMENTAL"
    echo "lambda_value=$LAMBDA_VALUE"
    echo "max_lambda=$MAX_LAMBDA"
    echo "threads_list=$THREADS_LIST"
    echo "max_cases=$MAX_CASES"
    echo "mode=$MODE"
    echo "rpi_arch=$RPI_ARCH"
    echo "resume=$RESUME"
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
  } > "$OUT/run_manifest.txt"
}

semantic_case_name() {
  printf '%s_focus_bgq128' "$1"
}

should_include_tier() {
  local tier="$1"
  if [[ "$tier" == "experimental" && "$INCLUDE_EXPERIMENTAL" != "1" ]]; then
    return 1
  fi
  return 0
}

generate_qmaps() {
  echo "==> Generate Q-maps"
  ./sorteny_fq_qmap \
    --calibration "$CALIBRATION" \
    --target-from-q 204 \
    --output-qmap "$OUT/qmaps/q204.raw" \
    --summary-tsv "$OUT/semantic_tsv/q204.tsv"
  check_size "$OUT/qmaps/q204.raw" "$EXPECTED_QMAP_BYTES"

  ./sorteny_fq_qmap \
    --calibration "$CALIBRATION" \
    --adaptive-difficulty \
    --q-mean 204 \
    --adaptive-strength 8 \
    --output-qmap "$OUT/qmaps/adaptive_s8.raw" \
    --summary-tsv "$OUT/semantic_tsv/adaptive_s8.tsv"
  check_size "$OUT/qmaps/adaptive_s8.raw" "$EXPECTED_QMAP_BYTES"

  CASES=(
    "q204|$OUT/qmaps/q204.raw|$OUT/semantic_tsv/q204.tsv"
    "adaptive_s8|$OUT/qmaps/adaptive_s8.raw|$OUT/semantic_tsv/adaptive_s8.tsv"
  )

  while IFS=$'\t' read -r preset threshold tier description; do
    [[ -z "${preset:-}" || "$preset" == "preset" || "$preset" == \#* ]] && continue
    [[ -n "${threshold:-}" ]] || die "Missing threshold for preset $preset in $THRESHOLD_CONFIG"
    tier="${tier:-operational}"
    if ! should_include_tier "$tier"; then
      continue
    fi

    local case_name qmap summary
    case_name="$(semantic_case_name "$preset")"
    qmap="$OUT/qmaps/${case_name}.raw"
    summary="$OUT/semantic_tsv/${case_name}.tsv"

    ./sorteny_semantic_qmap \
      --input "$INPUT" \
      --calibration "$CALIBRATION" \
      --preset "$preset" \
      --semantic-policy focus \
      --foreground-boost 16 \
      --background-q 128 \
      --threshold "$threshold" \
      --band-layout sentinel2-8 \
      --output-qmap "$qmap" \
      --summary-tsv "$summary"
    check_size "$qmap" "$EXPECTED_QMAP_BYTES"
    CASES+=("${case_name}|${qmap}|${summary}")
  done < "$THRESHOLD_CONFIG"
}

run_case() {
  local case_name="$1"
  local qmap="$2"
  local summary_tsv="$3"
  local threads="$4"
  local run_dir="$OUT/runs/${case_name}/threads_${threads}"
  local done_marker="$run_dir/DONE"
  local bitstream="$run_dir/bitstream.bin"
  local bitstream_gz="$run_dir/bitstream.bin.gz"
  local recon="$run_dir/reconstruction.raw"
  local tag="${case_name}_t${threads}"
  local -a omp_env=(
    env
    "OMP_NUM_THREADS=$threads"
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
    echo "  RESUME $case_name threads=$threads"
    return
  fi

  rm -f "$bitstream" "$bitstream_gz" "$recon" "$done_marker"
  echo "==> Run $case_name threads=$threads"
  run_timed "${tag}_compress" "${omp_env[@]}" ./sorteny_compressor \
    "$INPUT" "$LAMBDA_VALUE" "$bitstream" weights/encoder "$MAX_LAMBDA" "$qmap"
  run_timed "${tag}_decompress" "${omp_env[@]}" ./sorteny_decompressor \
    "$bitstream" "$recon" weights/decoder "$MAX_LAMBDA"

  check_size "$recon" "$EXPECTED_RAW_BYTES"
  gzip -9 -c "$bitstream" > "$bitstream_gz"

  {
    sha256sum "$INPUT" "$qmap" "$bitstream" "$recon" 2>/dev/null || true
  } > "$run_dir/sha256.txt"

  write_kv "$run_dir/run_meta.tsv" \
    case "$case_name" \
    threads "$threads" \
    input "$OUT/input/$(basename "$INPUT")" \
    qmap "$qmap" \
    summary_tsv "$summary_tsv" \
    bitstream "$bitstream" \
    bitstream_gzip "$bitstream_gz" \
    reconstruction "$recon" \
    compress_time_log "$OUT/logs/${tag}_compress.time.txt" \
    decompress_time_log "$OUT/logs/${tag}_decompress.time.txt" \
    compress_stdout "$OUT/logs/${tag}_compress.stdout.txt" \
    compress_stderr "$OUT/logs/${tag}_compress.stderr.txt" \
    decompress_stdout "$OUT/logs/${tag}_decompress.stdout.txt" \
    decompress_stderr "$OUT/logs/${tag}_decompress.stderr.txt" \
    thermal_compress_before "$OUT/thermal/${tag}_compress_before.txt" \
    thermal_compress_after "$OUT/thermal/${tag}_compress_after.txt" \
    thermal_decompress_before "$OUT/thermal/${tag}_decompress_before.txt" \
    thermal_decompress_after "$OUT/thermal/${tag}_decompress_after.txt" \
    input_bytes "$(file_size "$INPUT")" \
    qmap_bytes "$(file_size "$qmap")" \
    bitstream_bytes "$(file_size "$bitstream")" \
    bitstream_gzip_bytes "$(file_size "$bitstream_gz")" \
    reconstruction_bytes "$(file_size "$recon")" \
    input_sha256 "$(sha_file "$INPUT")" \
    qmap_sha256 "$(sha_file "$qmap")" \
    bitstream_sha256 "$(sha_file "$bitstream")" \
    reconstruction_sha256 "$(sha_file "$recon")" \
    lambda_value "$LAMBDA_VALUE" \
    max_lambda "$MAX_LAMBDA" \
    omp_dynamic "$OMP_DYNAMIC_SETTING" \
    omp_proc_bind "$OMP_PROC_BIND_SETTING" \
    omp_places "$OMP_PLACES_SETTING"

  date -Is > "$done_marker"
}

preflight

INPUT_COPY="$OUT/input/$(basename "$INPUT")"
if [[ ! -f "$INPUT_COPY" ]]; then
  cp -p "$INPUT" "$INPUT_COPY"
fi

echo "==> Build C binaries"
make MODE="$MODE" OMP=1 RPI_ARCH="$RPI_ARCH" print_config > "$OUT/logs/make_print_config.txt"
make MODE="$MODE" OMP=1 RPI_ARCH="$RPI_ARCH"
make MODE="$MODE" OMP=1 RPI_ARCH="$RPI_ARCH" test_ops

log_snapshot "run_start"
generate_qmaps

case_index=0
for case_spec in "${CASES[@]}"; do
  case_index=$((case_index + 1))
  if [[ "$MAX_CASES" != "0" && "$case_index" -gt "$MAX_CASES" ]]; then
    break
  fi
  IFS='|' read -r case_name qmap summary_tsv <<< "$case_spec"
  for threads in $THREADS_LIST; do
    run_case "$case_name" "$qmap" "$summary_tsv" "$threads"
  done
done

log_snapshot "run_end"

{
  echo "# Raspberry lambda005 auto benchmark"
  echo
  echo "C-only benchmark. No pip/apt/numpy/TensorFlow/SORTENY.py was used."
  echo
  echo "- input: $INPUT"
  echo "- input copy: $INPUT_COPY"
  echo "- calibration: $CALIBRATION"
  echo "- threshold_config: $THRESHOLD_CONFIG"
  echo "- include_experimental: $INCLUDE_EXPERIMENTAL"
  echo "- lambda_value: $LAMBDA_VALUE"
  echo "- max_lambda: $MAX_LAMBDA"
  echo "- threads_list: $THREADS_LIST"
  echo "- mode: $MODE"
  echo "- rpi_arch: $RPI_ARCH"
  echo "- omp_dynamic: $OMP_DYNAMIC_SETTING"
  echo "- omp_proc_bind: $OMP_PROC_BIND_SETTING"
  echo "- omp_places: $OMP_PLACES_SETTING"
  echo
  echo "Copy this checkpoint back to PC/WSL and run:"
  echo
  echo '```bash'
  echo "python3 src/python/analysis/build_raspberry_benchmark_report.py $OUT"
  echo '```'
} > "$OUT/README.md"

echo "[OK] Raspberry benchmark written to $OUT"
