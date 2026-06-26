#!/usr/bin/env bash
set -euo pipefail

# Validacion corta para Raspberry Pi 3B+.
# Ejecuta solo casos representativos; no sustituye al run largo de 120 crops.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${OUT:-output/checkpoints/rpi_representative_validation_$(date +%Y%m%d_%H%M%S)}"
INPUT="${INPUT:-data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw}"
CALIBRATION="${CALIBRATION:-output/checkpoints/20260507_c_fixed_quality_qmap_wide/fq_calibration.tsv}"
if [[ ! -f "$CALIBRATION" ]]; then
  CALIBRATION="output/checkpoints/20260507_c_fixed_quality_qmap/fq_calibration.tsv"
fi
THREADS="${THREADS:-4}"
MODE="${MODE:-release}"
RPI_ARCH="${RPI_ARCH:-rpi3}"

mkdir -p "$OUT"/{qmaps,semantic_tsv,bitstreams,reconstructions,quality,logs,thermal}

log_thermal() {
  local tag="$1"
  {
    echo "tag=$tag"
    date -Is
    uptime || true
    vcgencmd measure_temp 2>/dev/null || true
    vcgencmd get_throttled 2>/dev/null || true
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || true
    free -h || true
  } > "$OUT/thermal/${tag}.txt"
}

run_timed() {
  local tag="$1"
  shift
  log_thermal "${tag}_before"
  if command -v /usr/bin/time >/dev/null 2>&1; then
    /usr/bin/time -v -o "$OUT/logs/${tag}.time.txt" "$@" > "$OUT/logs/${tag}.stdout.txt" 2> "$OUT/logs/${tag}.stderr.txt"
  else
    "$@" > "$OUT/logs/${tag}.stdout.txt" 2> "$OUT/logs/${tag}.stderr.txt"
  fi
  log_thermal "${tag}_after"
}

run_case() {
  local name="$1"
  local qmap="$2"
  local bitstream="$OUT/bitstreams/${name}.bin"
  local recon="$OUT/reconstructions/${name}.raw"

  echo "==> Case: $name"
  run_timed "${name}_compress" env OMP_NUM_THREADS="$THREADS" ./sorteny_compressor "$INPUT" 0.1 "$bitstream" weights/encoder 0.125 "$qmap"
  run_timed "${name}_decompress" env OMP_NUM_THREADS="$THREADS" ./sorteny_decompressor "$bitstream" "$recon" weights/decoder 0.125
  python3 src/python/analysis/analyze_block_quality.py \
    "$INPUT" "$recon" \
    --qmap "$qmap" \
    --output-json "$OUT/quality/${name}.json" \
    --output-csv "$OUT/quality/${name}.csv" \
    > "$OUT/logs/${name}_quality.stdout.txt" 2> "$OUT/logs/${name}_quality.stderr.txt"
  gzip -9 -c "$bitstream" > "$OUT/bitstreams/${name}.bin.gz"
}

echo "==> Build Raspberry representative validation"
make clean
make MODE="$MODE" OMP=1 RPI_ARCH="$RPI_ARCH"
make MODE="$MODE" OMP=1 RPI_ARCH="$RPI_ARCH" test_ops

log_thermal "run_start"

echo "==> Generate representative Q-maps"
./sorteny_fq_qmap \
  --calibration "$CALIBRATION" \
  --target-from-q 204 \
  --output-qmap "$OUT/qmaps/q204.raw" \
  --summary-tsv "$OUT/semantic_tsv/q204.tsv"

./sorteny_fq_qmap \
  --calibration "$CALIBRATION" \
  --adaptive-difficulty \
  --q-mean 204 \
  --adaptive-strength 8 \
  --output-qmap "$OUT/qmaps/adaptive_s8.raw" \
  --summary-tsv "$OUT/semantic_tsv/adaptive_s8.tsv"

python3 src/python/utils/generate_roi_map.py \
  "$OUT/qmaps/manual_center_roi.raw" \
  --pattern center \
  --output-tsv "$OUT/semantic_tsv/manual_center_roi.tsv" \
  --summary-json "$OUT/semantic_tsv/manual_center_roi_summary.json" \
  > "$OUT/logs/manual_center_roi.stdout.txt"

./sorteny_semantic_qmap \
  --calibration "$CALIBRATION" \
  --preset manual \
  --semantic-policy focus \
  --foreground-boost 16 \
  --background-q 128 \
  --roi-map "$OUT/qmaps/manual_center_roi.raw" \
  --output-qmap "$OUT/qmaps/manual_center_focus_bgq128.raw" \
  --summary-tsv "$OUT/semantic_tsv/manual_center_focus_bgq128.tsv"

./sorteny_semantic_qmap \
  --input "$INPUT" \
  --calibration "$CALIBRATION" \
  --preset vegetation \
  --semantic-policy focus \
  --foreground-boost 16 \
  --background-q 128 \
  --threshold 0.40 \
  --band-layout sentinel2-8 \
  --output-qmap "$OUT/qmaps/vegetation_focus_bgq128.raw" \
  --summary-tsv "$OUT/semantic_tsv/vegetation_focus_bgq128.tsv"

run_case "q204" "$OUT/qmaps/q204.raw"
run_case "adaptive_s8" "$OUT/qmaps/adaptive_s8.raw"
run_case "manual_center_focus_bgq128" "$OUT/qmaps/manual_center_focus_bgq128.raw"
run_case "vegetation_focus_bgq128" "$OUT/qmaps/vegetation_focus_bgq128.raw"

log_thermal "run_end"

{
  echo "# Raspberry Representative Validation"
  echo
  echo "- input: $INPUT"
  echo "- calibration: $CALIBRATION"
  echo "- threads: $THREADS"
  echo "- mode: $MODE"
  echo "- rpi_arch: $RPI_ARCH"
  echo
  echo "Cases: q204, adaptive_s8, manual_center_focus_bgq128, vegetation_focus_bgq128."
  echo "Quality JSON/CSV: $OUT/quality"
  echo "Thermal/throttling snapshots: $OUT/thermal"
  echo "Bitstream gzip proxy: $OUT/bitstreams/*.bin.gz"
} > "$OUT/README.md"

echo "[OK] Resultados en $OUT"
