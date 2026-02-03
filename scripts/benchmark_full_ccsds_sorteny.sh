#!/bin/bash
#
# Benchmark Completo: CCSDS 122 vs SORTENY
# ========================================
# Ejecutar en Raspberry Pi 3B+
#
# Autor: Christian Añez
# Fecha: 2026-01-05
#

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$HOME/benchmark_full_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/benchmark_full.log"

# Rutas
MHDC_DIR="$HOME/mhdc_source_code_20210208"
SORTENY_C_DIR="$HOME/sorteny_benchmark/c_code"
SORTENY_PY_DIR="$HOME/sorteny_benchmark/python_code"

# Imagen
IMAGE_BE="$MHDC_DIR/T31TCG_20230907T104629_5.8_512_512_2_1_0_be.raw"
IMAGE_LE_C="$SORTENY_C_DIR/data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw"
IMAGE_LE_PY="$SORTENY_PY_DIR/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw"

WIDTH=512
HEIGHT=512
BANDS=8
ORIGINAL_SIZE=$((WIDTH * HEIGHT * BANDS * 2))

log() {
    echo "$@" | tee -a "$LOG_FILE"
}

log "=========================================="
log "BENCHMARK COMPLETO: CCSDS 122 vs SORTENY"
log "=========================================="
log "Fecha: $(date)"
log "RAM: $(free -h | grep Mem | awk '{print $2}')"
log ""
log "Imagen: T31TCG_20230907T104629"
log "Tamaño: ${WIDTH}x${HEIGHT}x${BANDS} = $ORIGINAL_SIZE bytes (4 MB)"
log "=========================================="

#############################################
# CCSDS 122 LOSSLESS (U=0, D=0)
#############################################
log ""
log "=========================================="
log "1. CCSDS 122 - LOSSLESS (U=0, D=0)"
log "=========================================="

cd "$MHDC_DIR"
OUT_CCSDS_LL="$OUTPUT_DIR/ccsds_lossless.122"
OUT_CCSDS_LL_DEC="$OUTPUT_DIR/ccsds_lossless_decoded.raw"

START=$(date +%s.%N)
/usr/bin/time -v ./mhdcEncoder.sh \
    -i "$IMAGE_BE" \
    -x $WIDTH -y $HEIGHT -z $BANDS \
    -s yes \
    -o "$OUT_CCSDS_LL" \
    -t iwt \
    -U 0 -D 0 \
    -R 256 -S 131072 -W 3 \
    -w Integer \
    2>&1 | tee -a "$LOG_FILE"
END=$(date +%s.%N)
CCSDS_LL_TIME=$(echo "$END - $START" | bc)
CCSDS_LL_SIZE=$(stat -c%s "$OUT_CCSDS_LL")
log "TIEMPO_CCSDS_LOSSLESS_ENCODE: $CCSDS_LL_TIME segundos"
log "TAMAÑO_CCSDS_LOSSLESS: $CCSDS_LL_SIZE bytes"
log "RATIO_CCSDS_LOSSLESS: $(echo "scale=2; $ORIGINAL_SIZE / $CCSDS_LL_SIZE" | bc):1"

log ""
log "Descomprimiendo CCSDS Lossless..."
START=$(date +%s.%N)
/usr/bin/time -v ./mhdcDecoder.sh -i "$OUT_CCSDS_LL" -o "$OUT_CCSDS_LL_DEC" 2>&1 | tee -a "$LOG_FILE"
END=$(date +%s.%N)
log "TIEMPO_CCSDS_LOSSLESS_DECODE: $(echo "$END - $START" | bc) segundos"

#############################################
# CCSDS 122 NEAR-LOSSLESS (U=4, D=4)
#############################################
log ""
log "=========================================="
log "2. CCSDS 122 - NEAR-LOSSLESS (U=4, D=4)"
log "=========================================="

OUT_CCSDS_NL="$OUTPUT_DIR/ccsds_nearlossless.122"
OUT_CCSDS_NL_DEC="$OUTPUT_DIR/ccsds_nearlossless_decoded.raw"

START=$(date +%s.%N)
/usr/bin/time -v ./mhdcEncoder.sh \
    -i "$IMAGE_BE" \
    -x $WIDTH -y $HEIGHT -z $BANDS \
    -s yes \
    -o "$OUT_CCSDS_NL" \
    -t iwt \
    -U 4 -D 4 \
    -R 256 -S 131072 -W 3 \
    -w Integer \
    2>&1 | tee -a "$LOG_FILE"
END=$(date +%s.%N)
CCSDS_NL_TIME=$(echo "$END - $START" | bc)
CCSDS_NL_SIZE=$(stat -c%s "$OUT_CCSDS_NL")
log "TIEMPO_CCSDS_NEARLOSSLESS_ENCODE: $CCSDS_NL_TIME segundos"
log "TAMAÑO_CCSDS_NEARLOSSLESS: $CCSDS_NL_SIZE bytes"
log "RATIO_CCSDS_NEARLOSSLESS: $(echo "scale=2; $ORIGINAL_SIZE / $CCSDS_NL_SIZE" | bc):1"

log ""
log "Descomprimiendo CCSDS Near-Lossless..."
START=$(date +%s.%N)
/usr/bin/time -v ./mhdcDecoder.sh -i "$OUT_CCSDS_NL" -o "$OUT_CCSDS_NL_DEC" 2>&1 | tee -a "$LOG_FILE"
END=$(date +%s.%N)
log "TIEMPO_CCSDS_NEARLOSSLESS_DECODE: $(echo "$END - $START" | bc) segundos"

#############################################
# SORTENY C (Lambda = 0.1)
#############################################
log ""
log "=========================================="
log "3. SORTENY C (Lambda = 0.1)"
log "=========================================="

cd "$SORTENY_C_DIR"
OUT_SORTENY_C="$OUTPUT_DIR/sorteny_c.bin"

# Sintaxis: ./sorteny_compress <input.raw> <lambda> <output.bin> [weights_dir] [max_lambda]
START=$(date +%s.%N)
/usr/bin/time -v ./sorteny_compress \
    "$IMAGE_LE_C" \
    0.1 \
    "$OUT_SORTENY_C" \
    "./pesos_ieec050_spatial" \
    0.125 \
    2>&1 | tee -a "$LOG_FILE"
END=$(date +%s.%N)
SORTENY_C_TIME=$(echo "$END - $START" | bc)
SORTENY_C_SIZE=$(stat -c%s "$OUT_SORTENY_C")
log "TIEMPO_SORTENY_C: $SORTENY_C_TIME segundos"
log "TAMAÑO_SORTENY_C: $SORTENY_C_SIZE bytes"
log "RATIO_SORTENY_C: $(echo "scale=4; $ORIGINAL_SIZE / $SORTENY_C_SIZE" | bc):1"

#############################################
# SORTENY Python (Lambda = 0.1) - USAR python3.9
#############################################
log ""
log "=========================================="
log "4. SORTENY Python (Lambda = 0.1)"
log "=========================================="

cd "$SORTENY_PY_DIR"
OUT_SORTENY_PY="$OUTPUT_DIR/sorteny_py.bin"

# Sintaxis: python3.9 SORTENY_no_montsec.py compress <input> --quality <lambda> --width --height --bands
START=$(date +%s.%N)
/usr/bin/time -v python3.9 SORTENY_no_montsec.py compress \
    "$IMAGE_LE_PY" \
    --quality 0.1 \
    --width $WIDTH \
    --height $HEIGHT \
    --bands $BANDS \
    --endianess 1 \
    --datatype 2 \
    2>&1 | tee -a "$LOG_FILE"
END=$(date +%s.%N)
SORTENY_PY_TIME=$(echo "$END - $START" | bc)

# El output de Python puede tener nombre diferente, buscarlo
PY_OUTPUT=$(find . -name "*.bin" -newer "$IMAGE_LE_PY" -type f 2>/dev/null | head -1)
if [ -n "$PY_OUTPUT" ] && [ -f "$PY_OUTPUT" ]; then
    mv "$PY_OUTPUT" "$OUT_SORTENY_PY"
    SORTENY_PY_SIZE=$(stat -c%s "$OUT_SORTENY_PY")
    log "TIEMPO_SORTENY_PY: $SORTENY_PY_TIME segundos"
    log "TAMAÑO_SORTENY_PY: $SORTENY_PY_SIZE bytes"
    log "RATIO_SORTENY_PY: $(echo "scale=4; $ORIGINAL_SIZE / $SORTENY_PY_SIZE" | bc):1"
else
    log "TIEMPO_SORTENY_PY: $SORTENY_PY_TIME segundos"
    log "NOTA: No se encontró archivo de salida Python"
fi

#############################################
# MÉTRICAS DE CALIDAD CCSDS
#############################################
log ""
log "=========================================="
log "5. MÉTRICAS DE CALIDAD"
log "=========================================="

python3 << 'PYEOF'
import numpy as np
import os
import glob

# Buscar el directorio más reciente
dirs = glob.glob(os.path.expanduser('~/benchmark_full_*'))
if dirs:
    output_dir = max(dirs, key=os.path.getctime)
else:
    output_dir = '.'

image_be = os.path.expanduser('~/mhdc_source_code_20210208/T31TCG_20230907T104629_5.8_512_512_2_1_0_be.raw')

def calc_metrics(orig_path, recon_path):
    if not os.path.exists(recon_path):
        return None
    orig = np.fromfile(orig_path, dtype='>u2').astype(np.float64)
    recon = np.fromfile(recon_path, dtype='>u2').astype(np.float64)
    if len(orig) != len(recon):
        return None
    mse = np.mean((orig - recon) ** 2)
    if mse == 0:
        return {'psnr': float('inf'), 'mse': 0, 'mae': 0}
    psnr = 10 * np.log10((65535.0 ** 2) / mse)
    mae = np.mean(np.abs(orig - recon))
    max_err = np.max(np.abs(orig - recon))
    return {'psnr': psnr, 'mse': mse, 'mae': mae, 'max_err': max_err}

files = [
    ('CCSDS Lossless', f'{output_dir}/ccsds_lossless_decoded.raw'),
    ('CCSDS Near-Lossless', f'{output_dir}/ccsds_nearlossless_decoded.raw'),
]

print("="*70)
print(f"{'Método':<25} {'PSNR (dB)':>12} {'MSE':>12} {'MAE':>10} {'Max Err':>10}")
print("="*70)
for name, path in files:
    m = calc_metrics(image_be, path)
    if m:
        psnr_str = 'inf' if m['psnr'] == float('inf') else f"{m['psnr']:.2f}"
        print(f"{name:<25} {psnr_str:>12} {m['mse']:>12.2f} {m['mae']:>10.2f} {m['max_err']:>10.0f}")
print("="*70)
print()
print("Nota: Métricas de SORTENY requieren descompresión con TensorFlow (hacer en local)")

# Guardar CSV
with open(f'{output_dir}/quality_metrics.csv', 'w') as f:
    f.write('Method,PSNR_dB,MSE,MAE,MaxError\n')
    for name, path in files:
        m = calc_metrics(image_be, path)
        if m:
            psnr_str = 'inf' if m['psnr'] == float('inf') else f'{m["psnr"]:.4f}'
            f.write(f'{name.replace(" ","_")},{psnr_str},{m["mse"]:.4f},{m["mae"]:.4f},{m["max_err"]:.0f}\n')
PYEOF

#############################################
# RESUMEN FINAL
#############################################
log ""
log "=========================================="
log "RESUMEN FINAL"
log "=========================================="
log ""
log "Tamaño original: $ORIGINAL_SIZE bytes (4.00 MB)"
log ""
log "Comprimidos:"
log "---------------------------------------------------------------"
log "| Método                  | Tamaño (bytes) | Ratio   | Tiempo |"
log "---------------------------------------------------------------"

for f in "$OUTPUT_DIR"/*.122 "$OUTPUT_DIR"/*.bin; do
    if [ -f "$f" ]; then
        name=$(basename "$f")
        size=$(stat -c%s "$f")
        ratio=$(echo "scale=2; $ORIGINAL_SIZE / $size" | bc)
        log "| $(printf '%-23s' "$name") | $(printf '%14s' "$size") | $(printf '%6s' "$ratio"):1 |        |"
    fi
done
log "---------------------------------------------------------------"

log ""
log "=========================================="
log "Benchmark completado: $(date)"
log "Resultados en: $OUTPUT_DIR"
log "=========================================="

# Listar todo
ls -la "$OUTPUT_DIR"
