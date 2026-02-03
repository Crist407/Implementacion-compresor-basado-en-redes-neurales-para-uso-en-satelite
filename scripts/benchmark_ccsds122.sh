#!/bin/bash
#
# Benchmark CCSDS 122 vs SORTENY en Raspberry Pi
# Autor: Christian Añez
# Fecha: 2026-01-05
#

set -e

# Configuración
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MHDC_DIR="$HOME/mhdc_source_code_20210208"
SORTENY_C_DIR="$HOME/SortenyC_Raspberry"
SORTENY_PY_DIR="$HOME/SortenyPython_Raspberry_sinCodificador"

# Imagen de prueba
INPUT_IMAGE="$HOME/SortenyC_Raspberry/data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw"
INPUT_IMAGE_BE="$MHDC_DIR/T31TCG_20230907T104629_5.8_512_512_2_1_0_be.raw"

# Dimensiones de la imagen
WIDTH=512
HEIGHT=512
BANDS=8

# Archivos de salida
OUTPUT_DIR="$HOME/benchmark_comparison_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/benchmark_comparison.log"

# Función para medir tiempo y recursos
measure() {
    local name=$1
    shift
    echo "========================================"
    echo "Ejecutando: $name"
    echo "Comando: $@"
    echo "========================================"
    
    /usr/bin/time -v "$@" 2>&1
}

log() {
    echo "$@" | tee -a "$LOG_FILE"
}

log "=========================================="
log "BENCHMARK COMPARATIVO: SORTENY vs CCSDS 122"
log "=========================================="
log "Fecha: $(date)"
log "Imagen: T31TCG_20230907T104629"
log "Dimensiones: ${WIDTH}x${HEIGHT}x${BANDS} (uint16)"
log "=========================================="
log ""

# Verificar que existen las imágenes
if [ ! -f "$INPUT_IMAGE" ]; then
    log "ERROR: No existe la imagen de entrada: $INPUT_IMAGE"
    exit 1
fi

# Crear imagen big endian si no existe
if [ ! -f "$INPUT_IMAGE_BE" ]; then
    log "Convirtiendo imagen a big endian..."
    python3 -c "
import numpy as np
data = np.fromfile('$INPUT_IMAGE', dtype='<u2')  # little endian
data.astype('>u2').tofile('$INPUT_IMAGE_BE')  # big endian
print('Conversión completada')
"
fi

log "=========================================="
log "1. BENCHMARK CCSDS 122 (MHDC)"
log "=========================================="

# CCSDS 122 con diferentes configuraciones
# Modo IWT (Integer Wavelet Transform) - lossless
log ""
log "--- CCSDS 122: Modo IWT (Lossless) ---"
cd "$MHDC_DIR"

OUTPUT_CCSDS_IWT="$OUTPUT_DIR/output_ccsds_iwt.122"
OUTPUT_CCSDS_IWT_DEC="$OUTPUT_DIR/output_ccsds_iwt_decoded.raw"

{
    START_TIME=$(date +%s.%N)
    /usr/bin/time -v ./mhdcEncoder.sh \
        -i "$INPUT_IMAGE_BE" \
        -x $WIDTH -y $HEIGHT -z $BANDS \
        -s yes \
        -o "$OUTPUT_CCSDS_IWT" \
        -t iwt \
        -U 4 -D 4 \
        -R 256 -S 131072 -W 3 \
        -w Integer \
        2>&1
    END_TIME=$(date +%s.%N)
    echo "TIEMPO_COMPRESION_IWT: $(echo "$END_TIME - $START_TIME" | bc) segundos"
} 2>&1 | tee -a "$LOG_FILE"

CCSDS_IWT_SIZE=$(stat -c%s "$OUTPUT_CCSDS_IWT" 2>/dev/null || echo "0")
log "Tamaño comprimido IWT: $CCSDS_IWT_SIZE bytes"
log "Ratio de compresión IWT: $(echo "scale=4; $((WIDTH*HEIGHT*BANDS*2)) / $CCSDS_IWT_SIZE" | bc)"

# Descomprimir
log ""
log "Descomprimiendo CCSDS 122 IWT..."
{
    START_TIME=$(date +%s.%N)
    /usr/bin/time -v ./mhdcDecoder.sh \
        -i "$OUTPUT_CCSDS_IWT" \
        -o "$OUTPUT_CCSDS_IWT_DEC" \
        2>&1
    END_TIME=$(date +%s.%N)
    echo "TIEMPO_DESCOMPRESION_IWT: $(echo "$END_TIME - $START_TIME" | bc) segundos"
} 2>&1 | tee -a "$LOG_FILE"

# CCSDS 122 con rate allocation (lossy)
log ""
log "--- CCSDS 122: Modo IWT con Rate Allocation (Lossy) ---"

# Calcular tamaño objetivo similar al de SORTENY (~12.5MB para los latentes)
TARGET_RATE=12583946  # mismo tamaño que output de SORTENY

OUTPUT_CCSDS_LOSSY="$OUTPUT_DIR/output_ccsds_lossy.122"
OUTPUT_CCSDS_LOSSY_DEC="$OUTPUT_DIR/output_ccsds_lossy_decoded.raw"

{
    START_TIME=$(date +%s.%N)
    /usr/bin/time -v ./mhdcEncoder.sh \
        -i "$INPUT_IMAGE_BE" \
        -x $WIDTH -y $HEIGHT -z $BANDS \
        -s yes \
        -o "$OUTPUT_CCSDS_LOSSY" \
        -t iwt \
        -U 4 -D 4 \
        -R 256 -S 131072 -W 3 \
        -w Integer \
        -r $TARGET_RATE \
        -a reverse-waterfill \
        2>&1
    END_TIME=$(date +%s.%N)
    echo "TIEMPO_COMPRESION_LOSSY: $(echo "$END_TIME - $START_TIME" | bc) segundos"
} 2>&1 | tee -a "$LOG_FILE"

CCSDS_LOSSY_SIZE=$(stat -c%s "$OUTPUT_CCSDS_LOSSY" 2>/dev/null || echo "0")
log "Tamaño comprimido Lossy: $CCSDS_LOSSY_SIZE bytes"
log "Ratio de compresión Lossy: $(echo "scale=4; $((WIDTH*HEIGHT*BANDS*2)) / $CCSDS_LOSSY_SIZE" | bc)"

# Descomprimir
log ""
log "Descomprimiendo CCSDS 122 Lossy..."
{
    START_TIME=$(date +%s.%N)
    /usr/bin/time -v ./mhdcDecoder.sh \
        -i "$OUTPUT_CCSDS_LOSSY" \
        -o "$OUTPUT_CCSDS_LOSSY_DEC" \
        2>&1
    END_TIME=$(date +%s.%N)
    echo "TIEMPO_DESCOMPRESION_LOSSY: $(echo "$END_TIME - $START_TIME" | bc) segundos"
} 2>&1 | tee -a "$LOG_FILE"

# CCSDS 122 con alta compresión
log ""
log "--- CCSDS 122: Alta compresión (1 bpp) ---"

# 1 bit per pixel = width * height * bands * 1 / 8 bytes
TARGET_1BPP=$((WIDTH * HEIGHT * BANDS / 8))

OUTPUT_CCSDS_1BPP="$OUTPUT_DIR/output_ccsds_1bpp.122"
OUTPUT_CCSDS_1BPP_DEC="$OUTPUT_DIR/output_ccsds_1bpp_decoded.raw"

{
    START_TIME=$(date +%s.%N)
    /usr/bin/time -v ./mhdcEncoder.sh \
        -i "$INPUT_IMAGE_BE" \
        -x $WIDTH -y $HEIGHT -z $BANDS \
        -s yes \
        -o "$OUTPUT_CCSDS_1BPP" \
        -t iwt \
        -U 4 -D 4 \
        -R 256 -S 131072 -W 3 \
        -w Integer \
        -r $TARGET_1BPP \
        -a reverse-waterfill \
        2>&1
    END_TIME=$(date +%s.%N)
    echo "TIEMPO_COMPRESION_1BPP: $(echo "$END_TIME - $START_TIME" | bc) segundos"
} 2>&1 | tee -a "$LOG_FILE"

CCSDS_1BPP_SIZE=$(stat -c%s "$OUTPUT_CCSDS_1BPP" 2>/dev/null || echo "0")
log "Tamaño comprimido 1bpp: $CCSDS_1BPP_SIZE bytes"

# Descomprimir
{
    START_TIME=$(date +%s.%N)
    /usr/bin/time -v ./mhdcDecoder.sh \
        -i "$OUTPUT_CCSDS_1BPP" \
        -o "$OUTPUT_CCSDS_1BPP_DEC" \
        2>&1
    END_TIME=$(date +%s.%N)
    echo "TIEMPO_DESCOMPRESION_1BPP: $(echo "$END_TIME - $START_TIME" | bc) segundos"
} 2>&1 | tee -a "$LOG_FILE"

log ""
log "=========================================="
log "2. BENCHMARK SORTENY C"
log "=========================================="

cd "$SORTENY_C_DIR"
OUTPUT_SORTENY_C="$OUTPUT_DIR/output_sorteny_c.bin"

{
    START_TIME=$(date +%s.%N)
    /usr/bin/time -v ./sorteny_compress \
        -i "data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw" \
        -o "$OUTPUT_SORTENY_C" \
        -w "../weights/pesos_ieec050_transposed" \
        -W 512 -H 512 -B 8 \
        -q 0.1 \
        2>&1
    END_TIME=$(date +%s.%N)
    echo "TIEMPO_SORTENY_C: $(echo "$END_TIME - $START_TIME" | bc) segundos"
} 2>&1 | tee -a "$LOG_FILE"

SORTENY_C_SIZE=$(stat -c%s "$OUTPUT_SORTENY_C" 2>/dev/null || echo "0")
log "Tamaño comprimido SORTENY C: $SORTENY_C_SIZE bytes"

log ""
log "=========================================="
log "3. BENCHMARK SORTENY Python"
log "=========================================="

cd "$SORTENY_PY_DIR"
OUTPUT_SORTENY_PY="$OUTPUT_DIR/output_sorteny_py.bin"

{
    START_TIME=$(date +%s.%N)
    /usr/bin/time -v python3 SORTENY_no_montsec.py \
        --input "../SortenyC_Raspberry/data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw" \
        --output "$OUTPUT_SORTENY_PY" \
        --width 512 --height 512 --bands 8 \
        --lambda_param 0.1 \
        2>&1
    END_TIME=$(date +%s.%N)
    echo "TIEMPO_SORTENY_PY: $(echo "$END_TIME - $START_TIME" | bc) segundos"
} 2>&1 | tee -a "$LOG_FILE"

SORTENY_PY_SIZE=$(stat -c%s "$OUTPUT_SORTENY_PY" 2>/dev/null || echo "0")
log "Tamaño comprimido SORTENY Python: $SORTENY_PY_SIZE bytes"

log ""
log "=========================================="
log "4. CÁLCULO DE MÉTRICAS DE CALIDAD"
log "=========================================="

python3 << 'PYTHON_SCRIPT'
import numpy as np
import os

def calculate_metrics(original, reconstructed, name):
    """Calcula PSNR, MSE, MAE entre original y reconstruido"""
    if not os.path.exists(reconstructed):
        print(f"{name}: Archivo no encontrado - {reconstructed}")
        return None
    
    orig = np.fromfile(original, dtype=np.uint16)
    
    # Detectar si es big endian (archivos CCSDS)
    if '_be' in original or 'ccsds' in reconstructed.lower():
        recon = np.fromfile(reconstructed, dtype='>u2').astype(np.uint16)
    else:
        recon = np.fromfile(reconstructed, dtype=np.uint16)
    
    if orig.shape != recon.shape:
        print(f"{name}: Tamaños diferentes - orig={orig.shape}, recon={recon.shape}")
        return None
    
    mse = np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)
    mae = np.mean(np.abs(orig.astype(np.float64) - recon.astype(np.float64)))
    max_val = 65535.0
    psnr = 10 * np.log10((max_val ** 2) / mse) if mse > 0 else float('inf')
    
    print(f"{name}:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print()
    
    return {'mse': mse, 'mae': mae, 'psnr': psnr}

# Imagen original
original_le = os.path.expanduser("~/SortenyC_Raspberry/data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw")
original_be = os.path.expanduser("~/mhdc_source_code_20210208/T31TCG_20230907T104629_5.8_512_512_2_1_0_be.raw")
output_dir = os.path.expanduser("~/benchmark_comparison_$TIMESTAMP")

print("="*60)
print("MÉTRICAS DE CALIDAD")
print("="*60)
print()

# CCSDS 122 IWT (lossless)
calculate_metrics(original_be, f"{output_dir}/output_ccsds_iwt_decoded.raw", "CCSDS 122 IWT (Lossless)")

# CCSDS 122 Lossy
calculate_metrics(original_be, f"{output_dir}/output_ccsds_lossy_decoded.raw", "CCSDS 122 Lossy")

# CCSDS 122 1bpp
calculate_metrics(original_be, f"{output_dir}/output_ccsds_1bpp_decoded.raw", "CCSDS 122 1bpp")

print("Nota: Métricas de SORTENY requieren descompresión con TensorFlow (ejecutar en local)")
PYTHON_SCRIPT

log ""
log "=========================================="
log "5. RESUMEN DE TAMAÑOS"
log "=========================================="

ORIGINAL_SIZE=$((WIDTH * HEIGHT * BANDS * 2))
log "Tamaño original: $ORIGINAL_SIZE bytes (4 MB)"
log ""
log "Comprimidos:"
log "  CCSDS 122 IWT (lossless): $CCSDS_IWT_SIZE bytes (ratio: $(echo "scale=2; $ORIGINAL_SIZE / $CCSDS_IWT_SIZE" | bc):1)"
log "  CCSDS 122 Lossy:          $CCSDS_LOSSY_SIZE bytes (ratio: $(echo "scale=2; $ORIGINAL_SIZE / $CCSDS_LOSSY_SIZE" | bc):1)"
log "  CCSDS 122 1bpp:           $CCSDS_1BPP_SIZE bytes (ratio: $(echo "scale=2; $ORIGINAL_SIZE / $CCSDS_1BPP_SIZE" | bc):1)"
log "  SORTENY C:                $SORTENY_C_SIZE bytes (ratio: $(echo "scale=2; $ORIGINAL_SIZE / $SORTENY_C_SIZE" | bc):1)"
log "  SORTENY Python:           $SORTENY_PY_SIZE bytes (ratio: $(echo "scale=2; $ORIGINAL_SIZE / $SORTENY_PY_SIZE" | bc):1)"

log ""
log "=========================================="
log "Benchmark completado: $(date)"
log "Resultados en: $OUTPUT_DIR"
log "=========================================="

# Listar archivos generados
ls -la "$OUTPUT_DIR"
