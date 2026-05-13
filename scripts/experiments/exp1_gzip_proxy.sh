#!/usr/bin/env bash
# ===========================================================================
# Experimento 1: Proxy gzip - demostrar que la calidad adaptativa reduce entropia
# ===========================================================================
# Comprime los bitstreams SORTENY ya generados con gzip -9 y compara los
# tamanos. El bitstream actual contiene cabecera + Q-map + latentes int32 sin
# codificador entropico final. Como todos los bitstreams comparados tienen el
# mismo tamano sin comprimir, gzip es un proxy reproducible de compresibilidad.
#
# Uso: bash scripts/experiments/exp1_gzip_proxy.sh
# ===========================================================================

set -euo pipefail

RESULTS_DIR="output/experiments/exp1_gzip_proxy"
mkdir -p "$RESULTS_DIR"

TSV="$RESULTS_DIR/gzip_compression_results.tsv"
echo -e "category\tlabel\tbitstream_file\traw_bytes\tgzip_bytes\tratio\treduction_pct" > "$TSV"

compress_and_log() {
    local category="$1"
    local label="$2"
    local bitstream_file="$3"

    if [ ! -f "$bitstream_file" ]; then
        echo "  SKIP: $bitstream_file (no existe)"
        return
    fi

    local raw_bytes
    raw_bytes=$(stat -c%s "$bitstream_file")

    # Comprimir con gzip -9 a un fichero temporal
    local tmp_gz
    tmp_gz=$(mktemp /tmp/sorteny_bitstream_XXXXXX.gz)
    gzip -9 -c "$bitstream_file" > "$tmp_gz"
    local gz_bytes
    gz_bytes=$(stat -c%s "$tmp_gz")
    rm -f "$tmp_gz"
    
    # Calcular ratio y reducción
    local ratio
    ratio=$(echo "scale=4; $raw_bytes / $gz_bytes" | bc)
    local reduction
    reduction=$(echo "scale=2; 100 * (1 - $gz_bytes / $raw_bytes)" | bc)
    
    echo -e "${category}\t${label}\t$(basename "$bitstream_file")\t${raw_bytes}\t${gz_bytes}\t${ratio}\t${reduction}%" >> "$TSV"
    printf "  %-40s  raw=%7s  gz=%7s  ratio=%5s  -%s%%\n" "$label" "$raw_bytes" "$gz_bytes" "$ratio" "$reduction"
}

echo "=================================================================="
echo "  Experimento 1: Proxy gzip sobre bitstreams SORTENY"
echo "=================================================================="
echo ""

# --- Grupo 1: Constant Q vs Fixed-Quality Adaptive ---
echo "--- GRUPO 1: Constant Q (204) vs Fixed-Quality Adaptive ---"
compress_and_log "baseline" "constant_q204" \
    "output/checkpoints/20260507_c_fixed_quality_qmap_wide/latent_from_q204.bin"
compress_and_log "adaptive_fq" "fq_target_psnr_76_8" \
    "output/checkpoints/20260507_c_fixed_quality_qmap_wide/latent_target_psnr_76_8.bin"
echo ""

# --- Grupo 2: Semantic vegetation (boost-only) ---
echo "--- GRUPO 2: Semántica vegetation boost-only ---"
compress_and_log "semantic_boost" "vegetation_boost8" \
    "output/checkpoints/20260508_semantic_qmap_c/latent_semantic_vegetation.bin"
echo ""

# --- Grupo 3: Semantic vegetation con threshold variable ---
echo "--- GRUPO 3: Semántica vegetation - threshold variable ---"
for f in output/checkpoints/20260509_semantic_validation_vegetation/latent_*.bin; do
    label=$(basename "$f" .bin | sed 's/latent_//')
    compress_and_log "threshold_sweep" "$label" "$f"
done
echo ""

# --- Grupo 4: Semantic focus (foreground boost + background penalty) ---
echo "--- GRUPO 4: Semántica focus (fg boost + bg penalty) ---"
for f in output/checkpoints/20260510_semantic_focus_vegetation/latent_*.bin; do
    label=$(basename "$f" .bin | sed 's/latent_//')
    compress_and_log "focus_penalty" "$label" "$f"
done
echo ""

# --- Grupo 5: Semantic focus con background Q fijo ---
echo "--- GRUPO 5: Semántica focus con background Q fijo ---"
for f in output/checkpoints/20260511_semantic_background_q_vegetation/latent_*.bin; do
    label=$(basename "$f" .bin | sed 's/latent_//')
    compress_and_log "focus_bgq" "$label" "$f"
done
echo ""

# --- Grupo 6: Manual ROI ---
echo "--- GRUPO 6: Manual ROI ---"
compress_and_log "manual_roi" "manual_roi_focus" \
    "output/checkpoints/20260514_manual_roi_focus/latent_manual_roi.bin"
echo ""

echo "=================================================================="
echo "  Resultados guardados en: $TSV"
echo "=================================================================="
echo ""

# Generar resumen ordenado por ratio de compresión
echo "--- TOP 10 mejores ratios (más comprimible = más entropía reducida) ---"
tail -n +2 "$TSV" | sort -t$'\t' -k6 -rn | head -10 | \
    awk -F'\t' '{printf "  %-40s ratio=%s (%s)\n", $2, $6, $7}'
echo ""

echo "--- TOP 5 peores ratios (baseline) ---"
tail -n +2 "$TSV" | sort -t$'\t' -k6 -n | head -5 | \
    awk -F'\t' '{printf "  %-40s ratio=%s (%s)\n", $2, $6, $7}'
