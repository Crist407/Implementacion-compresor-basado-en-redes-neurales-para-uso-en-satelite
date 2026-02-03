#!/bin/bash
# =============================================================================
# Script para recuperar resultados de la Raspberry Pi
# Ejecutar desde el PC local
# =============================================================================

# Configuración
RASPBERRY_HOST="${RASPBERRY_HOST:-raspberry@158.109.79.167}"
REMOTE_RESULTS="/home/raspberry/sorteny_benchmark/results"
LOCAL_RESULTS="raspberry_results"

echo "=============================================================="
echo " SORTENY - Recuperar Resultados de Raspberry Pi"
echo "=============================================================="
echo " Host: $RASPBERRY_HOST"
echo " Remoto: $REMOTE_RESULTS"
echo " Local: $LOCAL_RESULTS"
echo "=============================================================="

# Crear directorio local
mkdir -p "$LOCAL_RESULTS"

# Transferir resultados
echo ""
echo "=== Descargando resultados ==="
rsync -avz --progress "$RASPBERRY_HOST:$REMOTE_RESULTS/" "$LOCAL_RESULTS/"

# Listar archivos descargados
echo ""
echo "=== Archivos descargados ==="
ls -lh "$LOCAL_RESULTS/"

# Ejecutar análisis automático
echo ""
echo "=== Ejecutando análisis ==="
if [ -f "scripts/analyze_benchmark_results.py" ]; then
    python3 scripts/analyze_benchmark_results.py "$LOCAL_RESULTS"
else
    echo "Script de análisis no encontrado. Ejecutar manualmente:"
    echo "  python3 scripts/analyze_benchmark_results.py $LOCAL_RESULTS"
fi

echo ""
echo "=============================================================="
echo " ✅ Resultados recuperados en: $LOCAL_RESULTS/"
echo "=============================================================="
echo ""
echo "Comandos útiles:"
echo "  # Analizar resultados:"
echo "  python3 scripts/analyze_benchmark_results.py $LOCAL_RESULTS"
echo ""
echo "  # Descomprimir y comparar:"
echo "  python3 scripts/decompress_and_compare.py $LOCAL_RESULTS/output_c_*.bin \\"
echo "      --compare $LOCAL_RESULTS/output_py_*.bin \\"
echo "      --model_dir models/ieec050"
echo ""
