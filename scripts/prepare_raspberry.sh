#!/bin/bash
# =============================================================================
# Script para preparar y enviar archivos a la Raspberry Pi
# Ejecutar desde el PC local
# =============================================================================

# Configuración
RASPBERRY_HOST="${RASPBERRY_HOST:-raspberry@158.109.79.167}"
REMOTE_DIR="/home/raspberry/sorteny_benchmark"

echo "=============================================================="
echo " SORTENY - Preparar Raspberry Pi para Benchmark"
echo "=============================================================="
echo " Host: $RASPBERRY_HOST"
echo " Directorio remoto: $REMOTE_DIR"
echo "=============================================================="

# Archivos a transferir
LOCAL_C_DIR="SortenyC_Raspberry/sorteny_c_fixed"
LOCAL_PY_DIR="SortenyPython_Raspberry_sinCodificador"

# Crear directorio remoto
echo ""
echo "=== Creando estructura de directorios en Raspberry ==="
ssh "$RASPBERRY_HOST" "mkdir -p $REMOTE_DIR/{c_code,python_code,data,results}"

# Transferir código C
echo ""
echo "=== Transfiriendo código C ==="
rsync -avz --progress \
    --exclude '*.o' \
    --exclude 'sorteny_compress' \
    --exclude '*.Zone.Identifier' \
    --exclude 'benchmark_results/' \
    "$LOCAL_C_DIR/" "$RASPBERRY_HOST:$REMOTE_DIR/c_code/"

# Transferir código Python
echo ""
echo "=== Transfiriendo código Python ==="
rsync -avz --progress \
    --exclude '*.Zone.Identifier' \
    --exclude '*.bin' \
    --exclude '__pycache__/' \
    "$LOCAL_PY_DIR/" "$RASPBERRY_HOST:$REMOTE_DIR/python_code/"

# Transferir imagen de prueba
echo ""
echo "=== Transfiriendo imagen de prueba ==="
if [ -f "$LOCAL_C_DIR/data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw" ]; then
    rsync -avz --progress \
        "$LOCAL_C_DIR/data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw" \
        "$RASPBERRY_HOST:$REMOTE_DIR/data/"
fi

# Crear script de ejecución en la Raspberry
echo ""
echo "=== Creando script de benchmark en Raspberry ==="
ssh "$RASPBERRY_HOST" "cat > $REMOTE_DIR/run_benchmark.sh" << 'REMOTE_SCRIPT'
#!/bin/bash
# Script maestro de benchmark en Raspberry Pi
set -e

BENCHMARK_DIR="/home/raspberry/sorteny_benchmark"
RESULTS_DIR="$BENCHMARK_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
IMAGE="$BENCHMARK_DIR/data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw"
LAMBDA="${1:-0.1}"

mkdir -p "$RESULTS_DIR"

echo "=============================================================="
echo " SORTENY Benchmark - Raspberry Pi 3B+"
echo " Timestamp: $TIMESTAMP"
echo " Lambda: $LAMBDA"
echo "=============================================================="

# Info del sistema
echo ""
echo "=== Sistema ==="
uname -a
cat /proc/cpuinfo | grep -E "model name|Hardware" | head -2
free -h
cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null | awk '{print "Temperatura CPU: " $1/1000 "°C"}' || true

# =============================================================================
# BENCHMARK C
# =============================================================================
echo ""
echo "=============================================================="
echo " BENCHMARK C"
echo "=============================================================="
cd "$BENCHMARK_DIR/c_code"

# Compilar
echo "Compilando..."
make clean 2>/dev/null || true
make -j4

OUTPUT_C="$RESULTS_DIR/output_c_$TIMESTAMP.bin"
LOG_C="$RESULTS_DIR/benchmark_c_$TIMESTAMP.log"

echo ""
echo "Ejecutando compresión C..."
/usr/bin/time -v ./sorteny_compress "$IMAGE" "$LAMBDA" "$OUTPUT_C" pesos_ieec050_spatial 0.125 2>&1 | tee "$LOG_C"

if [ -f "$OUTPUT_C" ]; then
    ls -lh "$OUTPUT_C"
    md5sum "$OUTPUT_C" > "$RESULTS_DIR/checksum_c_$TIMESTAMP.md5"
fi

# =============================================================================
# BENCHMARK PYTHON (si TensorFlow está disponible)
# =============================================================================
echo ""
echo "=============================================================="
echo " BENCHMARK PYTHON"
echo "=============================================================="
cd "$BENCHMARK_DIR/python_code"

# Buscar Python con TensorFlow
PYTHON_CMD=""
for py in python3.9 python3.8 python3; do
    if command -v $py &> /dev/null && $py -c "import tensorflow" 2>/dev/null; then
        PYTHON_CMD=$py
        break
    fi
done

if [ -n "$PYTHON_CMD" ]; then
    echo "Usando: $PYTHON_CMD"
    $PYTHON_CMD -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
    
    # Copiar imagen si es necesario
    IMAGE_NAME=$(basename "$IMAGE")
    [ -f "$IMAGE_NAME" ] || cp "$IMAGE" .
    
    OUTPUT_PY="$RESULTS_DIR/output_py_$TIMESTAMP.bin"
    LOG_PY="$RESULTS_DIR/benchmark_py_$TIMESTAMP.log"
    
    echo ""
    echo "Ejecutando compresión Python..."
    /usr/bin/time -v $PYTHON_CMD SORTENY_no_montsec.py --model_name ieec050 compress \
        "$IMAGE_NAME" --quality "$LAMBDA" 2>&1 | tee "$LOG_PY"
    
    if [ -f "${IMAGE_NAME}.bin" ]; then
        mv "${IMAGE_NAME}.bin" "$OUTPUT_PY"
        ls -lh "$OUTPUT_PY"
        md5sum "$OUTPUT_PY" > "$RESULTS_DIR/checksum_py_$TIMESTAMP.md5"
    fi
else
    echo "⚠️  TensorFlow no disponible, saltando benchmark Python"
fi

# =============================================================================
# RESUMEN
# =============================================================================
echo ""
echo "=============================================================="
echo " RESULTADOS"
echo "=============================================================="
ls -lh "$RESULTS_DIR/"*"$TIMESTAMP"*

echo ""
echo "Para transferir resultados al PC:"
echo "  scp -r $RASPBERRY_HOST:$RESULTS_DIR ."
echo ""
echo "Benchmark completado: $TIMESTAMP"
REMOTE_SCRIPT

ssh "$RASPBERRY_HOST" "chmod +x $REMOTE_DIR/run_benchmark.sh"

echo ""
echo "=============================================================="
echo " ✅ Preparación completada"
echo "=============================================================="
echo ""
echo "Para ejecutar el benchmark en la Raspberry:"
echo "  ssh $RASPBERRY_HOST"
echo "  cd $REMOTE_DIR"
echo "  ./run_benchmark.sh [lambda]  # default: 0.1"
echo ""
echo "Para recuperar resultados:"
echo "  ./scripts/fetch_raspberry_results.sh"
echo ""
