# SORTENY: Compresor de Imágenes Satelitales Sentinel-2

Implementación en C de un compresor de imágenes hiperespectrales basado en redes neuronales, optimizado para dispositivos embebidos (Raspberry Pi). Incluye validación contra el modelo TensorFlow original.

> **Trabajo de Fin de Grado** - Grado en Ingeniería Informática  
> Cristhian Omar Añez López · Universitat Autònoma de Barcelona · 2025

## Descripción

Este proyecto implementa el encoder SORTENY (desarrollado por el IEEC) en C puro, eliminando la dependencia de TensorFlow para inferencia en dispositivos con recursos limitados. El encoder procesa imágenes Sentinel-2 de 8 bandas espectrales y genera una representación latente comprimida.

**Características principales:**
- Encoder completo en C con optimizaciones NEON para ARM
- Validación bit-exact contra TensorFlow
- Soporte para Raspberry Pi 3B+/4
- Pipeline: Transformada Espectral → Analysis Transform (Conv2D + GDN) → Modulación

## Estructura del Proyecto

```
├── src/
│   ├── c/                      # Implementación C del encoder
│   │   ├── main.c              # Pipeline principal
│   │   ├── sorteny_layers.c    # Capas: Conv2D, GDN, Dense, ReLU
│   │   ├── sorteny_model.c     # Carga de pesos desde TSV/bin
│   │   └── io_helpers.c        # E/S de imágenes RAW
│   │
│   └── python/
│       ├── core/               # Lógica principal
│       │   ├── validar_python.py   # Validador de referencia TensorFlow
│       │   ├── pesos.py            # Extracción de pesos del modelo
│       │   └── pesos_ieec.py       # Extracción modelo ieec050
│       ├── benchmark/          # Scripts de rendimiento
│       ├── analysis/           # Comparación C vs Python
│       ├── utils/              # Utilidades diversas
│       └── reference/          # Código original IEEC (SORTENY.py)
│
├── scripts/                    # Scripts bash (benchmark, deploy)
├── models/                     # Modelos TensorFlow SavedModel
├── weights/                    # Pesos exportados (float32 binarios)
├── data/                       # Imágenes de prueba (RAW BSQ u16)
├── docs/                       # Documentación e informes
└── debug_dumps/                # Volcados de debug (ignorado en git)
```

## Requisitos

### Sistema
- Linux (probado en Ubuntu 22.04 y Raspberry Pi OS 64-bit)
- GCC con soporte C11
- Make

### Para compilación ARM (Raspberry Pi)
- GCC con soporte NEON (automático en aarch64)

### Para validación Python
- Python 3.9+
- Dependencias: `pip install -r requirements.txt`
  - tensorflow==2.14.1
  - tensorflow-compression==2.14.1
  - numpy>=1.26,<2.0

## Inicio Rápido

### 1. Compilar el encoder C

```bash
# Compilación release (optimizado)
make clean && make MODE=release

# Para Raspberry Pi (auto-detecta arquitectura)
make clean && make MODE=release
```

### 2. Ejecutar el encoder

```bash
./sorteny_compressor <imagen.raw> <lambda> <salida.bin> [weights_dir] [max_lambda]

# Ejemplo:
./sorteny_compressor data/T31TCG_...raw 0.01 output/latent.bin weights/pesos_bin 0.125
```

### 3. Validar contra Python (opcional)

```bash
# Activar entorno virtual
source .venv/bin/activate

# Generar ground truth
python src/python/core/validar_python.py

# Comparar salidas
python src/python/analysis/compare_products.py --C debug_dumps --PY debug_dumps
```

## Compilación

### Modos de compilación

```bash
# Release (optimizado para producción)
make MODE=release

# Debug (con símbolos para depuración)
make MODE=debug
```

### Opciones del Makefile

| Variable | Descripción |
|----------|-------------|
| `MODE=release\|debug` | Nivel de optimización |
| `OMP=1` | Habilitar OpenMP |
| `RPI_ARCH=rpi3\|rpi4` | Optimizar para Raspberry Pi específica |

## Variables de Entorno

### Modo de ejecución

| Variable | Descripción |
|----------|-------------|
| `STRICT_PARITY=1` | Modo determinista + redondeo half-to-even |
| `USE_HALF_EVEN=1` | Redondeo half-to-even (como tf.round) |

### Volcados de debug

| Variable | Archivo generado |
|----------|-----------------|
| `DUMP_SPECTRAL=1` | spectral_c.bin |
| `DUMP_STAGES=1` | conv0_pre_c.bin, gdn0_c.bin, etc. |
| `DUMP_Y_PRE=1` | Y_pre_c.bin |
| `DUMP_M=1` | M_c.bin |
| `DUMP_Y_FLOAT=1` | Y_float_c.bin |

## Arquitectura del Encoder

```
Imagen RAW (8 bandas × 512×512)
           │
           ▼
┌─────────────────────┐
│ Transformada        │
│ Espectral (8×8)     │
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│ Analysis Transform  │
│ Conv2D 5×5 + GDN ×4 │
│ (stride=2 cada capa)│
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│ Modulating Transform│
│ Dense + ReLU        │
│ (escala según λ)    │
└─────────────────────┘
           │
           ▼
Latente cuantizado (8 × 384 × 32×32)
```

## Notas Técnicas

### Convolución
- Semántica de correlación (como TensorFlow)
- Padding `same_zeros` de SignalConv2D
- Kernels 5×5 con stride 2

### GDN (Generalized Divisive Normalization)
- Implementación exacta de tensorflow-compression
- Fórmula: `y = x / (beta + sum(gamma * |x|))`
- alpha=1, epsilon=1

### Redondeo
- Por defecto: `roundf` (half-away-from-zero)
- Con `USE_HALF_EVEN=1`: half-to-even (como tf.round)

## Benchmarks

Resultados en Raspberry Pi 4 (4GB) con imagen 512×512×8 bandas:

| Métrica | C (OpenMP) | Python TensorFlow |
|---------|------------|-------------------|
| **Tiempo total** | 5:08 (308s) | 7:31 (451s) |
| **Memoria pico** | 88 MB | 816 MB |
| **Speedup** | **1.46×** | - |
| **Ahorro memoria** | **9.3×** | - |

*Nota: El speedup en tiempo es modesto, pero el ahorro de memoria permite ejecutar en dispositivos con RAM limitada.*

## Licencia

Este proyecto es parte de un Trabajo de Fin de Grado. Consultar con el autor antes de usar.

## Agradecimientos

- Institut d'Estudis Espacials de Catalunya (IEEC) - Modelo SORTENY original
- Universitat Autònoma de Barcelona - Supervisión académica

## Contacto

Christian Añez - 1635157@uab.cat