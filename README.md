# SORTENY: Compresor/Descompresor de Imágenes Satelitales Sentinel-2

Implementación en C puro del pipeline completo de compresión y descompresión SORTENY, optimizado intensamente para dispositivos embebidos (Raspberry Pi). Logra una **paridad exacta garantizada de ~76 dB** con respecto a su modelo de referencia en TensorFlow.

> **Trabajo de Fin de Grado** - Grado en Ingeniería Informática  
> Cristhian Omar Añez López · Universitat Autònoma de Barcelona · 2025

## Descripción

Este proyecto implementa el codificador y el decodificador de la red neuronal SORTENY (desarrollada originalmente en TensorFlow por el IEEC). Se reescribió la inferencia en C 11 sin dependencias de terceros para correr fluidamente en satélites y dispositivos con recursos mínimos.

Estructura central:
- Encoder y Decoder íntegros en C con optimizaciones matriciales y SIMD/NEON para ARM.
- Paridad matemática (*half-to-even rounding* y *bitwise transforms*).
- Uso intensivo del stack en capas GDN e IGDN minimizando el impacto de malloc.

## Estructura del Proyecto

```
├── src/
│   ├── c/                      # Implementación C
│   │   ├── main.c              # Compresor (Encoder)
│   │   ├── main_decompress.c   # Descompresor (Decoder)
│   │   ├── sorteny_layers.c    # Capas: Conv2D, GDN, IGDN, Dense, ReLU
│   │   ├── sorteny_model.c     # Parámetros y fallback de pesos
│   │   └── io_helpers.c        # Lectura y manipulación BSQ/planar flotante
│   │
│   └── python/                 # Lógica de Python (Scripts de apoyo y validación)
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

### 2. Ejecutar el pipeline (Enlace y Desenlace)

```bash
# Comprimir:
./sorteny_compressor <imagen_original.raw> <lambda> <latente.bin> [dir_pesos_encoder] [max_lambda]

# Ejemplo:
./sorteny_compressor data/T31TCG_20230907...raw 0.1 output/latent.bin weights/encoder 0.125

# Descomprimir:
./sorteny_decompressor <latente.bin> <imagen_reconstruida.raw> [dir_pesos_decoder] [max_lambda]

# Ejemplo:
./sorteny_decompressor output/latent.bin output/reconstructed.raw weights/decoder 0.125
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
           │
           ▼
┌─────────────────────┐
│ Modulating Inverse  │
│ Dense + ReLU        │
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│ Synthesis Transform │
│ IGDN ×3 + Conv2D 5×5│
│ (Upsampling x2)     │
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│ Transformada        │
│ Espectral Inversa   │
└─────────────────────┘
           │
           ▼
Imagen Reconstruida RAW BSQ
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

## Benchmarks Oficiales (Raspberry Pi 4)

Resultados de la validación end-to-end (Compresión + Descompresión) ejecutados directamente sobre una placa **Raspberry Pi 4 (4GB)**, utilizando la imagen de prueba de *512×512×8 bandas* y `lambda=0.1`.

| Métrica | C (OpenMP - 4 hilos) | Python TensorFlow |
|---------|------------|-------------------|
| **Velocidad Compresión** | **193.31 s** | 547.79 s |
| **Velocidad Descompresión**| **159.70 s** | 578.64 s |
| **Tiempo Total Pipeline** | **353.01 s** | 1126.43 s |
| **Consumo Máximo de RAM** | **~135 MB** | ~816 MB |
| **PSNR vs Imagen Original**| **76.73 dB** | 76.73 dB |
| **Fidelidad C vs Python** | **105.41 dB (PSNR)** | - |

### Conclusiones de Rendimiento:
- **Speedup de Tiempo:** La implementación optimizada en C es **3.19× veces más rápida** que el framework base de TensorFlow corriendo en la placa ARM (pasando de casi 19 minutos a menos de 6 minutos en el proceso completo).
- **Ahorro de Memoria:** Reduce la huella en memoria dramáticamente (de 816 MB a tan solo 135 MB reales), habilitando ejecuciones concurrentes o en hardware satelital de recursos altamente limitados.
- **Calidad Conservada (Pixel-Perfect):** Ambos decompressors devuelven exactamente *105.41 dB* de similitud entre ellos, con diferencias triviales de redondeo originadas por NEON. Funcionalmente, la calidad es idéntica a la red pre-entrenada por el IEEC.

## Licencia

Este proyecto es parte de un Trabajo de Fin de Grado. Consultar con el autor antes de usar.

## Agradecimientos

- Institut d'Estudis Espacials de Catalunya (IEEC) - Modelo SORTENY original
- Universitat Autònoma de Barcelona - Supervisión académica
- En especial al Doctor Sebastià Mijares i Verdú por la posibilidad de realizar este proyecto

## Contacto

Cristhian Omar Añez Lopez - 1635157@uab.cat