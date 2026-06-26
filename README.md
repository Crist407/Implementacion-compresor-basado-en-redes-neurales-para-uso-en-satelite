# SORTENY fixed-quality C pipeline

Implementacion C y scripts de reproduccion para estudiar compresion a calidad
fija en datos multiespectrales Sentinel-2. El repositorio contiene el codec C,
pesos exportados, generadores de Q-map, una ruta medida `lambda/Q`, dataset RAW
y fuentes de la memoria final.

Trabajo de Fin de Grado, Universitat Autonoma de Barcelona, 2025/2026.

## Contenido principal

```text
src/c/                         Codec C y generadores de Q-map
src/python/analysis/           Scripts reproducibles y analisis seleccionados
src/python/reference/          Referencia Python/TensorFlow de SORTENY
src/python/utils/              Utilidades de quicklook/ROI
scripts/raspberry/             Benchmarks autocontenidos para Raspberry
scripts/docs/                  Regeneracion de figuras y compilacion del informe
config/                        Calibracion lambda005 y thresholds
weights/                       Pesos exportados del modelo
data/                          RAWs publicos y guia de visualizacion
docs/informe_final/            Memoria final, figuras y bibliografia
```

Los checkpoints completos, bundles Raspberry, entornos virtuales, binarios y
temporales de LaTeX no se versionan. La politica exacta esta en
`PUBLIC_ARTIFACTS.md`.

## Requisitos

- Linux
- GCC con C11
- `make`
- Python 3.9+
- NumPy para la ruta medida

Instalacion Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Compilacion C

```bash
make MODE=release OMP=1
make MODE=release OMP=1 test_ops
```

Binarios principales:

- `sorteny_compressor`
- `sorteny_decompressor`
- `sorteny_fq_qmap`
- `sorteny_semantic_qmap`
- `sorteny_global_qmap`
- `sorteny_quicklook`

## Quickstart reproducible

El siguiente comando ejecuta una prueba local autocontenida con la muestra RAW
publica y la calibracion `lambda005` incluida en `config/`:

```bash
python3 src/python/analysis/run_lambda005_measured_quality_route.py \
  --output-dir output/checkpoints/quickstart_measured_route \
  --modes focus_bgq128_measured \
  --threads 4
```

La ruta realiza:

```text
RAW -> compresion/descompresion base -> MSE0 -> lambda/Q -> Q-map
    -> compresion final -> descompresion final -> metricas
```

Salidas esperadas:

- `qmaps/*.raw`: Q-maps de 1,024 bytes.
- `bitstreams/*.bin`: contenedores C.
- `reconstructions/*.raw`: reconstrucciones de 4,194,304 bytes.
- `metrics.csv`, `metrics.json`, `run_meta.json`: metricas y trazabilidad.

Comprobacion rapida:

```bash
CHECKPOINT=output/checkpoints/quickstart_measured_route
find "$CHECKPOINT/qmaps" -name '*.raw' -printf '%p %s\n'
find "$CHECKPOINT/reconstructions" -name '*.raw' -printf '%p %s\n'
sed -n '1,5p' "$CHECKPOINT/metrics.csv"
```

## Dataset RAW

`data/Sentinel2A_crop_test/` contiene 120 crops RAW. Cada archivo tiene:

- `8 x 512 x 512` muestras;
- tipo `uint16`;
- layout BSQ;
- 4,194,304 bytes.

Orden de bandas: B02, B03, B04, B05, B06, B07, B08, B8A.

Los RAWs no se pueden abrir como imagen comun porque no incluyen cabecera y
contienen ocho bandas. Para visualizarlos se recomienda Fiji/ImageJ:

- `File -> Import -> Raw...`
- image type: `16-bit Unsigned`
- width: `512`
- height: `512`
- number of images: `8`
- offset: `0`
- gap: `0`
- byte order: little-endian

Para RGB aproximado: B04 como rojo, B03 como verde y B02 como azul. Hay mas
detalle en `data/README.md`.

Tambien se puede crear un quicklook PNG:

```bash
python3 src/python/utils/manual_roi_quicklook.py \
  --input data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw \
  --output-dir output/checkpoints/quicklook_example \
  --pattern empty \
  --skip-qmap
```

## Memoria final

Las fuentes LaTeX estan en `docs/informe_final/`. Para compilar:

```bash
scripts/docs/build_informe_final_compat.sh
```

El PDF generado queda en `docs/informe_final/main.pdf`. Los archivos auxiliares
de LaTeX no se versionan.

## Reproduccion avanzada

Las rutas CSMR y Raspberry estan documentadas en el Apendice A de la memoria.
Requieren dependencias o artefactos adicionales:

- CSMR: TensorFlow/TensorFlow Compression y checkpoints de seleccion.
- Raspberry: bundle autocontenido y placa Raspberry configurada.

Estas rutas reproducen resultados de la memoria, pero no forman parte del
quickstart minimo.

## Checks recomendados

```bash
python3 -m py_compile src/python/analysis/run_lambda005_measured_quality_route.py
make MODE=release OMP=1
make MODE=release OMP=1 test_ops
git diff --check
```

Antes de publicar, revisar que no se incluyan `output/`, `.venv/`, bundles,
binarios compilados ni temporales de LaTeX.
