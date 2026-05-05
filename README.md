# SORTENY C: Compresor y Descompresor de Imagenes Sentinel-2

Implementacion en C11 del pipeline SORTENY para comprimir y descomprimir imagenes satelitales Sentinel-2 RAW BSQ `uint16`. La base actual incluye encoder C, decoder C, pesos exportados y utilidades Python de validacion.

> Trabajo de Fin de Grado - Grado en Ingenieria Informatica
> Cristhian Omar Anez Lopez · Universitat Autonoma de Barcelona · 2025/2026

## Estado Actual

- Pipeline C completo: compresion, bitstream latente, descompresion y reconstruccion RAW.
- Plataforma oficial de benchmark: Raspberry Pi 3B+ con Raspberry Pi OS 64-bit.
- Configuracion canonica actual: imagen `8 x 512 x 512`, `lambda=0.1`, `max_lambda=0.125`.
- Pesos C versionados:
  - encoder: `weights/encoder`
  - decoder: `weights/decoder`
- La codificacion entropica del latente sigue pendiente; el bitstream actual guarda Q-map y latentes `int32` sin codificador entropico final.

Los resultados vigentes proceden de Progreso 1. Los informes `docs/informes/Entrega Final.tex` y `docs/presentacion/Presentacion.tex` son documentos historicos de fases anteriores y no son la fuente canonica de esta base.

## Estructura

```text
src/c/
  main.c              Encoder C
  decompress.c        Decoder C
  sorteny_layers.c    Conv2D, GDN, IGDN, Dense, ReLU y operadores auxiliares
  sorteny_model.c     Carga/liberacion de pesos exportados
  io_helpers.c        Lectura/escritura RAW BSQ y conversiones

src/python/
  core/               Extraccion de pesos y referencia TensorFlow historica
  analysis/           Validacion end-to-end y comparaciones
  benchmark/          Benchmarks locales ligeros; no sustituyen al harness de Raspberry
  reference/          Codigo SORTENY original de referencia

weights/
  encoder/            Pesos usados por sorteny_compressor
  decoder/            Pesos usados por sorteny_decompressor

scripts/
  rpi_sys_probe.sh    Perfil de sistema Raspberry
  *.sh                Scripts historicos marcados como legacy si usan rutas antiguas
```

## Requisitos

- Linux.
- GCC con C11, `make` y `libm`.
- OpenMP opcional para modo rapido.
- Python 3.9+ solo para validacion y analisis.

Dependencias Python:

```bash
pip install -r requirements.txt
```

## Compilacion

```bash
make clean
make MODE=release OMP=1
```

Variables utiles del `Makefile`:

| Variable | Uso |
|---|---|
| `MODE=release` | Build optimizado y conservador para paridad numerica |
| `MODE=release_fast` | Build mas agresivo; no es el perfil canonico de Progreso 1 |
| `OMP=1` | Compila con OpenMP |
| `RPI_ARCH=rpi3` | Fuerza flags para Raspberry Pi 3B+ |
| `BENCH_THREADS=4` | Numero de hilos para `make run_fast` |

Targets utiles:

```bash
make run          # solo encoder
make run_dec      # decoder sobre output/latent.bin
make run_pipeline # encoder + decoder
make run_parity   # pipeline con STRICT_PARITY=1
make run_fast     # pipeline con OMP_NUM_THREADS=4 por defecto
make test_ops     # tests unitarios de operadores decoder
```

## Uso C

Comprimir:

```bash
./sorteny_compressor \
  data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw \
  0.1 \
  output/latent.bin \
  weights/encoder \
  0.125
```

Descomprimir:

```bash
./sorteny_decompressor \
  output/latent.bin \
  output/reconstructed.raw \
  weights/decoder \
  0.125
```

Los directorios de pesos son opcionales porque los binarios usan por defecto `weights/encoder` y `weights/decoder`.

## Modos de Ejecucion

| Modo | Comando | Uso |
|---|---|---|
| Paridad | `STRICT_PARITY=1 ./sorteny_compressor ...` | Determinismo y redondeo half-to-even con 1 hilo |
| Fast Pi 3B+ | `OMP_NUM_THREADS=4 ./sorteny_compressor ...` | Perfil recomendado de rendimiento en Raspberry Pi 3B+ |

`lambda` se recorta a `[0, max_lambda]`, se cuantiza a 8 bits en el Q-map y se reconstruye durante la descompresion. Para los benchmarks oficiales se usa `lambda=0.1` y `max_lambda=0.125`.

Las variables `DUMP_*` del encoder (`DUMP_Y_PRE`, `DUMP_M`, `DUMP_Y_FLOAT`, `DUMP_STAGES`, `DUMP_SPECTRAL`) son legacy. La ruta optimizada actual no promete esos volcados como interfaz estable y el binario los ignora con aviso.

## Arquitectura Implementada

Encoder:

```text
RAW BSQ uint16 (8 x 512 x 512)
  -> transformada espectral 8x8
  -> normalizacion por banda
  -> analysis transform espacial:
       conv0 + GDN
       conv1 + GDN
       conv2 + GDN
       conv3 sin GDN final
  -> modulating transform segun lambda cuantizada
  -> cuantizacion half-even opcional
  -> bitstream: cabecera + Q-map + latentes int32
```

Decoder:

```text
bitstream
  -> lectura de cabecera, Q-map y latentes
  -> reconstruccion de lambda cuantizada
  -> desmodulacion
  -> synthesis transform espacial con IGDN
  -> transformada espectral inversa
  -> clamp + cast a uint16
  -> RAW BSQ reconstruido
```

La GDN/IGDN C sigue la forma implementada por los pesos exportados usados en esta base: `beta + sum(gamma * abs(x))`, con division en GDN y multiplicacion en IGDN.

## Benchmarks Oficiales en Raspberry Pi 3B+

Fuente canonica: Progreso 1, ejecutado en Raspberry Pi 3B+ sobre `T31TCG_20230907T104629_5.8_512_512_2_1_0.raw`, `lambda=0.1`, `max_lambda=0.125`, 3 repeticiones + 1 warmup.

| Metrica | Baseline C, 1 hilo | Fast C, OpenMP 4 hilos |
|---|---:|---:|
| Compresion media | 542.9492 s | 193.3121 s |
| Descompresion media | 401.4604 s | 159.7029 s |
| Total medio | 944.4096 s | 353.0149 s |
| PSNR vs original | 76.7255 dB | 76.7255 dB |
| MAE vs original | 6.8350 | 6.8350 |
| CPU media compresion | 100.5061 % | 293.1598 % |
| CPU media descompresion | 100.4733 % | 260.0170 % |
| RAM pico descompresion | 134.6641 MB | 135.6953 MB |
| Hilos maximos observados | 1 | 4 |

Speedups del modo fast:

| Comparativa | Valor |
|---|---:|
| Compresion baseline/fast | 2.8087x |
| Descompresion baseline/fast | 2.5138x |
| Total baseline/fast | 2.6753x |
| Reduccion de tiempo total | 62.62 % |

Comparativa contra Python de referencia sin codificador entropico:

| Metrica | C fast | Python |
|---|---:|---:|
| Compresion | 193.3121 s | 547.79 s |
| Descompresion | 159.7029 s | 578.64 s |
| Total | 353.0149 s | 1126.43 s |

Speedup C fast vs Python: `2.8337x` en compresion, `3.6232x` en descompresion y `3.1909x` en total.

## Paridad C vs Python

| Metrica | Valor |
|---|---:|
| Tamano bitstream C | 12,583,946 bytes |
| Tamano bitstream Python | 12,583,946 bytes |
| Bytes diferentes en bitstream | 412 |
| Primer byte distinto | offset 17,278 (`C=73`, `PY=74`) |
| PSNR recon C vs recon Python | 105.4077 dB |
| MSE recon C vs recon Python | 0.1236453 |
| MAE recon C vs recon Python | 0.0713868 |
| Max abs diff | 8 |
| Pixeles identicos | 94.5585 % |

Interpretacion: las reconstrucciones son muy proximas y conservan la calidad frente al original, pero no son pixel-perfect. Las diferencias son coherentes con redondeo y orden de operaciones entre implementaciones.

## Validacion Local

Smoke test recomendado:

```bash
make clean && make MODE=release OMP=1
make run_pipeline
python3 src/python/analysis/validate_e2e.py \
  data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw \
  output/reconstructed.raw \
  --lmbda 0.1 \
  --max-lambda 0.125
make test_ops
```

Comprobaciones estaticas:

```bash
gcc -Wall -Wextra -std=c11 -fopenmp -fsyntax-only src/c/*.c
python3 -m py_compile $(find src/python scripts -name '*.py')
bash -n scripts/*.sh scripts/raspberry/*.sh 2>/dev/null || true
```

## Scripts y Artefactos de Raspberry

Los informes de Progreso 1 mencionan `scripts/raspberry/run_benchmark.sh` y `scripts/raspberry/benchmark_pipeline.py`, pero esas fuentes no estan versionadas actualmente en este repo. Si se recuperan por SSH desde la Raspberry, deben entrar en `scripts/raspberry/` junto con los artefactos de evidencia o sus resumenes.

Los scripts antiguos de `scripts/` que usan `sorteny_compress`, `pesos_ieec050_*`, `models/ieec050` o arboles remotos historicos estan marcados como legacy y no se ejecutan salvo con `SORTENY_RUN_LEGACY=1`.

## Licencia

Este proyecto forma parte de un Trabajo de Fin de Grado. Consultar con el autor antes de reutilizarlo.

## Agradecimientos

- Institut d'Estudis Espacials de Catalunya (IEEC), modelo SORTENY original.
- Universitat Autonoma de Barcelona, supervision academica.
- Dr. Sebastia Mijares i Verdu, por la direccion y apoyo del proyecto.

## Contacto

Cristhian Omar Anez Lopez - 1635157@uab.cat
