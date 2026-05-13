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
  fixed_quality_qmap.c Generador C de Q-map local desde calibracion TSV
  semantic_qmap.c     Generador C de Q-map semantico Sentinel-2
  sorteny_layers.c    Conv2D, GDN, IGDN, Dense, ReLU y operadores auxiliares
  sorteny_model.c     Carga/liberacion de pesos exportados
  io_helpers.c        Lectura/escritura RAW BSQ y conversiones

src/python/
  core/               Extraccion de pesos y referencia TensorFlow historica
  analysis/           Calibracion auxiliar, validacion end-to-end y comparaciones
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

Los binarios `sorteny_fq_qmap` y `sorteny_semantic_qmap` se compilan con `make MODE=release OMP=1`. Son la ruta C para convertir una calibracion de calidad fija local o un preset semantico Sentinel-2 en un Q-map. Los scripts Python de Fase 2 son auxiliares de calibracion y evidencia; no forman parte de la ruta final prevista para Raspberry.

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

El compresor acepta opcionalmente un Q-map externo `uint8` como ultimo argumento. Para la imagen canonica de `512 x 512`, el mapa debe tener `32 x 32 = 1024` bytes. Si no se proporciona, se genera el Q-map constante a partir de `lambda`.

```bash
python3 src/python/utils/generate_qmap.py \
  output/qmap_32x32_u8.bin \
  --pattern horizontal-split \
  --q-low 180 \
  --q-high 204 \
  --split 16

./sorteny_compressor \
  data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw \
  0.1 \
  output/latent.bin \
  weights/encoder \
  0.125 \
  output/qmap_32x32_u8.bin
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

Analisis local por bloques `16 x 16`:

```bash
python3 src/python/analysis/analyze_block_quality.py \
  data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw \
  output/reconstructed.raw \
  --bitstream output/latent.bin \
  --output-json output/block_quality.json \
  --output-csv output/block_quality.csv
```

Barrido empirico de Q para calibracion local:

```bash
python3 src/python/analysis/sweep_q_quality.py \
  --output-dir output/checkpoints/q_sweep_calibration \
  --q-values 160 176 192 204 216 232 \
  --threads 4
```

Para estudiar viabilidad local, usar un rango mas amplio:

```bash
python3 src/python/analysis/sweep_q_quality.py \
  --output-dir output/checkpoints/q_sweep_calibration_wide \
  --q-values 128 144 160 176 192 204 216 232 240 248 255 \
  --threads 4
```

Seleccion global de Q para un objetivo de calidad:

```bash
python3 src/python/analysis/select_q_for_target.py \
  --sweep output/checkpoints/q_sweep_calibration/sweep_results.json \
  --target-psnr 76.8 \
  --output-qmap output/qmap_target_psnr.bin \
  --summary-json output/qmap_target_psnr.json
```

Ese selector global en Python es experimental. Para calidad fija local, la decision de Q-map debe hacerse con la utilidad C:

```bash
python3 src/python/analysis/build_fq_calibration.py \
  --sweep output/checkpoints/q_sweep_calibration_wide/sweep_results.json \
  --output-tsv output/fq_calibration.tsv \
  --summary-json output/fq_calibration_summary.json

./sorteny_fq_qmap \
  --calibration output/fq_calibration.tsv \
  --target-psnr 76.8 \
  --output-qmap output/qmap_fq_local.bin \
  --summary-tsv output/qmap_fq_local_summary.tsv

./sorteny_compressor \
  data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw \
  0.1 \
  output/latent_fq_local.bin \
  weights/encoder \
  0.125 \
  output/qmap_fq_local.bin
```

La calibracion inicial usa el modelo empirico por bloque `MSE ~= c0 + c1 / M(lambda)^2`. Si el objetivo local queda fuera del rango calibrado, `sorteny_fq_qmap` recorta Q a los limites disponibles y lo deja registrado en el resumen TSV.

Con la calibracion ampliada `Q=128..255`, la regresion `--target-from-q 204` reproduce el baseline byte a byte. En cambio, un objetivo local uniforme `--target-psnr 76.8` no es una politica suficiente: muchos bloques faciles se recortan a Q minimo y bloques dificiles a Q maximo. La herramienta C reporta esa viabilidad para decidir el siguiente ajuste.

Politica adaptativa C por dificultad de bloque:

```bash
./sorteny_fq_qmap \
  --calibration output/fq_calibration.tsv \
  --adaptive-difficulty \
  --q-mean 204 \
  --adaptive-strength 8 \
  --output-qmap output/qmap_adaptive_difficulty.bin \
  --summary-tsv output/qmap_adaptive_difficulty_summary.tsv
```

Esta politica usa `mse_at_baseline` por bloque, sube Q en bloques dificiles y baja Q en bloques faciles manteniendo aproximadamente el presupuesto medio. En la primera prueba local, `--adaptive-strength 8` mantiene `q_mean=203.98`, mejora ligeramente el PSNR global frente a Q=204 constante (`76.7471 dB` vs `76.7255 dB`) y reduce el MSE medio de los 10 peores bloques (`802.69` vs `808.24`).

Q-map semantico Sentinel-2 en C:

```bash
./sorteny_semantic_qmap \
  --input data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw \
  --calibration output/fq_calibration.tsv \
  --preset vegetation \
  --output-qmap output/qmap_semantic_vegetation.bin \
  --summary-tsv output/qmap_semantic_vegetation_summary.tsv \
  --bands 8 \
  --height 512 \
  --width 512 \
  --band-layout sentinel2-8 \
  --q-mean 204 \
  --adaptive-strength 8 \
  --semantic-boost 8
```

Catalogo semantico actual: `vegetation` (NDVI), `water` (NDMI), `burned` (NBR), `snow` (NDSI), `water_body` (NDWI), `chlorophyll` (NDCI), `vegetation_green` (GNDVI), `clouds` (CBY), `barren_soil` (BSI), `burned_area` (BAIS2), `uniform` y `manual`. La base adaptativa recomendada es `--q-mean 204 --adaptive-strength 8`; el preset semantico suma un boost conservador de Q en los bloques donde el indice cumple el umbral. Para la imagen canonica de 8 bandas se asume el orden `B02,B03,B04,B05,B06,B07,B08,B8A`. Los presets que requieren `B11` o `B12` registran `missing_bands` hasta trabajar con entradas Sentinel-2 completas de 13 bandas. `clouds` funciona con 8 bandas en modo CBY basico (`B03`,`B04`); el filtro con `B11` queda pendiente de dataset completo.

Checkpoint local `output/checkpoints/20260508_semantic_qmap_c`: `vegetation` genera un Q-map de 1024 bytes, aplica boost en 108/1024 bloques y obtiene MSE global `90.6000`, PSNR `76.7582 dB`. Es una primera evidencia semantica, no un benchmark oficial Raspberry.

Validacion local NDVI `output/checkpoints/20260509_semantic_validation_vegetation`: barrido `threshold={0.35,0.40,0.45}` y `semantic_boost={4,8,12,16}`. El candidato recomendado para `vegetation` es `--threshold 0.40 --semantic-boost 16`: actua sobre 108 bloques, mantiene `q_mean=205.6709`, mejora la region semantica frente a `adaptive_difficulty s=8` en `+0.3458 dB` y deja PSNR global `76.7681 dB`.

Politica de foco con degradacion de fondo:

```bash
./sorteny_semantic_qmap \
  --input data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw \
  --calibration output/fq_calibration.tsv \
  --preset vegetation \
  --semantic-policy focus \
  --foreground-boost 16 \
  --background-penalty 24 \
  --threshold 0.40 \
  --output-qmap output/qmap_focus_vegetation.bin \
  --summary-tsv output/qmap_focus_vegetation_summary.tsv \
  --bands 8 \
  --height 512 \
  --width 512 \
  --band-layout sentinel2-8 \
  --q-mean 204 \
  --adaptive-strength 8
```

Checkpoint local `output/checkpoints/20260510_semantic_focus_vegetation`: el candidato recomendado `foreground_boost=16`, `background_penalty=24` mantiene/mejora la ROI de vegetacion (`+0.3112 dB` frente a `adaptive_difficulty s=8`), degrada el fondo (`-0.3015 dB`), reduce `q_mean` de `205.6709` a `184.2021` frente al modo semantico boost-only anterior y baja la entropia empirica de latentes. La reduccion real de ancho de banda queda pendiente del codificador entropico.

Modo agresivo de fondo fijo:

```bash
./sorteny_semantic_qmap \
  --input data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw \
  --calibration output/fq_calibration.tsv \
  --preset vegetation \
  --semantic-policy focus \
  --foreground-boost 16 \
  --background-q 128 \
  --threshold 0.40 \
  --output-qmap output/qmap_focus_vegetation_bg128.bin \
  --summary-tsv output/qmap_focus_vegetation_bg128_summary.tsv \
  --bands 8 \
  --height 512 \
  --width 512 \
  --band-layout sentinel2-8 \
  --q-mean 204 \
  --adaptive-strength 8
```

Checkpoint local `output/checkpoints/20260511_semantic_background_q_vegetation`: `--background-q 128` fuerza el fondo a Q=128, mantiene la ROI por encima de `adaptive_difficulty s=8` (`+0.1749 dB`), degrada el fondo (`-1.3350 dB`), baja `q_mean` a `137.3955`, sube ceros latentes a `42.20%` y reduce entropia empirica a `4.1913 bits/simbolo`. El archivo binario todavia no adelgaza porque los latentes siguen escritos como `int32`; esta mejora es un proxy para el codificador entropico futuro.

Paquete de evidencia visual y estadistica del foco semantico:

```bash
python3 src/python/analysis/build_focus_evidence_report.py \
  --output-dir output/checkpoints/20260512_focus_evidence_report
```

Este script es solo auxiliar de analisis; la decision final del Q-map sigue estando en C (`sorteny_semantic_qmap`). El checkpoint `output/checkpoints/20260512_focus_evidence_report` cruza Q-map, NDVI, calidad local y latentes para comparar Q=204, adaptativo `s=8`, semantico boost-only y focus `background-q=128`. Genera `focus_evidence_summary.{json,csv,md}`, `block_evidence.csv`, `latent_histograms.csv` y mapas `PGM/PPM` a 32x32 y 512x512. La evidencia clave confirma `q_mean=137.3955`, ROI `+0.1749 dB`, fondo `-1.3350 dB`, ceros latentes `42.20%` y entropia `4.1913 bits/simbolo`.

Mapeo flexible de bandas y ROI manual:

```bash
# TSV de ejemplo para un RAW de 8 bandas en orden B02,B03,B04,B05,B06,B07,B08,B8A
cat > output/sentinel2_8_band_map.tsv <<'EOF'
band_name	index
B02	0
B03	1
B04	2
B05	3
B06	4
B07	5
B08	6
B8A	7
EOF

./sorteny_semantic_qmap \
  --input data/Sentinel2A_crop_test/T31TCH_20230907T104629_35.8_512_512_2_1_0.raw \
  --calibration output/fq_calibration.tsv \
  --preset vegetation \
  --semantic-policy focus \
  --foreground-boost 16 \
  --background-q 128 \
  --threshold 0.40 \
  --band-map output/sentinel2_8_band_map.tsv \
  --output-qmap output/qmap_vegetation_band_map.bin \
  --summary-tsv output/qmap_vegetation_band_map.tsv
```

`--band-map` permite preparar el calculo semantico para imagenes con distinto numero u orden de bandas. Si un preset necesita una banda que no existe (`B11`/`B12` en un dataset de 8 bandas), `sorteny_semantic_qmap` registra `missing_bands` y conserva la base adaptativa sin inventar una mascara semantica.

Para foco manual, la ruta final tambien es C. El ROI puede venir de un mapa `uint8` de 32x32 o de un TSV de bloques:

```bash
python3 src/python/utils/generate_roi_map.py \
  output/roi_manual_center.bin \
  --pattern center \
  --output-tsv output/roi_manual_center.tsv \
  --summary-json output/roi_manual_center_summary.json

./sorteny_semantic_qmap \
  --calibration output/fq_calibration.tsv \
  --preset manual \
  --semantic-policy focus \
  --foreground-boost 16 \
  --background-q 128 \
  --roi-map output/roi_manual_center.bin \
  --output-qmap output/qmap_manual_focus.bin \
  --summary-tsv output/qmap_manual_focus.tsv
```

Checkpoint local `output/checkpoints/20260514_manual_roi_focus`: ROI central de 256 bloques, Q-map de 1024 bytes, equivalencia byte a byte entre `--roi-map` y `--roi-tsv`, `q_mean=151.1191` y pipeline C completo compatible.

Validacion batch del dataset `data/Sentinel2A_crop_test`:

```bash
python3 src/python/analysis/run_semantic_dataset_smoke.py \
  --output-dir output/checkpoints/20260513_sentinel2a_8band_dataset_validation
```

Checkpoint local `output/checkpoints/20260513_sentinel2a_8band_dataset_validation`: valida 120/120 RAW como `8 x 512 x 512` BSQ `uint16`, confirma que `--band-map` equivalente produce el mismo Q-map que `sentinel2-8`, y comprueba que `water`, `burned` y `snow` reportan 1024 bloques `missing_bands` en el dataset de 8 bandas. El smoke completo se ejecuta sobre `T31TCH_20230907T104629_35.8_512_512_2_1_0.raw`, con PSNR `78.1528 dB`.

Auditoria reproducible del catalogo completo de presets:

```bash
python3 src/python/analysis/audit_semantic_presets.py \
  --output-dir output/checkpoints/20260515_semantic_preset_catalog_audit
```

Checkpoint local `output/checkpoints/20260515_semantic_preset_catalog_audit`: ejecuta los 12 presets sobre la imagen canonica y sobre los 120 RAW del dataset de 8 bandas, genera Q-maps de 1024 bytes y deja `preset_catalog.{csv,json}`, `dataset_preset_summary.{csv,json}` y `preset_catalog_audit.md`. La regresion `vegetation` es byte-identical frente al checkpoint anterior, `--band-map` vuelve a coincidir byte a byte con `sentinel2-8`, y `water`, `burned`, `snow`, `barren_soil` y `burned_area` reportan 1024 bloques `missing_bands` cuando faltan `B11/B12`. En la imagen canonica, `clouds` detecta 144/1024 bloques en modo CBY basico sin `B11`; el smoke completo con ese Q-map obtiene PSNR `76.7709 dB`, ceros latentes `39.45%` y entropia `4.5974 bits/simbolo`. Esta auditoria acepta el catalogo como reproducible, no como validacion cientifica definitiva de todos los presets.

Verificacion de formula del paper y evidencia de latentes:

```bash
python3 src/python/analysis/verify_fq_formula_and_latents.py \
  --output-dir output/checkpoints/20260516_formula_and_latent_coding_evidence
```

Checkpoint local `output/checkpoints/20260516_formula_and_latent_coding_evidence`: documenta la equivalencia usada con el paper (`c0 ~= MSE0`, `c1 ~= alpha*MSE0/4`, `M(lambda)=mod_a*lambda+mod_b`), recompone en Python los Q-maps generados por C y compara estadistica de latentes finales. Las recomputaciones de `target_from_q_204`, `target_psnr_76_8` y `adaptive_difficulty_s8` coinciden byte a byte con `sorteny_fq_qmap`. El barrido Q mantiene monotonia global, con `R2=0.998984`, MAE de MSE `2.1486` y RMSE de MSE `3.7390`. Frente a `adaptive_s8`, `focus_bgq128` aumenta los ceros latentes en `+2.6904` puntos porcentuales, reduce la entropia empirica en `-0.3989` bits/simbolo, reduce los bits ideales estimados en `-0.5983` bits por muestra de entrada y mejora el proxy `zlib` en `-0.8085` bps. En este contexto, "layers" significa canales latentes finales, no capas convolucionales internas; los dumps internos son legacy y no forman parte del pipeline optimizado. Esto es evidencia estadistica, no reduccion real de ancho de banda hasta integrar el codificador entropico.

Auditoria MCOS 2024 y correccion del experimento de histogramas:

```bash
python3 scripts/experiments/exp3_latent_histograms.py \
  --output-dir output/experiments/exp3_latent_histograms_fixed
```

`Prueba Extra NO GIT/mcos2024-production` se mantiene como referencia externa del paper, no como dependencia del pipeline C. La auditoria queda documentada en `docs/informes/MCOS2024_audit.md`: MCOS confirma la forma `MSE ~= MSE0 + alpha*MSE0/(4*M(lambda)^2)` y la cuantizacion de `lambda/Q`, pero usa TensorFlow/TensorFlow Compression, hyperprior y codificacion entropica. Por tanto, sirve para contraste teorico y diseno futuro, no para la ruta Raspberry actual.

El experimento 3 antiguo leia el bitstream desde un offset incorrecto. El formato real es cabecera de 10 bytes, Q-map de 1024 bytes y latentes `int32`. La salida corregida `output/experiments/exp3_latent_histograms_fixed` conserva la conclusion: frente a `constant_q204`, `focus_bgq128` baja `std` de `52.3110` a `36.3995`, sube ceros de `39.5622%` a `42.1979%`, baja entropia de `4.5838` a `4.1913` bits/simbolo y mejora `zlib` de `8.9156` a `8.1391` bps. El experimento gzip (`scripts/experiments/exp1_gzip_proxy.sh`) comprime el bitstream SORTENY completo, no latentes aislados; sigue siendo un proxy valido porque los bitstreams comparados tienen el mismo tamano sin comprimir.

Demo reproducible de funcionamiento correcto:

```bash
python3 src/python/analysis/build_correctness_demo.py \
  --output-dir output/checkpoints/20260517_correct_functioning_demo
```

Checkpoint local `output/checkpoints/20260517_correct_functioning_demo`: ejecuta de principio a fin la ruta C para tres politicas (`q204`, `adaptive_s8` y `vegetation_focus_bgq128`), generando Q-map, bitstream, reconstruccion, metricas locales y mapas visuales. `q204` reproduce la referencia con PSNR `76.7255 dB`. Frente a `adaptive_s8`, `vegetation_focus_bgq128` conserva/mejora la ROI (`+0.1748 dB`), degrada el fondo (`-1.3350 dB`), baja `q_mean` de `203.9834` a `137.3955`, aumenta ceros latentes en `+2.6904` puntos porcentuales y reduce entropia en `-0.3989 bits/simbolo`. El checkpoint deja `correctness_demo_summary.{json,csv}`, `correctness_demo_report.md`, `block_correctness_evidence.csv`, `latent_policy_summary.csv`, logs por comando y mapas `PGM/PPM` a 32x32 y 512x512. La reduccion real de ancho de banda sigue pendiente del codificador entropico porque el bitstream actual escribe latentes `int32`.

Validacion matematica PSNR/MSE sobre el dataset Sentinel-2A:

```bash
python3 src/python/analysis/validate_target_psnr_dataset.py \
  --output-dir output/checkpoints/20260518_target_psnr_dataset_validation
```

Este script ejecuta la cadena `target PSNR -> target MSE -> sorteny_fq_qmap -> compresion C -> descompresion C -> PSNR real` sobre `data/Sentinel2A_crop_test`. Por defecto valida 120 crops y 8 objetivos (`74.5,75.0,75.5,76.0,76.5,76.8,77.0,77.5` dB) con la calibracion canonica `output/checkpoints/20260507_c_fixed_quality_qmap_wide/fq_calibration.tsv`, y recalibra 5 crops representativos con Q `128..255` para separar error de formula y error de generalizacion. La ejecucion completa es costosa y es reanudable; para smoke:

```bash
python3 src/python/analysis/validate_target_psnr_dataset.py \
  --output-dir output/checkpoints/20260518_target_psnr_dataset_validation_smoke \
  --max-crops 1 \
  --targets 74.5 76.8 \
  --recalibrate-count 1 \
  --keep-artifacts sample
```

Smoke local ejecutado en `output/checkpoints/20260518_target_psnr_dataset_validation_smoke`: valida un crop, dos objetivos y una recalibracion por crop. Con calibracion fija, el error absoluto medio fue `1.0747 dB`; recalibrando el crop bajó a `0.6026 dB`, sin violaciones de monotonia. Para el objetivo `76.8 dB`, la calibracion fija quedó en `75.5479 dB` con solo `14.65%` de bloques alcanzables, mientras que la recalibracion alcanzó `76.4506 dB` con `63.48%` de bloques alcanzables. Esto no sustituye al barrido completo: confirma que el procedimiento funciona y que la saturacion Q=128/Q=255 debe interpretarse explicitamente.

Demostracion de los tres significados de un objetivo PSNR/MSE:

```bash
python3 src/python/analysis/demonstrate_psnr_objectives.py \
  --output-dir output/checkpoints/20260519_psnr_objective_modes_demo \
  --max-crops 3 \
  --targets 74.5 76.8 77.5
```

Checkpoint local `output/checkpoints/20260519_psnr_objective_modes_demo`: separa tres casos que no deben confundirse. En objetivo local por bloque, `--target-psnr 76.8` solo deja `14.65%` de bloques alcanzables y el PSNR global medio queda en `75.9903 dB`; esto demuestra saturacion local, no fallo de la formula. En objetivo global de imagen, la busqueda por Q constante alcanza `76.8 dB` con PSNR medio `76.8220 dB` y error absoluto medio `0.0220 dB`. En objetivo semantico ROI/fondo, `vegetation_focus_bgq128` baja el PSNR global medio a `76.2269 dB`, pero mantiene ROI por encima de `adaptive_s8` (`77.1768 dB` vs `77.0562 dB`), degrada fondo (`75.4068 dB` vs `76.9495 dB`), baja Q medio (`158.8073` vs `203.9834`), sube ceros latentes (`39.93%` vs `38.55%`) y reduce entropia (`4.6870` vs `4.9546`). El checkpoint genera `local_block_target_results.csv`, `global_image_target_results.csv`, `semantic_roi_target_results.csv`, `block_level_errors.csv`, `latent_proxy_summary.csv`, resumen Markdown/CSV/JSON y mapas `PGM/PPM`.

Validacion robusta de presets semanticos sobre dataset:

```bash
python3 src/python/analysis/evaluate_semantic_presets_dataset.py \
  --mode smoke \
  --smoke-crops 12 \
  --output-dir output/checkpoints/20260520_semantic_presets_dataset_smoke
```

Checkpoint local `output/checkpoints/20260520_semantic_presets_dataset_smoke`: ejecuta 12 crops estratificados, los controles `q204` y `adaptive_s8`, y los presets compatibles con `sentinel2-8` (`vegetation`, `vegetation_green`, `chlorophyll`, `water_body`, `clouds`) con dos politicas focus: conservadora `background_penalty=24` y agresiva `background_q=128`. Genera `semantic_preset_dataset_results.{csv,json}`, `semantic_preset_group_summary.csv`, `semantic_preset_recommendations.md`, `latent_proxy_by_preset.csv`, `missing_bands_summary.csv`, logs y mapas representativos. En el grupo interpretable `mid_roi`, `clouds + focus_bgq128` tiene 5/5 exitos, mantiene ROI (`+0.0960 dB`), degrada fondo (`-1.5876 dB`), baja Q medio (`-44.4992`) y reduce entropia (`-0.2639 bits/simbolo`) frente a `adaptive_s8`. `vegetation + focus_bgq128` tambien cumple 3/3, con ROI `+0.1654 dB`, fondo `-1.4225 dB`, Q medio `-42.2708` y entropia `-0.2381`. `water`, `burned`, `snow`, `barren_soil` y `burned_area` reportan 1024 bloques `missing_bands` en los 12 crops porque faltan `B11/B12`.

Antes de lanzar el modo completo, ejecutar el preflight:

```bash
python3 src/python/analysis/evaluate_semantic_presets_dataset.py \
  --mode preflight \
  --output-dir output/checkpoints/20260521_semantic_presets_dataset_preflight
```

Preflight local ejecutado en `output/checkpoints/20260521_semantic_presets_dataset_preflight`: valida 120 RAW, binarios C, pesos, calibracion, escritura en el directorio de salida, Q-maps de control de 1024 bytes y espacio libre. El full run esperado produce 2400 filas de resultados y 1440 pipelines C.

La ejecucion larga es reanudable con `--resume`. El checkpoint guarda `run_manifest.json`, `progress.csv` y `cases/<crop>/<preset>/<policy>/case_result.json`. Prueba local de resume ejecutada en `output/checkpoints/20260521_resume_smoke_test`: 2 crops, 40 filas, 34 `case_result.json`; la segunda ejecucion reutiliza los casos completados y no repite compresion/descompresion.

La validacion puede incluir tambien un control de ROI manual reproducible. Esta ruta sigue consumiendo el Q-map desde C con `sorteny_semantic_qmap --preset manual`; Python solo genera la mascara `uint8` y orquesta la prueba:

```bash
python3 src/python/analysis/evaluate_semantic_presets_dataset.py \
  --mode preflight \
  --include-manual-roi \
  --output-dir output/checkpoints/20260522_manual_roi_preflight
```

Preflight manual local ejecutado en `output/checkpoints/20260522_manual_roi_preflight`: genera `manual_center`, una ROI central de 256 bloques (`25.0%`), y Q-maps de 1024 bytes para `focus_bgpen24` y `focus_bgq128`. Con esta opcion, el full run esperado pasa a 2880 filas de resultados y 1680 pipelines C.

Smoke manual local ejecutado en `output/checkpoints/20260522_manual_roi_resume_smoke_test`: 2 crops, 48 filas, 38 `case_result.json`; la segunda ejecucion con `--resume` salta controles, presets semanticos y `manual_center`. En el primer crop, `manual_center + focus_bgq128` mantiene la ROI frente a `adaptive_s8` (`+0.2274 dB`), degrada el fondo (`-1.4975 dB`) y deja `q_mean=151.1191`.

El modo completo de 120 crops queda preparado, pero no debe lanzarse hasta decidirlo explicitamente. Sin ROI manual:

```bash
python3 src/python/analysis/evaluate_semantic_presets_dataset.py \
  --mode full \
  --resume \
  --keep-heavy representative \
  --output-dir output/checkpoints/20260521_semantic_presets_dataset_full
```

Con ROI manual:

```bash
python3 src/python/analysis/evaluate_semantic_presets_dataset.py \
  --mode full \
  --resume \
  --include-manual-roi \
  --keep-heavy representative \
  --output-dir output/checkpoints/20260522_semantic_presets_dataset_full_manual
```

El modo completo es costoso: implica 120 crops y 1440 pipelines C sin ROI manual, o 1680 pipelines C con `--include-manual-roi`.

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
