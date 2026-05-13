# Auditoria MCOS 2024 como referencia externa

Fecha: 10 de mayo de 2026

Proyecto revisado: `Prueba Extra NO GIT/mcos2024-production`

## Conclusion operativa

`mcos2024-production` sirve como referencia teorica y de contraste para la Fase 2 de SORTENY, pero no debe integrarse como dependencia de ejecucion. La ruta final del proyecto sigue siendo C: `sorteny_fq_qmap`, `sorteny_semantic_qmap`, `sorteny_compressor` y `sorteny_decompressor`.

## Puntos utiles para SORTENY C

- Implementa la logica del paper "Fixed-Quality Compression of Remote Sensing Images With Neural Networks".
- Confirma la forma del modelo empirico:
  - `M(lambda) = a * lambda + b`
  - `MSE ~= MSE0 + alpha * MSE0 / (4 * M(lambda)^2)`
- Su expresion equivale al modelo usado en `sorteny_fq_qmap`:
  - `c0 ~= MSE0`
  - `c1 ~= alpha * MSE0 / 4`
  - `M(lambda) = mod_a * lambda + mod_b`
- Confirma la cuantizacion/decuantizacion de la informacion lateral:
  - `Q = round(255 * (lambda - lambda_min) / (lambda_max - lambda_min))`
  - `lambda_q = (Q / 255) * (lambda_max - lambda_min) + lambda_min`
- Recalca una decision metodologica importante: si el objetivo MSE queda fuera del rango alcanzable, el sistema satura en el minimo o maximo permitido. Esto coincide con nuestras validaciones de PSNR/MSE y con los contadores de saturacion Q.

## Diferencias que impiden usarlo directamente

- MCOS usa TensorFlow y TensorFlow Compression; SORTENY C debe ejecutarse en Raspberry Pi 3B+ sin depender de esos paquetes.
- MCOS conserva hyperprior y modelos de entropia (`side_entropy_model`, `entropy_model`); el pipeline C actual todavia escribe cabecera, Q-map y latentes `int32` sin codificador entropico.
- Los modelos/pesos de MCOS no estan versionados localmente dentro de `Prueba Extra NO GIT/mcos2024-production`.
- MCOS es una referencia de investigacion general; nuestro pipeline actual esta fijado al modelo SORTENY C de 8 bandas y a los pesos `weights/encoder` / `weights/decoder`.

## Revision de los tres experimentos del equipo

### Experimento 1: gzip

El experimento es util, pero debe describirse con precision: `exp1_gzip_proxy.sh` comprime con `gzip -9` el bitstream SORTENY completo, no solo una region de latentes aislada. Como los bitstreams comparados tienen el mismo tamano sin comprimir, el resultado sigue siendo un proxy valido de compresibilidad.

Interpretacion aceptada: `focus bgQ128` y penalizaciones fuertes de fondo hacen que el bitstream sea mas facil de comprimir por gzip. Esto no equivale todavia a ancho de banda real del sistema final, porque falta el codificador entropico propio.

### Experimento 2: Python frente a C

El log `validation_q204.log` valida correctamente el port C frente a Python para el caso `Q=204` / `lambda=0.1`:

- C vs original: PSNR aproximado `76.73 dB`.
- Python vs original: PSNR aproximado `76.73 dB`.
- C vs Python: PSNR aproximado `105.31 dB`.
- Pixeles identicos C/Python: `94.48%`.

Esta es una evidencia fuerte de fidelidad funcional del port C. El log de `FQ adaptive` disponible solo mide C frente al original; no debe citarse como validacion Python frente a C adaptativa salvo que se genere un log equivalente.

### Experimento 3: histogramas de latentes

La conclusion cualitativa es correcta, pero el parser original no lo era. El script antiguo leia el fichero entero como `int32` y saltaba `258` enteros, asumiendo que cabecera y Q-map estaban alineados a 32 bits. El formato real es:

- `10` bytes de cabecera (`5 x uint16`).
- `1024` bytes de Q-map para imagenes `512 x 512`.
- Latentes `int32` a continuacion.

Por tanto, los valores numericos antiguos de desviacion estandar y rangos no deben usarse en informes. Se ha corregido `scripts/experiments/exp3_latent_histograms.py` para leer el bitstream real y guardar los resultados en `output/experiments/exp3_latent_histograms_fixed`.

## Uso recomendado en adelante

- Usar MCOS como referencia para revisar formulas, saturaciones y diseno del futuro codificador entropico.
- No importar codigo MCOS en el pipeline C actual.
- No bloquear la prueba de 120 crops por MCOS: la prueba debe seguir ejecutandose con `sorteny_semantic_qmap` y los binarios C.
- Usar los valores corregidos del experimento 3, no los de `output/experiments/exp3_latent_histograms`.
- Si se quiere una validacion Python frente a C para Q-maps variables, generar un nuevo experimento especifico con el mismo Q-map externo en ambas rutas.
