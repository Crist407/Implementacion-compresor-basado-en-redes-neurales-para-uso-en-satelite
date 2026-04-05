#!/usr/bin/env python3
"""
Validación end-to-end: Compara reconstucción C vs Python (SORTENY.py).

Pipeline:
  1. Carga la imagen original RAW
  2. Aplica el pipeline Python completo (análisis + cuantización + síntesis) 
  3. Lee el resultado del pipeline C (latent.bin -> decompressor C -> reconstructed.raw)
  4. Compara pixel a pixel y calcula métricas (PSNR, max diff, etc.)

Uso:
  python validate_e2e.py <original.raw> <c_reconstructed.raw> [--lambda 0.01] [--max-lambda 0.125]
  
  O modo completo (también ejecuta la pipeline Python):
  python validate_e2e.py <original.raw> <c_reconstructed.raw> --run-python --model-dir models/SORTENY_Sentinel2_model
"""

import argparse
import sys
import os
import numpy as np

# Dimensiones por defecto del modelo
BANDS = 8
HEIGHT = 512
WIDTH = 512
BIT_LENGTH = 16


def load_raw_u16(path, bands=BANDS, height=HEIGHT, width=WIDTH):
    """Carga imagen RAW BSQ uint16."""
    data = np.fromfile(path, dtype=np.uint16)
    expected = bands * height * width
    if data.size != expected:
        raise ValueError(f"Tamaño inesperado: {data.size} vs {expected} esperados")
    return data.reshape(bands, height, width)


def compute_metrics(original, reconstructed):
    """Calcula métricas de comparación entre dos imágenes uint16."""
    orig = original.astype(np.float64)
    rec = reconstructed.astype(np.float64)
    
    diff = np.abs(orig - rec)
    mse = np.mean((orig - rec) ** 2)
    max_val = 65535.0
    psnr = 10 * np.log10(max_val**2 / (mse + 1e-12))
    
    print(f"  MSE:       {mse:.4f}")
    print(f"  PSNR:      {psnr:.2f} dB")
    print(f"  Max diff:  {diff.max():.0f}")
    print(f"  Mean diff: {diff.mean():.4f}")
    print(f"  Pixels exactos: {np.sum(diff == 0)}/{diff.size} ({100*np.sum(diff==0)/diff.size:.2f}%)")
    
    # Per-band PSNR
    print("\n  PSNR por banda:")
    for b in range(orig.shape[0]):
        band_mse = np.mean((orig[b] - rec[b]) ** 2)
        band_psnr = 10 * np.log10(max_val**2 / (band_mse + 1e-12))
        band_maxdiff = np.max(np.abs(orig[b] - rec[b]))
        print(f"    Banda {b}: PSNR={band_psnr:.2f} dB, max_diff={band_maxdiff:.0f}")
    
    return psnr, mse


def run_python_pipeline(image_path, lmbda, max_lambda, model_dir, output_path):
    """Ejecuta el pipeline completo Python: análisis + cuantización + síntesis."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    import tensorflow_compression as tfc
    
    # Necesitamos las clases custom para cargar el modelo
    bit_length = BIT_LENGTH
    
    class ModulatingTransform(tf.keras.Sequential):
        def __init__(self, hidden_nodes, num_filters, maxval):
            super().__init__()
            self.add(tf.keras.layers.Lambda(lambda x: x / maxval))
            self.add(tf.keras.layers.Dense(hidden_nodes, activation=tf.nn.relu, kernel_initializer='ones'))
            self.add(tf.keras.layers.Dense(num_filters, activation=tf.nn.relu, kernel_initializer='ones'))
    
    class AnalysisTransform(tf.keras.Sequential):
        def __init__(self, num_filters_hidden, num_filters_latent):
            super().__init__(name="analysis")
            self.add(tf.keras.layers.Lambda(lambda x: x / ((2**bit_length)-1)))
            self.add(tfc.SignalConv2D(num_filters_hidden, (5, 5), name="layer_0", corr=True, strides_down=2,
                                       padding="same_zeros", use_bias=True, activation=tfc.GDN(name="gdn_0")))
            self.add(tfc.SignalConv2D(num_filters_hidden, (5, 5), name="layer_1", corr=True, strides_down=2,
                                       padding="same_zeros", use_bias=True, activation=tfc.GDN(name="gdn_1")))
            self.add(tfc.SignalConv2D(num_filters_hidden, (5, 5), name="layer_2", corr=True, strides_down=2,
                                       padding="same_zeros", use_bias=True, activation=tfc.GDN(name="gdn_2")))
            self.add(tfc.SignalConv2D(num_filters_latent, (5, 5), name="layer_3", corr=True, strides_down=2,
                                       padding="same_zeros", use_bias=False, activation=None))
    
    class SynthesisTransform(tf.keras.Sequential):
        def __init__(self, num_filters):
            super().__init__(name="synthesis")
            self.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
            self.add(tfc.SignalConv2D(num_filters, (5, 5), name="layer_0", corr=False, strides_up=1,
                                       padding="same_zeros", use_bias=True,
                                       activation=tfc.GDN(name="igdn_0", inverse=True)))
            self.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
            self.add(tfc.SignalConv2D(num_filters, (5, 5), name="layer_1", corr=False, strides_up=1,
                                       padding="same_zeros", use_bias=True,
                                       activation=tfc.GDN(name="igdn_1", inverse=True)))
            self.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
            self.add(tfc.SignalConv2D(num_filters, (5, 5), name="layer_2", corr=False, strides_up=1,
                                       padding="same_zeros", use_bias=True,
                                       activation=tfc.GDN(name="igdn_2", inverse=True)))
            self.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
            self.add(tfc.SignalConv2D(1, (5, 5), name="layer_3", corr=False, strides_up=1,
                                       padding="same_zeros", use_bias=True, activation=None))
            self.add(tf.keras.layers.Lambda(lambda x: x * ((2**bit_length)-1)))
    
    class HyperAnalysisTransform(tf.keras.Sequential):
        def __init__(self, a, b):
            super().__init__(name="hyper_analysis")
            self.add(tfc.SignalConv2D(a, (3, 3), name="layer_0", corr=True, strides_down=1, padding="same_zeros", use_bias=True, activation=tf.nn.relu))
            self.add(tfc.SignalConv2D(a, (5, 5), name="layer_1", corr=True, strides_down=2, padding="same_zeros", use_bias=True, activation=tf.nn.relu))
            self.add(tfc.SignalConv2D(b, (5, 5), name="layer_2", corr=True, strides_down=2, padding="same_zeros", use_bias=False, activation=None))
    
    class HyperSynthesisTransform(tf.keras.Sequential):
        def __init__(self, a, b):
            super().__init__(name="hyper_synthesis")
            self.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
            self.add(tfc.SignalConv2D(a, (5, 5), name="layer_0", corr=False, strides_up=1, padding="same_zeros", use_bias=True, kernel_parameter="variable", activation=tf.nn.relu))
            self.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
            self.add(tfc.SignalConv2D(a, (5, 5), name="layer_1", corr=False, strides_up=1, padding="same_zeros", use_bias=True, kernel_parameter="variable", activation=tf.nn.relu))
            self.add(tfc.SignalConv2D(b, (3, 3), name="layer_2", corr=False, strides_up=1, padding="same_zeros", use_bias=True, kernel_parameter="variable", activation=None))
    
    class SpectralAnalysisTransform(tf.keras.Sequential):
        def __init__(self, num_filters_1D, init):
            super().__init__(name="spectral_analysis")
            self.add(tf.keras.layers.Dense(num_filters_1D, activation=None, use_bias=False, kernel_initializer=init))
    
    print("  Cargando modelo Keras...")
    # El modelo guardado tiene capas Lambda que referencian 'bit_length' global
    # del script original SORTENY.py. Inyectamos la variable en builtins.
    import builtins
    builtins.bit_length = BIT_LENGTH
    model = tf.keras.models.load_model(model_dir, compile=False)
    
    # Cargar imagen
    print("  Cargando imagen...")
    raw_data = np.fromfile(image_path, dtype=np.uint16).reshape(BANDS, HEIGHT, WIDTH)
    # Convertir a NHWC float32 (como espera TF)
    x = tf.cast(tf.transpose(tf.constant(raw_data, dtype=tf.float32), (1, 2, 0)), tf.float32)
    x = tf.expand_dims(x, 0)  # (1, H, W, B)
    
    # Pipeline: spectral analysis -> analysis transform -> cuantización -> desmodulación -> synthesis -> spectral inverse
    print("  Ejecutando spectral analysis...")
    spec = model.spectral_analysis_transform
    x_spec = spec(x)  # (1, H, W, B)
    
    # Separar por bandas como hace SORTENY.py
    x_1D = tf.transpose(x_spec, (0, 3, 1, 2))  # (1, B, H, W)
    x_1D = tf.reshape(x_1D, (BANDS, 1, HEIGHT, WIDTH))
    x_1D = tf.transpose(x_1D, (0, 2, 3, 1))  # (B, H, W, 1)
    
    print("  Ejecutando analysis transform...")
    y = model.analysis_transform(x_1D)  # (B, H/16, W/16, 384)
    
    # Modulación
    print(f"  Modulación (lambda={lmbda}, max_lambda={max_lambda})...")
    lmda_val = np.clip(lmbda, 0.0, max_lambda)
    q_byte = int(round((lmda_val - 0.0) / (max_lambda - 0.0) * 255))
    lambda_quant = (q_byte / 255.0) * (max_lambda - 0.0) + 0.0
    
    mod_input = tf.constant([[[[lambda_quant]]]], dtype=tf.float32)
    mod = model.modulating_transform(mod_input)  # (1, 1, 1, 3072)
    
    num_filters = 384
    y_3D = tf.transpose(y, (0, 3, 1, 2))  # (B, 384, H/16, W/16)
    y_3D = tf.reshape(y_3D, (1, num_filters * BANDS, HEIGHT // 16, WIDTH // 16))
    y_3D = tf.transpose(y_3D, (0, 2, 3, 1))  # (1, H/16, W/16, 3072)
    
    y_3D_mod = mod * y_3D
    
    # Cuantización
    print("  Cuantización...")
    y_hat_3D = tf.round(y_3D_mod)
    
    # Desmodulación
    demod = 1.0 / mod
    y_hat_3D_demod = demod * y_hat_3D
    
    # Reshape back a per-band
    y_hat_3D_r = tf.transpose(y_hat_3D_demod, (0, 3, 1, 2))
    y_hat_3D_r = tf.reshape(y_hat_3D_r, (BANDS, num_filters, HEIGHT // 16, WIDTH // 16))
    y_hat = tf.transpose(y_hat_3D_r, (0, 2, 3, 1))  # (B, H/16, W/16, 384)
    
    # Síntesis
    print("  Ejecutando synthesis transform...")
    x_hat_1D = model.synthesis_transform(y_hat)  # (B, H, W, 1)
    
    # Espectral inversa
    print("  Ejecutando spectral inverse...")
    I = tf.expand_dims(tf.expand_dims(tf.eye(BANDS), axis=1), axis=1)
    A = tf.squeeze(spec(I))
    B = tf.linalg.inv(A)
    
    x_hat_1D_t = tf.transpose(x_hat_1D, (0, 3, 1, 2))  # (B, 1, H, W)
    x_hat_1D_t = tf.reshape(x_hat_1D_t, (1, BANDS, HEIGHT, WIDTH))
    x_hat_1D_t = tf.transpose(x_hat_1D_t, (0, 2, 3, 1))  # (1, H, W, B)
    x_hat = tf.linalg.matvec(tf.linalg.matrix_transpose(B), x_hat_1D_t)
    
    # Cast a uint16
    x_hat = tf.saturate_cast(tf.saturate_cast(tf.round(x_hat), tf.int32), tf.uint16)
    
    # Guardar como RAW BSQ
    arr = np.transpose(np.array(x_hat[0]), (2, 0, 1))  # (B, H, W)
    arr.tofile(output_path)
    print(f"  Guardado: {output_path} ({arr.nbytes} bytes)")
    
    return arr


def main():
    parser = argparse.ArgumentParser(description="Validación end-to-end C vs Python SORTENY")
    parser.add_argument("original", help="Imagen original RAW (BSQ, uint16)")
    parser.add_argument("c_reconstructed", help="Imagen reconstruida por el decodificador C")
    parser.add_argument("--run-python", action="store_true", help="Ejecutar también la pipeline Python")
    parser.add_argument("--model-dir", default="models/SORTENY_Sentinel2_model", help="Ruta al SavedModel")
    parser.add_argument("--python-output", default=None, help="Ruta para guardar la reconstrucción Python")
    parser.add_argument("--lmbda", type=float, default=0.01, help="Valor de lambda")
    parser.add_argument("--max-lambda", type=float, default=0.125, help="Valor máximo de lambda")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SORTENY End-to-End Validation: C vs Python")
    print("=" * 60)
    
    # Cargar original
    print(f"\n[1] Cargando imagen original: {args.original}")
    original = load_raw_u16(args.original)
    print(f"    Forma: {original.shape}, rango: [{original.min()}, {original.max()}]")
    
    # Cargar reconstrucción C
    print(f"\n[2] Cargando reconstrucción C: {args.c_reconstructed}")
    c_rec = load_raw_u16(args.c_reconstructed)
    print(f"    Forma: {c_rec.shape}, rango: [{c_rec.min()}, {c_rec.max()}]")
    
    # Métricas C vs Original
    print(f"\n[3] Métricas: C reconstruido vs Original")
    c_psnr, c_mse = compute_metrics(original, c_rec)
    
    # Opcionalmente ejecutar Python
    if args.run_python:
        py_output = args.python_output or args.c_reconstructed.replace('.raw', '_python.raw')
        print(f"\n[4] Ejecutando pipeline Python completa...")
        py_rec = run_python_pipeline(args.original, args.lmbda, args.max_lambda, args.model_dir, py_output)
        
        print(f"\n[5] Métricas: Python reconstruido vs Original")
        py_psnr, py_mse = compute_metrics(original, py_rec)
        
        print(f"\n[6] Métricas: C reconstruido vs Python reconstruido")
        compute_metrics(py_rec, c_rec)
        
        print(f"\n{'=' * 60}")
        print("RESUMEN COMPARATIVO")
        print(f"{'=' * 60}")
        print(f"  PSNR C vs Original:      {c_psnr:.2f} dB")
        print(f"  PSNR Python vs Original:  {py_psnr:.2f} dB")
        print(f"  Diferencia PSNR:          {abs(c_psnr - py_psnr):.4f} dB")
        
        if abs(c_psnr - py_psnr) < 0.1:
            print("\n  ✅ PASS: Las reconstrucciones son prácticamente idénticas")
        elif abs(c_psnr - py_psnr) < 1.0:
            print("\n  ⚠️  WARNING: Diferencias menores detectadas (posible error de redondeo)")
        else:
            print("\n  ❌ FAIL: Diferencias significativas detectadas")
    
    print()


if __name__ == "__main__":
    main()
