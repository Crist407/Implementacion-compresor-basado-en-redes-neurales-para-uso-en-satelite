#!/usr/bin/env python3
"""
Comparación justa C vs Python SIN codificación aritmética.

Flujo idéntico al C:
  1. spectral_analysis → analysis_transform → modulation → round → demodulation
  2. synthesis_transform → spectral_inverse → clamp → uint16

Dumpa tensores intermedios para localizar dónde diverge el C.
"""
import builtins; builtins.bit_length = 16
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc

BANDS = 8
HEIGHT = 512
WIDTH = 512
NUM_FILTERS = 384
H4 = HEIGHT // 16
W4 = WIDTH // 16


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw"
    c_recon_path = sys.argv[2] if len(sys.argv) > 2 else "output/reconstructed_c.raw"
    model_dir = sys.argv[3] if len(sys.argv) > 3 else "models/SORTENY_Sentinel2_model"
    lmbda_val = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
    max_lambda = float(sys.argv[5]) if len(sys.argv) > 5 else 0.125

    # --- Clases custom necesarias para cargar el modelo ---
    from src.python.core.pesos_decoder import (
        ModulatingTransform, AnalysisTransform, SynthesisTransform,
        HyperAnalysisTransform, HyperSynthesisTransform, SpectralAnalysisTransform
    )

    print("=" * 60)
    print("Comparación justa C vs Python (sin entropy coding)")
    print("=" * 60)

    # 1. Cargar modelo
    print("\n[1] Cargando modelo...")
    model = tf.keras.models.load_model(model_dir, compile=False)

    # 2. Cargar imagen
    print("[2] Cargando imagen...")
    raw_data = np.fromfile(image_path, dtype=np.uint16).reshape(BANDS, HEIGHT, WIDTH)
    # Pasar a NHWC: (1, H, W, B)
    x = tf.cast(tf.constant(np.transpose(raw_data, (1, 2, 0))[np.newaxis], dtype=tf.float32), tf.float32)
    print(f"    x shape: {x.shape}, range: [{x.numpy().min():.0f}, {x.numpy().max():.0f}]")

    # 3. Spectral Analysis
    print("[3] Spectral Analysis...")
    x_spec = model.spectral_analysis_transform(x)  # (1, H, W, B)
    x_1D = tf.transpose(x_spec, (0, 3, 1, 2))
    x_1D = tf.reshape(x_1D, (BANDS, 1, HEIGHT, WIDTH))
    x_1D = tf.transpose(x_1D, (0, 2, 3, 1))  # (B, H, W, 1)
    print(f"    x_1D shape: {x_1D.shape}")

    # 4. Analysis Transform
    print("[4] Analysis Transform...")
    y = model.analysis_transform(x_1D)  # (B, H/16, W/16, num_filters)
    print(f"    y shape: {y.shape}, range: [{y.numpy().min():.4f}, {y.numpy().max():.4f}]")

    # 5. Modulation (idéntico al C)
    print(f"[5] Modulation (lambda={lmbda_val}, max_lambda={max_lambda})...")
    lmda_clipped = np.clip(lmbda_val, 0.0, max_lambda)
    q_byte = int(round(255.0 * (lmda_clipped - 0.0) / (max_lambda - 0.0)))
    lambda_quant = (q_byte / 255.0) * (max_lambda - 0.0) + 0.0
    print(f"    q_byte={q_byte}, lambda_quant={lambda_quant:.6f}")

    mod_input = tf.constant([[[[lambda_quant]]]], dtype=tf.float32)
    mod = model.modulating_transform(mod_input)  # (1, 1, 1, 3072)
    mod_flat = mod.numpy().flatten()
    print(f"    M[0]={mod_flat[0]:.4f}, M[100]={mod_flat[100]:.4f}")

    # Reshape y_3D
    y_3D = tf.transpose(y, (0, 3, 1, 2))
    y_3D = tf.reshape(y_3D, (1, NUM_FILTERS * BANDS, H4, W4))
    y_3D = tf.transpose(y_3D, (0, 2, 3, 1))  # (1, H4, W4, 3072)

    # Modular
    y_3D_mod = mod * y_3D
    print(f"    y_3D_mod range: [{y_3D_mod.numpy().min():.4f}, {y_3D_mod.numpy().max():.4f}]")

    # 6. Cuantización
    print("[6] Cuantización...")
    y_hat_3D = tf.round(y_3D_mod)
    y_hat_i32 = y_hat_3D.numpy().astype(np.int32)
    print(f"    y_hat_3D range: [{y_hat_i32.min()}, {y_hat_i32.max()}]")

    # 7. Desmodulación
    print("[7] Desmodulación...")
    demod = 1.0 / mod
    y_hat_3D_demod = demod * y_hat_3D  # (1, H4, W4, 3072)

    # Reshape a per-band NHWC: (bands, H4, W4, num_filters)
    y_hat_r = tf.transpose(y_hat_3D_demod, (0, 3, 1, 2))
    y_hat_r = tf.reshape(y_hat_r, (BANDS, NUM_FILTERS, H4, W4))
    y_hat = tf.transpose(y_hat_r, (0, 2, 3, 1))

    print(f"    y_hat (per band) shape: {y_hat.shape}, range: [{y_hat.numpy().min():.4f}, {y_hat.numpy().max():.4f}]")

    # ============ DUMPAR TENSOR DE ENTRADA A SÍNTESIS ============
    # Este es el tensor que tanto C como Python usan como entrada a la synthesis
    # Python: y_hat[band, h, w, c] (NHWC)
    # C: band_latent[c, h, w] (CHW planar)
    y_hat_np = y_hat.numpy()
    for b in range(1):  # Solo banda 0 para diagnóstico
        # Guardar como CHW para comparar directamente con el C
        y_band = y_hat_np[b]  # (H4, W4, C)
        y_band_chw = np.transpose(y_band, (2, 0, 1))  # (C, H4, W4) = (384, 32, 32)
        y_band_chw.astype(np.float32).tofile(f'/tmp/py_band{b}_input_chw.bin')
        print(f"    Band {b} input to synthesis (CHW): shape={y_band_chw.shape}, "
              f"range=[{y_band_chw.min():.4f}, {y_band_chw.max():.4f}]")
        print(f"    Band {b} first values: {y_band_chw.flatten()[:5]}")

    # 8. Synthesis Transform (layer by layer, dumping intermediates)
    print("\n[8] Synthesis Transform (layer by layer)...")

    for b in range(1):  # Solo banda 0
        x_cur = y_hat[b:b+1]  # (1, H4, W4, C)
        print(f"\n  --- Band {b} ---")

        for i, layer in enumerate(model.synthesis_transform.layers):
            x_cur = layer(x_cur)
            xnp = x_cur.numpy()
            lname = layer.__class__.__name__
            if hasattr(layer, 'name'):
                lname = f"{layer.name}({lname})"
            print(f"  Layer {i} {lname}: shape={xnp.shape}, "
                  f"range=[{xnp.min():.4f}, {xnp.max():.4f}], "
                  f"mean={xnp.mean():.4f}")

            # Guardar como CHW
            chw = np.transpose(xnp[0], (2, 0, 1))
            chw.astype(np.float32).tofile(f'/tmp/py_band{b}_layer{i}.bin')

    # 9. Synthesis completa + spectral inverse
    print("\n[9] Synthesis completa + spectral inverse...")
    x_hat_1D = model.synthesis_transform(y_hat)  # (B, H, W, 1)
    print(f"    x_hat_1D shape: {x_hat_1D.shape}, "
          f"range: [{x_hat_1D.numpy().min():.4f}, {x_hat_1D.numpy().max():.4f}]")

    # Spectral inverse
    I = tf.expand_dims(tf.expand_dims(tf.eye(BANDS), axis=1), axis=1)
    A = tf.squeeze(model.spectral_analysis_transform(I))
    B_inv = tf.linalg.inv(A)
    print(f"    A:\n{A.numpy()}")
    print(f"    B_inv:\n{B_inv.numpy()}")

    x_hat_1D_t = tf.transpose(x_hat_1D, (0, 3, 1, 2))
    x_hat_1D_t = tf.reshape(x_hat_1D_t, (1, BANDS, HEIGHT, WIDTH))
    x_hat_1D_t = tf.transpose(x_hat_1D_t, (0, 2, 3, 1))  # (1, H, W, B)
    x_hat = tf.linalg.matvec(tf.linalg.matrix_transpose(B_inv), x_hat_1D_t)

    # Clamp + cast
    x_hat = tf.saturate_cast(tf.saturate_cast(tf.round(x_hat), tf.int32), tf.uint16)
    py_rec = np.transpose(x_hat.numpy()[0], (2, 0, 1))  # (B, H, W)
    py_rec.tofile('output/reconstructed_python_fair.raw')
    print(f"    Python reconstruida: shape={py_rec.shape}, range=[{py_rec.min()}, {py_rec.max()}]")

    # 10. Comparar con C
    print("\n" + "=" * 60)
    print("[10] COMPARACIÓN")
    print("=" * 60)

    orig = raw_data  # (B, H, W)
    c_rec = np.fromfile(c_recon_path, dtype=np.uint16).reshape(BANDS, HEIGHT, WIDTH)

    # Python vs Original
    mse_py = np.mean((orig.astype(float) - py_rec.astype(float)) ** 2)
    psnr_py = 10 * np.log10(65535.0**2 / (mse_py + 1e-12))

    # C vs Original
    mse_c = np.mean((orig.astype(float) - c_rec.astype(float)) ** 2)
    psnr_c = 10 * np.log10(65535.0**2 / (mse_c + 1e-12))

    # C vs Python
    mse_cp = np.mean((py_rec.astype(float) - c_rec.astype(float)) ** 2)
    psnr_cp = 10 * np.log10(65535.0**2 / (mse_cp + 1e-12))

    print(f"  PSNR Python vs Original:  {psnr_py:.2f} dB")
    print(f"  PSNR C vs Original:       {psnr_c:.2f} dB")
    print(f"  PSNR C vs Python:         {psnr_cp:.2f} dB")
    print(f"  Max diff C vs Python:     {np.max(np.abs(c_rec.astype(int) - py_rec.astype(int)))}")
    print(f"  C rango:      [{c_rec.min()}, {c_rec.max()}]")
    print(f"  Python rango: [{py_rec.min()}, {py_rec.max()}]")

    if psnr_cp > 50:
        print("\n  ✅ PASS")
    else:
        print(f"\n  ❌ FAIL: PSNR C vs Python = {psnr_cp:.2f} dB (esperado >50)")


if __name__ == "__main__":
    main()
