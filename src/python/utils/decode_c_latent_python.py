import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

# Simple decoder for C latent dumps: header + Q map + int32 latents

def read_header_and_latent(path: Path):
    header = np.fromfile(path, dtype=np.uint16, count=5)
    if header.size != 5:
        raise ValueError("Cabecera incompleta")
    bands, height, width, dtype_code, num_filters = header.tolist()
    h4 = height // 16
    w4 = width // 16
    q_map = np.fromfile(path, dtype=np.uint8, count=h4 * w4, offset=10)
    if q_map.size != h4 * w4:
        raise ValueError("Q map incompleto")
    latent = np.fromfile(path, dtype=np.int32, offset=10 + h4 * w4)
    expected = bands * num_filters * h4 * w4
    if latent.size != expected:
        raise ValueError(f"Latente incompleto: {latent.size} vs {expected}")
    latent = latent.reshape(bands, num_filters, h4, w4)
    return header, q_map.reshape(h4, w4), latent


def decode(latent_path: Path, model_root: Path, out_raw: Path, ref_raw=None, mod_bin: Path = None):
    header, q_map, latent = read_header_and_latent(latent_path)
    bands, height, width, dtype_code, num_filters = header
    h4 = height // 16
    w4 = width // 16

    lambda_clipped = (q_map[0, 0] / 255.0) * 0.125  # max_lambda fixed = 0.125

    # Load models
    modulating = tf.saved_model.load(str(model_root / "modulating")).signatures["serving_default"]
    synthesis = tf.saved_model.load(str(model_root / "synthesis")).signatures["serving_default"]
    spectral = tf.saved_model.load(str(model_root / "spectral")).signatures["serving_default"]

    if mod_bin is not None and mod_bin.exists():
        m_np = np.fromfile(mod_bin, dtype=np.float32).reshape(1, 1, 1, bands * num_filters)
        mod = tf.convert_to_tensor(m_np, dtype=tf.float32)
    else:
        q_tensor = tf.convert_to_tensor(np.full((1, h4, w4, 1), lambda_clipped, dtype=np.float32))
        mod = modulating(q_tensor)["dense_2"]  # shape (1,h4,w4, bands*num_filters)

    # Arrange latent to NHWC, dequantize
    latent_nhwc = latent.transpose(2, 3, 0, 1).reshape(1, h4, w4, bands * num_filters)
    y_hat = tf.cast(latent_nhwc, tf.float32) / mod

    # Back to (bands, h4, w4, C)
    y_hat = tf.transpose(y_hat, (0, 3, 1, 2))
    y_hat = tf.reshape(y_hat, (bands, num_filters, h4, w4))
    y_hat = tf.transpose(y_hat, (0, 2, 3, 1))

    # Synthesis
    x_hat_1d = synthesis(y_hat)["lambda_5"]

    # Inverse spectral
    I = tf.expand_dims(tf.expand_dims(tf.eye(bands), axis=1), axis=1)
    A = tf.squeeze(spectral(I)["dense"])
    B = tf.linalg.inv(A)
    x_hat_1d = tf.transpose(x_hat_1d, (0, 3, 1, 2))
    x_hat_1d = tf.reshape(x_hat_1d, (1, bands, height, width))
    x_hat_1d = tf.transpose(x_hat_1d, (0, 2, 3, 1))
    x_hat = tf.linalg.matvec(tf.linalg.matrix_transpose(B), x_hat_1d)

    # Cast to uint16
    x_hat = tf.saturate_cast(tf.saturate_cast(tf.round(x_hat), tf.int32), tf.uint16)
    arr = np.transpose(np.array(x_hat), (0, 3, 1, 2))
    arr.tofile(out_raw)

    if ref_raw is not None and ref_raw.exists():
        ref = np.fromfile(ref_raw, dtype=np.uint16).reshape(bands, height, width)
        rec = arr.reshape(bands, height, width)
        mse = np.mean((ref.astype(np.float64) - rec.astype(np.float64)) ** 2)
        psnr = 10 * np.log10((65535.0 ** 2) / (mse + 1e-12))
        print(f"PSNR vs ref: {psnr:.4f} dB")


def main():
    p = argparse.ArgumentParser(description="Decode C latent dump using TF SavedModels")
    p.add_argument("latent", type=Path)
    p.add_argument("model_root", type=Path)
    p.add_argument("out_raw", type=Path)
    p.add_argument("--ref_raw", type=Path, default=None, help="Optional reference RAW to compute PSNR")
    p.add_argument("--mod_bin", type=Path, default=None, help="Optional modulator bin (float32, 3072) to bypass SavedModel")
    args = p.parse_args()
    decode(args.latent, args.model_root, args.out_raw, args.ref_raw, args.mod_bin)


if __name__ == "__main__":
    main()
