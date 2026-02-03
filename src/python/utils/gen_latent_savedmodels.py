import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

BANDS = 8
H = 512
W = 512
MAX_LAMBDA = 0.125
SM_LAMBDA_DIV = 0.05  # el SavedModel divide internamente; no preescalar al llamar
C3_AN = 384
H4 = H // 16
W4 = W // 16


def main():
    p = argparse.ArgumentParser(description="Generate latent bin using TF SavedModels (spectral/analysis/modulating)")
    p.add_argument("raw_path", type=Path)
    p.add_argument("model_root", type=Path)
    p.add_argument("out_bin", type=Path)
    p.add_argument("--lambda_val", type=float, default=0.01)
    args = p.parse_args()

    lambda_val = float(args.lambda_val)
    if lambda_val < 0:
        lambda_val = 0.0
    if lambda_val > MAX_LAMBDA:
        lambda_val = MAX_LAMBDA
    q_byte = int(np.rint((lambda_val / MAX_LAMBDA) * 255.0))

    spectral = tf.saved_model.load(str(args.model_root / "spectral")).signatures["serving_default"]
    analysis = tf.saved_model.load(str(args.model_root / "analysis")).signatures["serving_default"]
    modulating = tf.saved_model.load(str(args.model_root / "modulating")).signatures["serving_default"]

    # RAW en BSQ (band-major). Convertimos a NHWC para los SavedModels.
    raw_bsq = np.fromfile(args.raw_path, dtype=np.uint16).reshape(BANDS, H, W)
    raw = np.transpose(raw_bsq, (1, 2, 0))
    x = tf.convert_to_tensor(raw[None, ...], dtype=tf.float32)

    x_spec = spectral(dense_input=x)["dense"]
    x_spec = x_spec / 65535.0
    x_1D = tf.transpose(x_spec, (0, 3, 1, 2))
    x_1D = tf.reshape(x_1D, (BANDS, 1, H, W))
    x_1D = tf.transpose(x_1D, (0, 2, 3, 1))

    y = analysis(lambda_input=x_1D)["layer_3"]  # expected (8,32,32,384)
    print(f"[debug] y shape {y.shape}")

    # Volcar y previo a modulación para paridad (band, channel, plane)
    y_np = y.numpy().reshape(BANDS, H4 * W4, C3_AN)
    y_dump = y_np.transpose(0, 2, 1).astype(np.float32)
    (args.out_bin.parent / "y_py.bin").write_bytes(y_dump.tobytes())

    lmda_quant = (q_byte / 255.0) * MAX_LAMBDA
    mod = modulating(lambda_8_input=tf.constant([[[[lmda_quant]]]], dtype=tf.float32))["dense_2"].numpy().reshape(BANDS, C3_AN)
    # dump modulator for comparison
    (args.out_bin.parent / "mod_py.bin").write_bytes(mod.astype(np.float32).tobytes())

    plane_sz = H4 * W4

    # Variante A: reshaping por bandas (la que ya usábamos)
    latents_A = np.empty((BANDS, C3_AN, plane_sz), dtype=np.int32)
    for b in range(BANDS):
        prod = y_np[b] * mod[b]
        latents_A[b] = np.rint(prod).astype(np.int32).T

    # Variante B: seguir exactamente el orden de SORTENY.py (flatten canales totales)
    y_3D = tf.transpose(y, (0, 3, 1, 2))               # (B, C, H4, W4)
    y_flat = tf.reshape(y_3D, (1, BANDS * C3_AN, H4, W4))
    y_flat = tf.transpose(y_flat, (0, 2, 3, 1))        # (1, H4, W4, B*C)
    mod_flat = mod.reshape(1, 1, 1, BANDS * C3_AN)
    y_hat_flat = np.rint((y_flat.numpy()) * mod_flat)
    # Reordenar a (band, channel, plane)
    latents_B = y_hat_flat.transpose(0, 3, 1, 2).reshape(BANDS, C3_AN, plane_sz)

    # Elegir variante a escribir (A por defecto)
    latents = latents_A

    args.out_bin.parent.mkdir(parents=True, exist_ok=True)
    with args.out_bin.open("wb") as f:
        header = np.array([BANDS, H, W, 2, C3_AN], dtype=np.uint16)
        header.tofile(f)
        q_map = np.full((plane_sz,), q_byte, dtype=np.uint8)
        q_map.tofile(f)
        latents.reshape(-1).tofile(f)
    print(f"Latente (variante A) escrito en {args.out_bin}")

    # Si existe un C de referencia, calcular stats
    c_path = Path("debug_dumps/out_c_new.bin")
    if c_path.exists():
        header_c = np.fromfile(c_path, dtype=np.uint16, count=5)
        Hb = int(header_c[0])
        Hc = int(header_c[1]); Wc = int(header_c[2])
        Cc = int(header_c[4]); H4c = Hc//16; W4c = Wc//16
        lat_c = np.fromfile(c_path, dtype=np.int32, offset=10+H4c*W4c).reshape(Hb, Cc, H4c*W4c)
        diffA = lat_c - latents_A
        diffB = lat_c - latents_B
        for name,d in (("A",diffA),("B",diffB)):
            absd = np.abs(d)
            print(f"[diff vs C variante {name}] mean_abs={absd.mean():.4f} max={absd.max()} zero_frac={np.count_nonzero(d==0)/d.size:.4f}")


if __name__ == "__main__":
    main()
