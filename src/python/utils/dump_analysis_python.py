import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

# Dimensions for ieec050 model
BANDS = 8
H_IN = 512
W_IN = 512
C0 = 128
C1 = 128
C2 = 128
C3_AN = 384
H1, W1 = H_IN // 2, W_IN // 2
H2, W2 = H1 // 2, W1 // 2
H3, W3 = H2 // 2, W2 // 2
H4, W4 = H3 // 2, W3 // 2
NORM_CONST = 65535.0

# Helper --------------------------------------------------------------------

def parse_shape(shape_str: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in shape_str.replace("X", "x").split("x"))


def load_weights(base: Path) -> Dict[str, np.ndarray]:
    index_path = base / "weights_index.tsv"
    weights: Dict[str, np.ndarray] = {}
    with index_path.open("r") as f:
        header = f.readline()
        if not header:
            raise ValueError("weights_index.tsv vacío")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            fname, dtype, size_bytes, shape_str = parts[:4]
            shape = parse_shape(shape_str)
            dtype_np = np.float32 if dtype == "float32" else None
            if dtype_np is None:
                continue
            arr = np.fromfile(base / fname, dtype=dtype_np)
            if np.prod(shape) != arr.size:
                raise ValueError(f"Tamaño inesperado para {fname}: {arr.size} vs {shape}")
            weights[fname] = arr.reshape(shape)
    return weights


def pick(weights: Dict[str, np.ndarray], names: List[str]) -> np.ndarray:
    for n in names:
        if n in weights:
            return weights[n]
    raise KeyError(f"Ningún peso encontrado para {names}")


def pad_same_tf(x: np.ndarray, kH: int, kW: int, stride: int) -> Tuple[np.ndarray, int, int]:
    H, W = x.shape[1:]
    H_out = (H + stride - 1) // stride
    W_out = (W + stride - 1) // stride
    pad_total_y = max((H_out - 1) * stride + kH - H, 0)
    pad_total_x = max((W_out - 1) * stride + kW - W, 0)
    pad_top = pad_total_y // 2
    pad_left = pad_total_x // 2
    pad_bottom = pad_total_y - pad_top
    pad_right = pad_total_x - pad_left
    x_pad = np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
    return x_pad, H_out, W_out


def conv2d_same_stride2_tf(x: np.ndarray, kernel: np.ndarray, bias: Optional[np.ndarray]) -> np.ndarray:
    """Convolución rápida con tf.nn.conv2d (datos en formato C,H,W)."""
    # x: (C_in, H, W) -> NHWC for TF
    x_tf = tf.convert_to_tensor(x[None].transpose(0, 2, 3, 1), dtype=tf.float32)
    k_tf = tf.convert_to_tensor(kernel, dtype=tf.float32)  # kH,kW,Cin,Cout
    y = tf.nn.conv2d(x_tf, k_tf, strides=2, padding="SAME")
    if bias is not None:
        y = y + tf.reshape(tf.convert_to_tensor(bias, dtype=tf.float32), (1, 1, 1, -1))
    y = tf.transpose(y[0], (2, 0, 1))  # -> (C_out, H_out, W_out)
    return y.numpy()


def gdn_tf(x: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """GDN vectorizada usando einsum (x en formato C,H,W)."""
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    beta_tf = tf.convert_to_tensor(np.maximum(beta, 1e-6), dtype=tf.float32)
    gamma_tf = tf.convert_to_tensor(gamma, dtype=tf.float32)  # gamma[j, i]
    x_abs = tf.abs(x_tf)
    # denom[i] = beta[i] + sum_j gamma[j,i] * |x_j|
    denom = beta_tf[:, None, None] + tf.einsum("ji,jhw->ihw", gamma_tf, x_abs)
    out = x_tf / denom
    return out.numpy()


def save_planar(path: Path, tensor: np.ndarray) -> None:
    # tensor shape (C, H, W)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor.astype(np.float32).tofile(path)


def main():
    parser = argparse.ArgumentParser(description="Dump Python-side spectral and analysis intermediates")
    parser.add_argument("raw_path", type=Path)
    parser.add_argument("weights_dir", type=Path)
    parser.add_argument("out_dir", type=Path, default=Path("debug_dumps"))
    parser.add_argument("--model_root", type=Path, default=Path("Raspberry/sorteny/models/ieec050"))
    args = parser.parse_args()

    weights = load_weights(args.weights_dir)

    # Map weights
    spectral_w = pick(weights, [
        "spectral_analysis_kernel.bin",
        "spectral__layer_with_weights-0_kernel_.ATTRIBUTES_VARIABLE_VALUE.bin",
    ])  # shape (8,8)

    conv0_k = pick(weights, [
        "analysis_conv_0_kernel.bin",
        "analysis__variables_3_.ATTRIBUTES_VARIABLE_VALUE.bin",
        "analysis__variables_4_.ATTRIBUTES_VARIABLE_VALUE.bin",
    ])
    conv0_b = pick(weights, [
        "analysis_conv_0_bias.bin",
        "analysis__layer_with_weights-0__bias_parameter_.ATTRIBUTES_VARIABLE_VALUE.bin",
    ])
    gdn0_beta = pick(weights, ["analysis_gdn_0_beta.bin", "analysis__variables_1_.ATTRIBUTES_VARIABLE_VALUE.bin"])
    gdn0_gamma = pick(weights, ["analysis_gdn_0_gamma.bin", "analysis__variables_2_.ATTRIBUTES_VARIABLE_VALUE.bin"])

    conv1_k = pick(weights, [
        "analysis_conv_1_kernel.bin",
        "analysis__variables_8_.ATTRIBUTES_VARIABLE_VALUE.bin",
        "analysis__variables_9_.ATTRIBUTES_VARIABLE_VALUE.bin",
    ])
    conv1_b = pick(weights, [
        "analysis_conv_1_bias.bin",
        "analysis__layer_with_weights-1__bias_parameter_.ATTRIBUTES_VARIABLE_VALUE.bin",
    ])
    gdn1_beta = pick(weights, ["analysis_gdn_1_beta.bin", "analysis__variables_6_.ATTRIBUTES_VARIABLE_VALUE.bin"])
    gdn1_gamma = pick(weights, ["analysis_gdn_1_gamma.bin", "analysis__variables_7_.ATTRIBUTES_VARIABLE_VALUE.bin"])

    conv2_k = pick(weights, [
        "analysis_conv_2_kernel.bin",
        "analysis__variables_13_.ATTRIBUTES_VARIABLE_VALUE.bin",
        "analysis__variables_14_.ATTRIBUTES_VARIABLE_VALUE.bin",
    ])
    conv2_b = pick(weights, [
        "analysis_conv_2_bias.bin",
        "analysis__layer_with_weights-2__bias_parameter_.ATTRIBUTES_VARIABLE_VALUE.bin",
    ])
    gdn2_beta = pick(weights, ["analysis_gdn_2_beta.bin", "analysis__variables_11_.ATTRIBUTES_VARIABLE_VALUE.bin"])
    gdn2_gamma = pick(weights, ["analysis_gdn_2_gamma.bin", "analysis__variables_12_.ATTRIBUTES_VARIABLE_VALUE.bin"])

    conv3_k = pick(weights, [
        "analysis_conv_3_kernel.bin",
        "analysis__variables_15_.ATTRIBUTES_VARIABLE_VALUE.bin",
        "analysis__variables_16_.ATTRIBUTES_VARIABLE_VALUE.bin",
    ])

    # Load models for spectral only (avoids reimplementing spectral inverse)
    spectral_model = tf.saved_model.load(str(args.model_root / "spectral")).signatures["serving_default"]

    # Load image
    # Entrada en formato BSQ (band-major). Reordenamos a NHWC para el SavedModel.
    raw_bsq = np.fromfile(args.raw_path, dtype=np.uint16).reshape(BANDS, H_IN, W_IN)
    raw = np.transpose(raw_bsq, (1, 2, 0))
    x_tf = tf.convert_to_tensor(raw[None, ...], dtype=tf.float32)
    x_spec = spectral_model(dense_input=x_tf)["dense"].numpy()[0]  # (H,W,B)
    spec_norm = x_spec / NORM_CONST

    # Save spectral dumps
    spec_planar = np.transpose(spec_norm, (2, 0, 1))  # (B,H,W)
    save_planar(args.out_dir / "spectral_py.bin", spec_planar)

    # Allocate outputs
    conv0_all = np.empty((BANDS, C0, H1, W1), dtype=np.float32)
    gdn0_all = np.empty_like(conv0_all)
    conv1_all = np.empty((BANDS, C1, H2, W2), dtype=np.float32)
    gdn1_all = np.empty_like(conv1_all)
    conv2_all = np.empty((BANDS, C2, H3, W3), dtype=np.float32)
    gdn2_all = np.empty_like(conv2_all)
    conv3_all = np.empty((BANDS, C3_AN, H4, W4), dtype=np.float32)

    # Kernels to NHWC? We need kH,kW,Cin,Cout already.
    k0 = np.array(conv0_k, dtype=np.float32)
    k1 = np.array(conv1_k, dtype=np.float32)
    k2 = np.array(conv2_k, dtype=np.float32)
    k3 = np.array(conv3_k, dtype=np.float32)

    b0 = np.array(conv0_b, dtype=np.float32)
    b1 = np.array(conv1_b, dtype=np.float32)
    b2 = np.array(conv2_b, dtype=np.float32)

    g0_beta = np.array(gdn0_beta, dtype=np.float32).reshape(-1)
    g0_gamma = np.array(gdn0_gamma, dtype=np.float32)
    g1_beta = np.array(gdn1_beta, dtype=np.float32).reshape(-1)
    g1_gamma = np.array(gdn1_gamma, dtype=np.float32)
    g2_beta = np.array(gdn2_beta, dtype=np.float32).reshape(-1)
    g2_gamma = np.array(gdn2_gamma, dtype=np.float32)

    for b in range(BANDS):
        band_in = spec_planar[b]
        # conv0
        c0 = conv2d_same_stride2_tf(band_in[None, ...], k0, b0)
        conv0_all[b] = c0
        g0 = gdn_tf(c0, g0_beta, g0_gamma)
        gdn0_all[b] = g0
        # conv1
        c1 = conv2d_same_stride2_tf(g0, k1, b1)
        conv1_all[b] = c1
        g1 = gdn_tf(c1, g1_beta, g1_gamma)
        gdn1_all[b] = g1
        # conv2
        c2 = conv2d_same_stride2_tf(g1, k2, b2)
        conv2_all[b] = c2
        g2 = gdn_tf(c2, g2_beta, g2_gamma)
        gdn2_all[b] = g2
        # conv3 (no bias)
        c3 = conv2d_same_stride2_tf(g2, k3, None)
        conv3_all[b] = c3

    # Save dumps in same planar ordering as C (band-major)
    save_planar(args.out_dir / "conv0_pre_py.bin", conv0_all.reshape(BANDS * C0, H1, W1))
    save_planar(args.out_dir / "gdn0_py.bin", gdn0_all.reshape(BANDS * C0, H1, W1))
    save_planar(args.out_dir / "gdn1_py.bin", gdn1_all.reshape(BANDS * C1, H2, W2))
    save_planar(args.out_dir / "gdn2_py.bin", gdn2_all.reshape(BANDS * C2, H3, W3))
    save_planar(args.out_dir / "conv3_py.bin", conv3_all.reshape(BANDS * C3_AN, H4, W4))

if __name__ == "__main__":
    main()
