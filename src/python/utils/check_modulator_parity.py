import numpy as np
import tensorflow as tf
from pathlib import Path

BANDS = 8
C_MID = 192
C_OUT = BANDS * 384  # 3072
MAX_LAMBDA = 0.125
SM_LAMBDA_DIV = 0.05  # el SavedModel divide internamente; no preescalar al llamar


def load_bins(weights_dir: Path):
    # Load modulator weights from bin files
    names0 = [
        "mod_dense_0_kernel.bin",
        "modulating__layer_with_weights-0_kernel_.ATTRIBUTES_VARIABLE_VALUE.bin",
    ]
    names1 = [
        "mod_dense_1_kernel.bin",
        "modulating__layer_with_weights-1_kernel_.ATTRIBUTES_VARIABLE_VALUE.bin",
    ]
    bias0 = [
        "mod_dense_0_bias.bin",
        "modulating__layer_with_weights-0_bias_.ATTRIBUTES_VARIABLE_VALUE.bin",
    ]
    bias1 = [
        "mod_dense_1_bias.bin",
        "modulating__layer_with_weights-1_bias_.ATTRIBUTES_VARIABLE_VALUE.bin",
    ]

    def pick(names):
        for n in names:
            p = weights_dir / n
            if p.exists():
                return np.fromfile(p, dtype=np.float32)
        return None

    k0 = pick(names0)
    k1 = pick(names1)
    b0 = pick(bias0)
    b1 = pick(bias1)
    # reshape
    if k0 is not None:
        k0 = k0.reshape(1, C_MID)
    if k1 is not None:
        k1 = k1.reshape(C_MID, C_OUT)
    return k0, b0, k1, b1


def mod_forward_numpy(lambda_quant: float, k0, b0, k1, b1):
    x = np.array([lambda_quant], dtype=np.float32)  # shape (1,)
    y = x @ k0  # (1,192)
    y = np.maximum(y + b0, 0.0)
    z = y @ k1  # (1,3072)
    z = np.maximum(z + b1, 0.0)
    return z.reshape(C_OUT)


def main():
    weights_dir = Path("weights/pesos_ieec050_k5x3")
    model_root = Path("Raspberry/sorteny/models/ieec050")

    # lambda
    lambda_val = 0.01
    q_byte = int(np.rint((lambda_val / MAX_LAMBDA) * 255.0))
    lambda_quant = (q_byte / 255.0) * MAX_LAMBDA
    print(f"lambda={lambda_val}, q_byte={q_byte}, lambda_quant={lambda_quant}")

    # 1) Modulator from bins (numpy)
    k0, b0, k1, b1 = load_bins(weights_dir)
    m_bin = mod_forward_numpy(lambda_quant, k0, b0, k1, b1)
    m_bin_scaled = mod_forward_numpy(lambda_quant / SM_LAMBDA_DIV, k0, b0, k1, b1)
    print("bin M stats:", m_bin.mean(), m_bin.std(), m_bin.min(), m_bin.max())
    print("bin M (lambda/0.05) stats:", m_bin_scaled.mean(), m_bin_scaled.std(), m_bin_scaled.min(), m_bin_scaled.max())

    # 2) Modulator from SavedModel
    mod_model = tf.saved_model.load(str(model_root / "modulating"))
    mod_sm = mod_model.signatures["serving_default"]
    # Pasar lambda cuantizado tal cual; el grafo ya hace la división interna por 0.05
    m_tf = mod_sm(lambda_8_input=tf.constant([[[[lambda_quant]]]], dtype=tf.float32))["dense_2"]
    m_tf = m_tf.numpy().reshape(C_OUT)
    print("TF M stats:", m_tf.mean(), m_tf.std(), m_tf.min(), m_tf.max())

    diff_sm = m_bin - m_tf
    diff_sm_scaled = m_bin_scaled - m_tf
    print("bin vs TF diff mean_abs=", np.mean(np.abs(diff_sm)), "max=", np.max(np.abs(diff_sm)))
    print("bin(lambda/0.05) vs TF diff mean_abs=", np.mean(np.abs(diff_sm_scaled)), "max=", np.max(np.abs(diff_sm_scaled)))

    # 2b) Weight parity bins vs SavedModel
    var_dict = {v.name: v.numpy() for v in mod_model.variables}
    k0_tf = b0_tf = k1_tf = b1_tf = None
    for name, val in var_dict.items():
        if val.shape == (1, C_MID) and k0_tf is None:
            k0_tf = val
        elif val.shape == (C_MID,) and b0_tf is None:
            b0_tf = val
        elif val.shape == (C_MID, C_OUT) and k1_tf is None:
            k1_tf = val
        elif val.shape == (C_OUT,) and b1_tf is None:
            b1_tf = val
    if k0_tf is not None:
        print("dense0 kernel diff mean_abs=", np.mean(np.abs(k0 - k0_tf)), "max=", np.max(np.abs(k0 - k0_tf)))
    if b0_tf is not None:
        print("dense0 bias diff mean_abs=", np.mean(np.abs(b0 - b0_tf)), "max=", np.max(np.abs(b0 - b0_tf)))
    if k1_tf is not None:
        print("dense1 kernel diff mean_abs=", np.mean(np.abs(k1 - k1_tf)), "max=", np.max(np.abs(k1 - k1_tf)))
    if b1_tf is not None:
        print("dense1 bias diff mean_abs=", np.mean(np.abs(b1 - b1_tf)), "max=", np.max(np.abs(b1 - b1_tf)))

    # 3) Modulator dumped by C and by Python latent generator if present
    m_c_path = Path("debug_dumps/M_c.bin")
    m_py_path = Path("debug_dumps/mod_py.bin")
    if m_c_path.exists():
        m_c = np.fromfile(m_c_path, dtype=np.float32)
        print("C M stats:", m_c.mean(), m_c.std(), m_c.min(), m_c.max())
        diff_c_bin = m_c - m_bin
        diff_c_bin_scaled = m_c - m_bin_scaled
        diff_c_tf = m_c - m_tf
        print("C vs bin diff mean_abs=", np.mean(np.abs(diff_c_bin)), "max=", np.max(np.abs(diff_c_bin)))
        print("C vs bin(lambda/0.05) diff mean_abs=", np.mean(np.abs(diff_c_bin_scaled)), "max=", np.max(np.abs(diff_c_bin_scaled)))
        print("C vs TF diff mean_abs=", np.mean(np.abs(diff_c_tf)), "max=", np.max(np.abs(diff_c_tf)))
    if m_py_path.exists():
        m_py = np.fromfile(m_py_path, dtype=np.float32)
        print("PY M stats:", m_py.mean(), m_py.std(), m_py.min(), m_py.max())
        diff_py_bin = m_py - m_bin
        print("PY vs bin diff mean_abs=", np.mean(np.abs(diff_py_bin)), "max=", np.max(np.abs(diff_py_bin)))


if __name__ == "__main__":
    main()
