"""
Script de extracción de pesos del DECODIFICADOR desde un modelo Keras/TensorFlow.

Extrae los pesos de:
  - SynthesisTransform (4 convs + 3 IGDNs)
  - Transformada Espectral Inversa (B = inv(A), guardada como B^T)
  - ModulatingTransform (reutilizada del encoder)

Genera un directorio con todos los pesos necesarios para el decodificador C,
incluyendo los del encoder que se reutilizan (modulating, spectral).
"""

from __future__ import annotations

import os
import hashlib
import argparse
from typing import List, Optional, Dict, Any

import numpy as np

try:
    import tensorflow as tf
except Exception as e:
    raise RuntimeError("TensorFlow no está instalado.") from e

try:
    import tensorflow_compression as tfc
except Exception as e:
    raise RuntimeError("tensorflow-compression no está instalado.") from e

# --- Configuración ---
MODEL_DIR = "models/SORTENY_Sentinel2_model"
OUTPUT_DIR = "weights/pesos_decoder"
BIT_LENGTH = 16

# Alias para compatibilidad con Lambda layers del modelo
bit_length = BIT_LENGTH

# --- Clases personalizadas (necesarias para cargar el modelo) ---

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
    def __init__(self, num_filters_hidden_hyperprior, num_filters_latent_hyperprior):
        super().__init__(name="hyper_analysis")
        self.add(tfc.SignalConv2D(num_filters_hidden_hyperprior, (3, 3), name="layer_0", corr=True, strides_down=1,
                                   padding="same_zeros", use_bias=True, activation=tf.nn.relu))
        self.add(tfc.SignalConv2D(num_filters_hidden_hyperprior, (5, 5), name="layer_1", corr=True, strides_down=2,
                                   padding="same_zeros", use_bias=True, activation=tf.nn.relu))
        self.add(tfc.SignalConv2D(num_filters_latent_hyperprior, (5, 5), name="layer_2", corr=True, strides_down=2,
                                   padding="same_zeros", use_bias=False, activation=None))

class HyperSynthesisTransform(tf.keras.Sequential):
    def __init__(self, num_filters_hidden_hyperprior, num_filters_latent):
        super().__init__(name="hyper_synthesis")
        self.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
        self.add(tfc.SignalConv2D(num_filters_hidden_hyperprior, (5, 5), name="layer_0", corr=False, strides_up=1,
                                   padding="same_zeros", use_bias=True, kernel_parameter="variable", activation=tf.nn.relu))
        self.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
        self.add(tfc.SignalConv2D(num_filters_hidden_hyperprior, (5, 5), name="layer_1", corr=False, strides_up=1,
                                   padding="same_zeros", use_bias=True, kernel_parameter="variable", activation=tf.nn.relu))
        self.add(tfc.SignalConv2D(num_filters_latent, (3, 3), name="layer_2", corr=False, strides_up=1,
                                   padding="same_zeros", use_bias=True, kernel_parameter="variable", activation=None))

class SpectralAnalysisTransform(tf.keras.Sequential):
    def __init__(self, num_filters_1D, init):
        super().__init__(name="spectral_analysis")
        self.add(tf.keras.layers.Dense(num_filters_1D, activation=None, use_bias=False, kernel_initializer=init))


# --- Utilidades ---

WEIGHTS_META: List[Dict[str, Any]] = []


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_tensor_to_bin(t, filename: str, out_dir: Optional[str] = None) -> None:
    """Guarda un tensor en binario float32 y registra metadatos."""
    if out_dir is None:
        out_dir = OUTPUT_DIR
    ensure_dir(out_dir)
    filepath = os.path.join(out_dir, filename)
    arr = np.array(t).astype("float32")
    print(f"  -> Guardando {filename:45s} forma={str(arr.shape):20s} bytes={arr.nbytes}")
    arr.tofile(filepath)
    h = hashlib.sha256(arr.tobytes()).hexdigest()
    WEIGHTS_META.append({
        "filename": filename,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size_bytes": int(arr.nbytes),
        "sha256": h,
    })


def iter_all_layers(root):
    yield root
    for l in getattr(root, "layers", []):
        yield from iter_all_layers(l)


# --- Extracción principal ---

def extract_decoder_weights(model, out_dir: str) -> None:
    """Extrae TODOS los pesos necesarios para el decodificador C."""
    global OUTPUT_DIR
    OUTPUT_DIR = out_dir

    # ======== 1. Spectral Analysis Kernel (para calcular la inversa) ========
    print("\n[1/5] Transformada Espectral (kernel + inversa)...")
    spec = getattr(model, "spectral_analysis_transform", None)
    if spec is None:
        # Fallback: buscar por nombre
        for l in iter_all_layers(model):
            if getattr(l, 'name', '') == 'spectral_analysis':
                spec = l
                break

    dense = None
    if spec is not None:
        for l in getattr(spec, "layers", []):
            if isinstance(l, tf.keras.layers.Dense):
                dense = l
                break

    if dense is not None and getattr(dense, "kernel", None) is not None:
        A = dense.kernel.numpy()  # shape (8, 8)
        save_tensor_to_bin(A, "spectral_analysis_kernel.bin")

        # Calcular la inversa y guardarla traspuesta para uso directo con matvec
        A_sq = np.squeeze(A)
        B = np.linalg.inv(A_sq)
        B_T = B.T  # Traspuesta para matvec directo (apply_spectral_synthesis usa kernel^T)
        save_tensor_to_bin(B_T, "spectral_synthesis_kernel.bin")
        print(f"       A cond={np.linalg.cond(A_sq):.4f}, det(A)={np.linalg.det(A_sq):.6f}")
    else:
        print("  ERROR: No se encontró el kernel espectral!")

    # ======== 2. Modulating Transform (reutilizada) ========
    print("\n[2/5] Modulating Transform...")
    mod = getattr(model, "modulating_transform", None)
    if mod is not None and getattr(mod, "layers", None):
        for idx, layer_name in [(1, "mod_dense_1"), (2, "mod_dense_2")]:
            try:
                l = mod.layers[idx]
                if isinstance(l, tf.keras.layers.Dense):
                    if getattr(l, "kernel", None) is not None:
                        save_tensor_to_bin(l.kernel, f"{layer_name}_kernel.bin")
                    if getattr(l, "bias", None) is not None:
                        save_tensor_to_bin(l.bias, f"{layer_name}_bias.bin")
            except (IndexError, AttributeError) as e:
                print(f"  Aviso: {layer_name}: {e}")
    else:
        print("  ERROR: No se encontró modulating_transform!")

    # ======== 3. Analysis Transform (reutilizada para coherencia) ========
    print("\n[3/5] Analysis Transform...")
    analysis = getattr(model, "analysis_transform", None)
    if analysis is not None and getattr(analysis, "layers", None):
        # Estructura: [Lambda, Conv+GDN, Conv+GDN, Conv+GDN, Conv]
        for i, layer_idx in enumerate([1, 2, 3, 4]):
            try:
                l = analysis.layers[layer_idx]
                if getattr(l, "kernel", None) is not None:
                    save_tensor_to_bin(l.kernel, f"analysis_conv_{i}_kernel.bin")
                try:
                    b = l.bias
                    if b is not None:
                        save_tensor_to_bin(b, f"analysis_conv_{i}_bias.bin")
                except Exception:
                    pass
                gdn = getattr(l, "activation", None)
                if gdn is not None:
                    if getattr(gdn, "beta", None) is not None:
                        save_tensor_to_bin(gdn.beta, f"analysis_gdn_{i}_beta.bin")
                    if getattr(gdn, "gamma", None) is not None:
                        save_tensor_to_bin(gdn.gamma, f"analysis_gdn_{i}_gamma.bin")
            except (IndexError, AttributeError) as e:
                print(f"  Aviso: analysis layer {i}: {e}")
    else:
        print("  ERROR: No se encontró analysis_transform!")

    # ======== 4. Synthesis Transform (NUEVO) ========
    print("\n[4/5] Synthesis Transform (4 convs + 3 IGDNs)...")
    synth = getattr(model, "synthesis_transform", None)
    if synth is None:
        for l in iter_all_layers(model):
            if getattr(l, 'name', '') == 'synthesis':
                synth = l
                break

    if synth is not None and getattr(synth, "layers", None):
        # Estructura: [Lambda(d2s), Conv+IGDN, Lambda(d2s), Conv+IGDN, Lambda(d2s), Conv+IGDN, Lambda(d2s), Conv, Lambda(*65535)]
        # Los SignalConv2D están en posiciones 1, 3, 5, 7
        conv_indices = []
        for idx, l in enumerate(synth.layers):
            if isinstance(l, tfc.SignalConv2D):
                conv_indices.append(idx)

        print(f"       Encontrados {len(conv_indices)} SignalConv2D en posiciones: {conv_indices}")

        for i, layer_idx in enumerate(conv_indices):
            l = synth.layers[layer_idx]
            layer_name = f"synthesis_conv_{i}"

            if getattr(l, "kernel", None) is not None:
                save_tensor_to_bin(l.kernel, f"{layer_name}_kernel.bin")
            else:
                print(f"  WARN: {layer_name} sin kernel!")

            try:
                b = l.bias
                if b is not None:
                    save_tensor_to_bin(b, f"{layer_name}_bias.bin")
            except Exception:
                pass

            # Extraer IGDN (inverse=True)
            gdn = getattr(l, "activation", None)
            if gdn is not None and hasattr(gdn, "beta"):
                igdn_name = f"synthesis_igdn_{i}"
                if getattr(gdn, "beta", None) is not None:
                    save_tensor_to_bin(gdn.beta, f"{igdn_name}_beta.bin")
                if getattr(gdn, "gamma", None) is not None:
                    save_tensor_to_bin(gdn.gamma, f"{igdn_name}_gamma.bin")
                # Verificar que es inverse=True
                is_inverse = getattr(gdn, "inverse", False)
                print(f"       {igdn_name}: inverse={is_inverse}")
            elif i < 3:
                print(f"  WARN: {layer_name} sin IGDN (esperada para capa {i})!")
    else:
        print("  ERROR: No se encontró synthesis_transform!")

    # ======== 5. Resumen ========
    print(f"\n[5/5] Resumen: {len(WEIGHTS_META)} tensores exportados.")
    total_bytes = sum(m["size_bytes"] for m in WEIGHTS_META)
    print(f"       Total: {total_bytes:,} bytes ({total_bytes / 1024 / 1024:.2f} MB)")


def write_indices(out_dir: str) -> None:
    """Genera weights_index.tsv para carga directa en C."""
    tsv_path = os.path.join(out_dir, "weights_index.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("filename\tdtype\tsize_bytes\tshape\tsha256\n")
        for m in WEIGHTS_META:
            shape_str = "x".join(str(d) for d in m["shape"])
            f.write(f"{m['filename']}\t{m['dtype']}\t{m['size_bytes']}\t{shape_str}\t{m['sha256']}\n")
    print(f"Índice TSV generado: {tsv_path}")

    # También generar JSON para referencia
    import json
    json_path = os.path.join(out_dir, "weights_index.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_dir": MODEL_DIR,
            "bit_length": BIT_LENGTH,
            "count": len(WEIGHTS_META),
            "weights": WEIGHTS_META,
        }, f, indent=2, ensure_ascii=False)
    print(f"Índice JSON generado: {json_path}")


def main():
    global MODEL_DIR
    parser = argparse.ArgumentParser(description="Extracción de pesos SORTENY (encoder + decoder)")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Ruta al SavedModel")
    parser.add_argument("--outdir", default="weights/pesos_decoder", help="Directorio de salida")
    args = parser.parse_args()

    MODEL_DIR = args.model_dir
    out_dir = args.outdir

    print(f"TensorFlow: {tf.__version__}")
    print(f"tensorflow-compression: {tfc.__version__ if hasattr(tfc, '__version__') else 'unknown'}")
    print(f"Cargando modelo desde '{MODEL_DIR}'...")

    try:
        model = tf.keras.models.load_model(MODEL_DIR, compile=False)
        print("Modelo Keras cargado correctamente.")
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return

    # Mostrar estructura
    print("\nEstructura de capas:")
    for l in model.layers:
        sublayers = getattr(l, "layers", [])
        print(f"  {l.name} [{l.__class__.__name__}] sublayers={len(sublayers)}")
        for sl in sublayers[:10]:
            print(f"    {sl.name} [{sl.__class__.__name__}]")

    extract_decoder_weights(model, out_dir)
    write_indices(out_dir)

    print(f"\n=== Extracción completada: {out_dir} ===")


if __name__ == "__main__":
    main()
