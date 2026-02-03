"""
Script de extracción de pesos desde un modelo Keras/TensorFlow.
"""

from __future__ import annotations

import os
import re
import json
import hashlib
import argparse
from typing import Iterable, List, Optional, Tuple, Dict, Any

import numpy as np
from importlib import metadata as importlib_metadata

try:
    import tensorflow as tf
except Exception as e:  # pragma: no cover - entorno sin TF
    raise RuntimeError(
        "TensorFlow no está instalado. Usa un entorno virtual y ejecuta: pip install tensorflow"
    ) from e

try:
    import tensorflow_compression as tfc  # noqa: F401
except Exception as e:
    raise RuntimeError(
        "tensorflow-compression no está instalado. En tu entorno virtual ejecuta: pip install tensorflow-compression"
    ) from e


# --- Configuración ---
MODEL_DIR = "models/SORTENY_Sentinel2_model"  
OUTPUT_DIR = "weights/pesos_bin"  # Puede ser sobrescrito por argumentos CLI
BIT_LENGTH = 16                         
EXPECTED_TF_PREFIX = "2.14." 
EXPECTED_TFC_PREFIX = "2.14." 

# Algunos modelos guardados usan una Lambda que referencia 'bit_length' global.
# Definimos el alias para que la carga Keras no falle por variable no definida.
bit_length = BIT_LENGTH


# --- INICIO: Clases personalizadas ---
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
    self.add(tfc.SignalConv2D(
        num_filters_hidden, (5, 5), name="layer_0", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_0")))
    self.add(tfc.SignalConv2D(
        num_filters_hidden, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_1")))
    self.add(tfc.SignalConv2D(
        num_filters_hidden, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_2"))) 
    self.add(tfc.SignalConv2D(
        num_filters_latent, (5, 5), name="layer_3", corr=True, strides_down=2,
        padding="same_zeros", use_bias=False,
        activation=None))

class SynthesisTransform(tf.keras.Sequential):
  def __init__(self, num_filters):
    super().__init__(name="synthesis")
    self.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_0", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_0", inverse=True)))
    self.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_1", inverse=True)))
    self.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_2", inverse=True)))
    self.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
    self.add(tfc.SignalConv2D(
        1, (5, 5), name="layer_3", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True,
        activation=None))
    self.add(tf.keras.layers.Lambda(lambda x: x * ((2**bit_length)-1)))

class HyperAnalysisTransform(tf.keras.Sequential):
  def __init__(self, num_filters_hidden_hyperprior, num_filters_latent_hyperprior):
    super().__init__(name="hyper_analysis")
    self.add(tfc.SignalConv2D(
        num_filters_hidden_hyperprior, (3, 3), name="layer_0", corr=True, strides_down=1,
        padding="same_zeros", use_bias=True,
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters_hidden_hyperprior, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters_latent_hyperprior, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=False,
        activation=None))

class HyperSynthesisTransform(tf.keras.Sequential):
  def __init__(self, num_filters_hidden_hyperprior, num_filters_latent):
    super().__init__(name="hyper_synthesis")
    self.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
    self.add(tfc.SignalConv2D(
        num_filters_hidden_hyperprior, (5, 5), name="layer_0", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=tf.nn.relu))
    self.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
    self.add(tfc.SignalConv2D(
        num_filters_hidden_hyperprior, (5, 5), name="layer_1", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters_latent, (3, 3), name="layer_2", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=None))
    
class SpectralAnalysisTransform(tf.keras.Sequential):
  def __init__(self, num_filters_1D, init):
    super().__init__(name="spectral_analysis")
    self.add(tf.keras.layers.Dense(num_filters_1D, activation=None, use_bias=False, kernel_initializer=init))

# --- FIN: Clases personalizadas ---

# Esta función extrae la parte mayor.menor de una versión tipo "X.Y.Z"
# Ejemplo: "2.14.1" -> "2.14"
#          "2.13.0" -> "2.13"
def _major_minor(v: str) -> str:
    parts = v.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else v

# Verifica que TensorFlow y tensorflow-compression estén alineados (2.14.x).
def check_versions() -> None:
    """Verifica que TensorFlow y tensorflow-compression estén alineados (2.14.x)."""
    tf_v = getattr(tf, "__version__", "")
    tfc_v = getattr(tfc, "__version__", None) or ""
    if not tfc_v:
        try:
            tfc_v = importlib_metadata.version("tensorflow-compression")
        except Exception:
            tfc_v = "unknown"
    print(f"Verificando versiones -> TensorFlow={tf_v} | tensorflow-compression={tfc_v}")
    issues = []
    if not tf_v.startswith(EXPECTED_TF_PREFIX):
        issues.append(f"TensorFlow esperado {EXPECTED_TF_PREFIX}*, encontrado {tf_v}")
    if tfc_v == "unknown":
        issues.append("No se pudo determinar la versión de tensorflow-compression; asegúrate de que está instalado en el entorno actual")
    elif not tfc_v.startswith(EXPECTED_TFC_PREFIX):
        issues.append(f"tensorflow-compression esperado {EXPECTED_TFC_PREFIX}*, encontrado {tfc_v}")
    if tfc_v != "unknown" and _major_minor(tf_v) != _major_minor(tfc_v):
        issues.append("Las versiones mayor.menor de TF y TFC deben coincidir (p. ej., 2.14.x)")
    if issues:
        raise RuntimeError(
            "Incompatibilidad de versiones detectada:\n- "
            + "\n- ".join(issues)
            + "\nSugerencia: activa tu venv e instala desde requirements.txt (pip install -r requirements.txt)."
        )

# Asegura que el directorio existe
def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


WEIGHTS_META: List[Dict[str, Any]] = []  # Registro de metadatos acumulados

# Guarda un tensor en binario float32 y añade metadatos al índice global.
# Ejemplo de uso: save_tensor_to_bin(tensor, "nombre.bin", out_dir="pesos_bin/")
def save_tensor_to_bin(t: tf.Tensor, filename: str, out_dir: Optional[str] = None) -> None:
    if out_dir is None:
        out_dir = OUTPUT_DIR  # Usar valor global actual
    ensure_dir(out_dir)
    filepath = os.path.join(out_dir, filename)
    arr = tf.convert_to_tensor(t).numpy().astype("float32")
    print(f"  -> Guardando {filename} con forma {arr.shape}")
    arr.tofile(filepath)
    h = hashlib.sha256(arr.tobytes()).hexdigest()
    WEIGHTS_META.append({
        "filename": filename,
        "path": filepath,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size_bytes": int(arr.nbytes),
        "sha256": h,
    })

# Imprime un árbol de capas (nombre, clase, nº params) para depuración.
def print_layer_tree(model: tf.keras.Model, max_depth: int = 4) -> None:
    
    def _fmt(layer: tf.keras.layers.Layer) -> str:
        try:
            params = layer.count_params()
        except Exception:
            params = "?"
        return f"{layer.name} [{layer.__class__.__name__}] params={params}"

    def _recurse(layer: tf.keras.layers.Layer, depth: int) -> None:
        print("  " * depth + "- " + _fmt(layer))
        if depth >= max_depth:
            return
        sublayers = getattr(layer, "layers", None)
        if sublayers:
            for l in sublayers:
                _recurse(l, depth + 1)

    print("\nEstructura de capas (vista parcial):")
    _recurse(model, 0)
    print()

# Itera recursivamente por todas las subcapas.
def iter_all_layers(root: tf.keras.layers.Layer) -> Iterable[tf.keras.layers.Layer]:
    yield root
    sublayers = getattr(root, "layers", None)
    if sublayers:
        for l in sublayers:
            yield from iter_all_layers(l)

# Busca una capa por nombre dentro del árbol de capas.
def find_layer_by_name(root: tf.keras.layers.Layer, name: str) -> Optional[tf.keras.layers.Layer]:
    for l in iter_all_layers(root):
        if l.name == name:
            return l
    return None

# Busca la primera capa que sea instancia de alguno de los tipos dados.
def find_first_layer_of_type(root: tf.keras.layers.Layer, layer_types: Tuple[type, ...]):
    for l in iter_all_layers(root):
        if isinstance(l, layer_types):
            return l
    return None

# Extrae pesos conocidos por nombres/atributos esperados. Omite silenciosamente si no existen.
def extract_known_weights(model: tf.keras.Model, include_hyper: bool = True) -> None:

    # 1) Transformada Espectral
    print("[1/4] Intentando extraer Transformada Espectral (kernel de Dense)...")
    spec = getattr(model, "spectral_analysis_transform", None)
    dense = None
    if spec is None:
        spec = find_layer_by_name(model, "spectral_analysis")
        
    if spec is not None and getattr(spec, "layers", None):
        dense = next((l for l in spec.layers if isinstance(l, tf.keras.layers.Dense)), None)
    
    if dense is None:
        # Fallback: El SORTENY.py original la llamaba "analysis"
        candidate = find_layer_by_name(model, "analysis")
        if candidate is not None:
            dense = next((l for l in getattr(candidate, "layers", []) if isinstance(l, tf.keras.layers.Dense)), None)
            
    if dense is not None and getattr(dense, "kernel", None) is not None:
        save_tensor_to_bin(dense.kernel, "spectral_analysis_kernel.bin")
    else:
        print("  (No se encontró Dense para Transformada Espectral; omitido)")

    # 2) Transformada de Análisis (SignalConv2D + GDN)
    print("[2/4] Intentando extraer Transformada de Análisis (4 convs)...")
    analysis = getattr(model, "analysis_transform", None)
    if analysis is None:
        # Fallback: buscar la primera capa tfc.SignalConv2D y coger su "padre"
        conv = find_first_layer_of_type(model, (tfc.SignalConv2D,))
        if conv and conv.parent:
            analysis = conv.parent
        else:
            analysis = find_layer_by_name(model, "analysis_transform")

    def _save_conv_with_gdn(layer: tf.keras.layers.Layer, basename: str) -> bool:
        try:
            kernel = getattr(layer, "kernel", None)
            if kernel is not None:
                save_tensor_to_bin(kernel, f"{basename}_kernel.bin")
            try:
                bias = layer.bias  # property
            except Exception:
                bias = None
            if bias is not None:
                save_tensor_to_bin(bias, f"{basename}_bias.bin")

            gdn = getattr(layer, "activation", None)
            beta = getattr(gdn, "beta", None)
            gamma = getattr(gdn, "gamma", None)
            if beta is not None:
                save_tensor_to_bin(beta, f"{basename.replace('conv', 'gdn')}_beta.bin")
            if gamma is not None:
                save_tensor_to_bin(gamma, f"{basename.replace('conv', 'gdn')}_gamma.bin")
            return True
        except Exception as e:
            print(f"  (Aviso: Error procesando {basename}: {e})")
            return False

    if analysis is not None and getattr(analysis, "layers", None):
        # Se asume estructura: [Lambda, Conv/GDN, Conv/GDN, Conv/GDN, Conv]
        ok0 = _save_conv_with_gdn(analysis.layers[1], "analysis_conv_0")
        ok1 = _save_conv_with_gdn(analysis.layers[2], "analysis_conv_1")
        ok2 = _save_conv_with_gdn(analysis.layers[3], "analysis_conv_2")
        try:
            l3 = analysis.layers[4]
            if getattr(l3, "kernel", None) is not None:
                save_tensor_to_bin(l3.kernel, "analysis_conv_3_kernel.bin")
        except Exception:
            pass
        if not all([ok0, ok1, ok2]):
            print("  (Aviso: No se encontraron todas las convs esperadas dentro de 'analysis_transform')")
    else:
        print("  (Error: No se encontró 'analysis_transform'; omitido)")

    # 3) Transformada de Modulación (Dense -> Dense)
    print("[3/4] Intentando extraer Transformada de Modulación (2 densas)...")
    mod = getattr(model, "modulating_transform", None)
    if mod is None:
        mod = find_layer_by_name(model, "modulating_transform")
    
    def _save_dense(container: tf.keras.layers.Layer, idx: int, base: str) -> bool:
        try:
            l = container.layers[idx]
        except Exception:
            return False
        if not isinstance(l, tf.keras.layers.Dense):
            return False
        if getattr(l, "kernel", None) is not None:
            save_tensor_to_bin(l.kernel, f"{base}_kernel.bin")
        if getattr(l, "bias", None) is not None:
            save_tensor_to_bin(l.bias, f"{base}_bias.bin")
        return True
        
    if mod is not None and getattr(mod, "layers", None):
        # Se asume estructura: [Lambda, Dense, Dense]
        okd0 = _save_dense(mod, 1, "mod_dense_0")
        okd1 = _save_dense(mod, 2, "mod_dense_1")
        if not any([okd0, okd1]):
            print("  (Aviso: No se encontraron densas esperadas dentro de 'modulating_transform')")
    else:
        print("  (Error: No se encontró 'modulating_transform'; omitido)")

    # 4) Hiper-Análisis (3 convs; la 3ª sin bias)
    if include_hyper:
        print("[4/4] Intentando extraer Hiper-Análisis (3 convs)...")
        hyper_an = getattr(model, "hyper_analysis_transform", None)
        if hyper_an is None:
            hyper_an = find_layer_by_name(model, "hyper_analysis_transform")

        if hyper_an is not None and getattr(hyper_an, "layers", None):
            for i in range(3):  # conv(relu), conv(relu), conv
                try:
                    l = hyper_an.layers[i]
                    if getattr(l, "kernel", None) is not None:
                        save_tensor_to_bin(l.kernel, f"hyper_an_conv_{i}_kernel.bin")
                    try:
                        b = l.bias
                    except Exception:
                        b = None
                    if b is not None:
                        save_tensor_to_bin(b, f"hyper_an_conv_{i}_bias.bin")
                except Exception:
                    print(f"  (Aviso: capa {i} de hyper_an no encontrada)")
        else:
            print("  (Error: No se encontró 'hyper_analysis_transform'; omitido)")
    else:
        print("[4/4] (minimal) Hiper-Análisis omitido por --minimal")

# Sanitiza nombres de archivo para evitar caracteres problemáticos.
# Ejemplo: "analysis/layer_0/kernel:0" -> "analysis_layer_0_kernel_0"
def sanitize_filename(name: str) -> str:
    name = name.replace(":", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return name

# Exporta TODAS las variables del objeto SavedModel genérico.
def export_all_variables_generic(trackable_obj, out_dir: str = OUTPUT_DIR) -> None:
    ensure_dir(out_dir)
    vars_list = list(trackable_obj.variables)
    print(f"Exportando {len(vars_list)} variables del SavedModel (fallback genérico)...")
    for v in vars_list:
        fname = sanitize_filename(v.name) + ".bin"
        save_tensor_to_bin(v, fname, out_dir)


def main():
    global MODEL_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Extracción de pesos SORTENY")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Ruta al SavedModel")
    parser.add_argument("--outdir", default=None, help="Directorio destino de pesos (por defecto auto según --minimal)")
    parser.add_argument("--minimal", action="store_true", help="Extrae sólo spectral, analysis y modulating (omitiendo hyper análisis)")
    args = parser.parse_args()

    MODEL_DIR = args.model_dir
    if args.outdir:
        OUTPUT_DIR = args.outdir
    else:
        OUTPUT_DIR = "weights/pesos_bin_minimal" if args.minimal else "weights/pesos_bin"
    # Verificación de versiones antes de continuar
    try:
        check_versions()
    except RuntimeError as e:
        print(f"Error de entorno: {e}")
        return

    print(f"Cargando modelo desde '{MODEL_DIR}'...")
    keras_model: Optional[tf.keras.Model] = None
    try:
        keras_model = tf.keras.models.load_model(MODEL_DIR, compile=False)
        print("Modelo Keras cargado correctamente (Modo de Extracción Específico).")
    except Exception as e:
        print(f"No se pudo cargar como Keras Model: {e}")
        keras_model = None

    if keras_model is not None:
        print_layer_tree(keras_model)
        extract_known_weights(keras_model, include_hyper=not args.minimal)
    else:
        print("Intentando fallback: tf.saved_model.load (Modo de Extracción Genérico)...")
        try:
            sm = tf.saved_model.load(MODEL_DIR)
            export_all_variables_generic(sm, OUTPUT_DIR)
        except Exception as e:
            print(f"Error fatal: No se pudo cargar el modelo ni como Keras ni como SavedModel genérico.")
            print(f"Detalle: {e}")
            return

    print("\n--- Extracción Completada ---")
    print(f"Todos los pesos han sido guardados en: {os.path.abspath(OUTPUT_DIR)}")

    # Escritura del índice JSON
    index_path = os.path.join(OUTPUT_DIR, "weights_index.json")
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_dir": MODEL_DIR,
                "bit_length": BIT_LENGTH,
                "count": len(WEIGHTS_META),
                "weights": WEIGHTS_META,
            }, f, indent=2, ensure_ascii=False)
        print(f"Índice JSON generado: {index_path}")
    except Exception as e:
        print(f"Error al escribir el índice JSON: {e}")

    # Generar índice TSV simple para carga directa en C
    tsv_path = os.path.join(OUTPUT_DIR, "weights_index.tsv")
    try:
        with open(tsv_path, "w", encoding="utf-8") as ftsv:
            ftsv.write("filename\tdtype\tsize_bytes\tshape\tsha256\n")
            for m in WEIGHTS_META:
                shape_str = "x".join(str(d) for d in m["shape"])
                ftsv.write(f"{m['filename']}\t{m['dtype']}\t{m['size_bytes']}\t{shape_str}\t{m['sha256']}\n")
        print(f"Índice TSV generado: {tsv_path}")
    except Exception as e:
        print(f"Error al escribir el índice TSV: {e}")


if __name__ == "__main__":
    main()