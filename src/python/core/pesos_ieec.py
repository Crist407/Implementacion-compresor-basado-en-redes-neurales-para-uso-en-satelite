#!/usr/bin/env python3
"""
Extracción de pesos del modelo TF ieec050 en formato binario float32.

- Lee los SavedModel checkpoint de `Raspberry/sorteny/models/ieec050/{spectral,analysis,synthesis,modulating}`.
- Exporta todos los tensores a un directorio destino (por defecto `weights/pesos_ieec050`).
- Opcional: aplica transposición heurística a tensores 4D (asumiendo orden [Out, In, Kh, Kw] -> [Kh, Kw, In, Out])
  para aproximar el layout que espera el encoder C. Activar con `--transpose-convs`.
- Genera un índice JSON/TSV con forma, dtype, bytes y SHA256 de cada tensor exportado.

NOTA IMPORTANTE: Las convoluciones del modelo ieec050 tienen formas (p.ej. 5x3) distintas de las que espera
el encoder C actual (5x5). Este script sirve para inspección y preparación, pero puede requerir
adaptaciones adicionales en el encoder C para aceptar estas formas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

BASE_MODEL_DIR = os.path.join("Raspberry", "sorteny", "models", "ieec050")
DEFAULT_OUT_DIR = os.path.join("weights", "pesos_ieec050")

# ----------------- Utilidades -----------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_array(arr: np.ndarray, fname: str, out_dir: str, meta: List[Dict]) -> None:
    ensure_dir(out_dir)
    path = os.path.join(out_dir, fname)
    arr.astype(np.float32).tofile(path)
    meta.append(
        {
            "filename": fname,
            "path": path,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size_bytes": int(arr.astype(np.float32).nbytes),
            "sha256": sha256_bytes(arr.astype(np.float32).tobytes()),
        }
    )
    print(f"[save] {fname:40s} shape={arr.shape} bytes={arr.nbytes}")


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name.replace(":", "_"))


def load_ckpt(subdir: str) -> Tuple[str, List[Tuple[str, Tuple[int, ...]]]]:
    ckpt = os.path.join(BASE_MODEL_DIR, subdir, "variables", "variables")
    vars_list = tf.train.list_variables(ckpt)
    return ckpt, vars_list


def maybe_transpose_conv(arr: np.ndarray, do_transpose: bool) -> np.ndarray:
    if not do_transpose:
        return arr
    if arr.ndim != 4:
        return arr
    # Heurística: TF list_variables devuelve [In, Out, Kh, Kw] en estos checkpoints.
    # Convertimos a layout C espera: [Kh, Kw, In, Out]
    return np.transpose(arr, (2, 3, 0, 1))


# ----------------- Exportadores -----------------
def export_submodel(name: str, out_dir: str, meta: List[Dict], transpose_convs: bool) -> None:
    ckpt, vars_list = load_ckpt(name)
    print(f"\n== Exportando submodelo {name} ({len(vars_list)} variables) ==")
    for var_name, shape in vars_list:
        arr = tf.train.load_variable(ckpt, var_name)
        arr_np = np.array(arr)
        if arr_np.dtype.kind not in ("f", "i", "u", "b"):
            print(f"[skip] {var_name} dtype={arr_np.dtype} no numérico")
            continue
        fname = f"{name}__{sanitize(var_name)}.bin"
        arr_np = maybe_transpose_conv(arr_np, transpose_convs)
        save_array(arr_np, fname, out_dir, meta)


def write_indexes(out_dir: str, meta: List[Dict]) -> None:
    ensure_dir(out_dir)
    idx_json = os.path.join(out_dir, "weights_index.json")
    with open(idx_json, "w", encoding="utf-8") as f:
        json.dump({"count": len(meta), "weights": meta}, f, indent=2)
    idx_tsv = os.path.join(out_dir, "weights_index.tsv")
    with open(idx_tsv, "w", encoding="utf-8") as f:
        f.write("filename\tdtype\tsize_bytes\tshape\tsha256\n")
        for m in meta:
            shape_str = "x".join(str(d) for d in m["shape"])
            f.write(f"{m['filename']}\t{m['dtype']}\t{m['size_bytes']}\t{shape_str}\t{m['sha256']}\n")
    print(f"\nÍndices escritos en:\n- {idx_json}\n- {idx_tsv}")


# ----------------- Main -----------------
def main() -> None:
    global BASE_MODEL_DIR

    parser = argparse.ArgumentParser(description="Exporta pesos de ieec050 a binarios float32")
    parser.add_argument("--model-dir", default=BASE_MODEL_DIR, help="Ruta al directorio del modelo ieec050")
    parser.add_argument("--outdir", default=DEFAULT_OUT_DIR, help="Directorio destino de los binarios")
    parser.add_argument("--transpose-convs", action="store_true", help="Trasponer tensores 4D asumiendo [Out, In, Kh, Kw] -> [Kh, Kw, In, Out]")
    args = parser.parse_args()

    BASE_MODEL_DIR = args.model_dir
    out_dir = args.outdir
    transpose_convs = args.transpose_convs

    meta: List[Dict] = []
    # Exportamos los cuatro submodelos disponibles; si alguno falta, se continúa.
    for sub in ("spectral", "analysis", "synthesis", "modulating"):
        try:
            export_submodel(sub, out_dir, meta, transpose_convs)
        except Exception as e:
            print(f"[WARN] No se pudo exportar {sub}: {e}")

    write_indexes(out_dir, meta)


if __name__ == "__main__":
    main()
