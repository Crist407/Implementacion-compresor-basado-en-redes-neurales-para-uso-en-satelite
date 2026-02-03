#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np

# Mapeo de nombres de tipos a numpy dtypes
DTYPE_MAP = {
    'float32': np.float32,
    'float64': np.float64,
    'uint16': np.uint16,
    'int16': np.int16,
    'uint32': np.uint32,
    'int32': np.int32,
    'uint8': np.uint8,
    'int8': np.int8,
}

def parse_shape(shape_str: str):
    # permite formatos como 3072x32x32 o 3072,32,32
    sep = 'x' if 'x' in shape_str.lower() else (',' if ',' in shape_str else None)
    if sep is None:
        raise ValueError("Formato de --shape inválido. Usa 'CxHxW' o 'C,H,W'.")
    parts = shape_str.lower().split(sep)
    dims = tuple(int(p.strip()) for p in parts if p.strip())
    if len(dims) < 1:
        raise ValueError("--shape debe contener al menos 1 dimensión")
    return dims

def load_file(path: str, dtype: np.dtype):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")
    arr = np.fromfile(path, dtype=dtype)
    return arr


def format_idx(idx: int, shape):
    if not shape:
        return str(idx)
    # Convertir índice lineal a coordenadas según shape (fila mayor C,H,W orden planar)
    coords = []
    rem = idx
    for dim in reversed(shape):
        coords.append(rem % dim)
        rem //= dim
    coords = list(reversed(coords))
    return '(' + ','.join(str(c) for c in coords) + ')'


def compare_arrays(a: np.ndarray, b: np.ndarray, dtype_name: str, shape, atol: float, rtol: float, show: int):
    if a.size != b.size:
        print(f"ERROR: Tamaños distintos: A={a.size} elems, B={b.size} elems", file=sys.stderr)
        return 3

    if shape is not None:
        expected = int(np.prod(shape))
        if expected != a.size:
            print(f"ERROR: --shape={shape} => {expected} elems, pero archivo tiene {a.size}", file=sys.stderr)
            return 4
        a = a.reshape(shape)
        b = b.reshape(shape)

    print(f"Comparando {dtype_name} con tolerancias atol={atol} rtol={rtol} ...")

    if np.issubdtype(a.dtype, np.floating):
        equal_mask = np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=False)
        mismatches = np.where(~equal_mask.ravel())[0]
        n_mis = mismatches.size
        if n_mis == 0:
            # métricas informativas
            diff = (a - b)
            print("RESULTADO: PASS (iguales dentro de tolerancia)")
            print(f"Resumen diffs: max|diff|={np.max(np.abs(diff)):.6g}, mean|diff|={np.mean(np.abs(diff)):.6g}")
            return 0
        else:
            print(f"RESULTADO: FAIL — {n_mis} diferencias.")
            # Métricas globales
            diff = (a - b).ravel()
            print(f"Resumen diffs: max|diff|={np.max(np.abs(diff)):.6g}, mean|diff|={np.mean(np.abs(np.abs(diff))):.6g}")
            # Muestras
            to_show = mismatches[:max(1, show)]
            for i, idx in enumerate(to_show, 1):
                coord = format_idx(idx, shape)
                va = a.ravel()[idx]
                vb = b.ravel()[idx]
                print(f" #{i}: idx {coord}: A={va:.6g}  B={vb:.6g}  diff={va-vb:.6g}")
            return 2
    else:
        # Tipos enteros: igualdad exacta
        equal_mask = (a == b)
        mismatches = np.where(~equal_mask.ravel())[0]
        n_mis = mismatches.size
        if n_mis == 0:
            print("RESULTADO: PASS (idénticos)")
            return 0
        else:
            print(f"RESULTADO: FAIL — {n_mis} diferencias (comparación exacta)")
            to_show = mismatches[:max(1, show)]
            for i, idx in enumerate(to_show, 1):
                coord = format_idx(idx, shape)
                va = int(a.ravel()[idx])
                vb = int(b.ravel()[idx])
                print(f" #{i}: idx {coord}: A={va}  B={vb}")
            return 2


def main():
    ap = argparse.ArgumentParser(description="Compara dos binarios (por defecto float32) con tolerancia opcional.")
    ap.add_argument('file_a', help='Primer archivo (ej: mi_output.bin)')
    ap.add_argument('file_b', help='Segundo archivo (ej: python_ground_truth.bin)')
    ap.add_argument('--dtype', default='float32', choices=sorted(DTYPE_MAP.keys()),
                    help='Tipo de datos en los binarios (por defecto: float32)')
    ap.add_argument('--shape', type=str, default=None,
                    help="Forma esperada como 'CxHxW' o 'C,H,W'. Si se proporciona, se valida contra el tamaño.")
    ap.add_argument('--atol', type=float, default=0.0, help='Tolerancia absoluta (solo para floats)')
    ap.add_argument('--rtol', type=float, default=0.0, help='Tolerancia relativa (solo para floats)')
    ap.add_argument('--show', type=int, default=5, help='Cuántas diferencias mostrar como ejemplo')

    args = ap.parse_args()

    dtype = DTYPE_MAP[args.dtype]
    shape = None
    if args.shape:
        try:
            shape = parse_shape(args.shape)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

    try:
        a = load_file(args.file_a, dtype)
        b = load_file(args.file_b, dtype)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Archivo A: {args.file_a} — {a.size} elems ({args.dtype})")
    print(f"Archivo B: {args.file_b} — {b.size} elems ({args.dtype})")
    if shape:
        print(f"Forma esperada: {shape}")

    rc = compare_arrays(a, b, args.dtype, shape, args.atol, args.rtol, args.show)
    return rc


if __name__ == '__main__':
    sys.exit(main())
