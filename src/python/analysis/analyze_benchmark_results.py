#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis de resultados del benchmark SORTENY
Ejecutar en PC local después de transferir resultados de la Raspberry Pi

Uso:
    python analyze_benchmark_results.py <directorio_resultados> [imagen_original.raw]
"""

import os
import sys
import glob
import re
import numpy as np
from pathlib import Path

# Parámetros del modelo
BANDS = 8
H_IN, W_IN = 512, 512
NUM_FILTERS = 384
H_LATENT, W_LATENT = 32, 32


def read_compressed_bin(filepath):
    """Lee un archivo .bin comprimido (formato SORTENY)."""
    with open(filepath, 'rb') as f:
        # Header: 5 x uint16
        header = np.fromfile(f, dtype=np.uint16, count=5)
        bands = int(header[0])
        height = int(header[1])
        width = int(header[2])
        datatype = int(header[3])
        num_filters = int(header[4])
        
        # Q-map
        qmap_size = (height // 16) * (width // 16)
        qmap = np.fromfile(f, dtype=np.uint8, count=qmap_size)
        
        # Latentes
        latents = np.fromfile(f, dtype=np.int32)
        
    return {
        'header': {'bands': bands, 'height': height, 'width': width, 
                   'datatype': datatype, 'num_filters': num_filters},
        'qmap': qmap,
        'latents': latents
    }


def compare_latents(data_c, data_py):
    """Compara latentes entre C y Python."""
    lat_c = data_c['latents']
    lat_py = data_py['latents']
    
    print("\n" + "="*60)
    print(" COMPARACIÓN DE LATENTES (C vs Python)")
    print("="*60)
    
    # Verificar dimensiones
    print(f"\nDimensiones:")
    print(f"  C:      {lat_c.shape} ({lat_c.size} elementos)")
    print(f"  Python: {lat_py.shape} ({lat_py.size} elementos)")
    
    if lat_c.size != lat_py.size:
        print("  ⚠️  TAMAÑOS DIFERENTES!")
        min_size = min(lat_c.size, lat_py.size)
        lat_c = lat_c[:min_size]
        lat_py = lat_py[:min_size]
    
    # Comparación exacta
    exact_match = np.array_equal(lat_c, lat_py)
    print(f"\n¿Match exacto?: {'✅ SÍ' if exact_match else '❌ NO'}")
    
    if not exact_match:
        # Diferencias
        diff = lat_c.astype(np.int64) - lat_py.astype(np.int64)
        abs_diff = np.abs(diff)
        
        num_diff = np.sum(diff != 0)
        pct_diff = 100.0 * num_diff / lat_c.size
        
        print(f"\nEstadísticas de diferencias:")
        print(f"  Elementos diferentes: {num_diff:,} / {lat_c.size:,} ({pct_diff:.4f}%)")
        print(f"  Diff absoluta media:  {np.mean(abs_diff):.6f}")
        print(f"  Diff absoluta máxima: {np.max(abs_diff)}")
        print(f"  Diff absoluta std:    {np.std(abs_diff):.6f}")
        
        # Histograma de diferencias
        if num_diff > 0:
            unique_diffs, counts = np.unique(diff[diff != 0], return_counts=True)
            print(f"\n  Distribución de diferencias (top 10):")
            sorted_idx = np.argsort(-counts)[:10]
            for idx in sorted_idx:
                print(f"    diff={unique_diffs[idx]:+d}: {counts[idx]:,} veces")
    
    # Estadísticas generales
    print(f"\nEstadísticas de valores:")
    print(f"  C:      min={lat_c.min()}, max={lat_c.max()}, mean={lat_c.mean():.2f}, std={lat_c.std():.2f}")
    print(f"  Python: min={lat_py.min()}, max={lat_py.max()}, mean={lat_py.mean():.2f}, std={lat_py.std():.2f}")
    
    return exact_match, diff if not exact_match else None


def parse_benchmark_log(filepath):
    """Extrae métricas de un log de benchmark."""
    metrics = {}
    
    if not os.path.exists(filepath):
        return metrics
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Tiempo de usuario
    match = re.search(r'User time \(seconds\): ([\d.]+)', content)
    if match:
        metrics['user_time'] = float(match.group(1))
    
    # Tiempo de sistema
    match = re.search(r'System time \(seconds\): ([\d.]+)', content)
    if match:
        metrics['system_time'] = float(match.group(1))
    
    # Tiempo total (elapsed)
    match = re.search(r'Elapsed \(wall clock\) time.*: ([\d:]+\.?\d*)', content)
    if match:
        time_str = match.group(1)
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 2:
                metrics['elapsed_time'] = float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                metrics['elapsed_time'] = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        else:
            metrics['elapsed_time'] = float(time_str)
    
    # Memoria máxima
    match = re.search(r'Maximum resident set size \(kbytes\): (\d+)', content)
    if match:
        metrics['max_memory_kb'] = int(match.group(1))
        metrics['max_memory_mb'] = metrics['max_memory_kb'] / 1024
    
    # Tiempo de inferencia (específico de SORTENY)
    match = re.search(r'\[BENCHMARK\] Tiempo Inferencia.*: ([\d.]+)', content)
    if match:
        metrics['inference_time'] = float(match.group(1))
    
    return metrics


def analyze_results(results_dir, original_image=None):
    """Analiza todos los resultados en un directorio."""
    results_path = Path(results_dir)
    
    print("="*60)
    print(" ANÁLISIS DE RESULTADOS SORTENY")
    print("="*60)
    print(f"Directorio: {results_path}")
    print()
    
    # Buscar archivos
    c_bins = sorted(results_path.glob("output_c_*.bin"))
    py_bins = sorted(results_path.glob("output_py_*.bin"))
    c_logs = sorted(results_path.glob("benchmark_c_*.log"))
    py_logs = sorted(results_path.glob("benchmark_py_*.log"))
    
    print(f"Archivos encontrados:")
    print(f"  Binarios C:      {len(c_bins)}")
    print(f"  Binarios Python: {len(py_bins)}")
    print(f"  Logs C:          {len(c_logs)}")
    print(f"  Logs Python:     {len(py_logs)}")
    
    # Analizar benchmarks
    if c_logs:
        print("\n" + "-"*60)
        print(" BENCHMARK C")
        print("-"*60)
        for log in c_logs:
            metrics = parse_benchmark_log(log)
            print(f"\n  {log.name}:")
            if 'elapsed_time' in metrics:
                print(f"    Tiempo total:     {metrics['elapsed_time']:.2f} s")
            if 'user_time' in metrics:
                print(f"    Tiempo usuario:   {metrics['user_time']:.2f} s")
            if 'max_memory_mb' in metrics:
                print(f"    Memoria máxima:   {metrics['max_memory_mb']:.1f} MB")
    
    if py_logs:
        print("\n" + "-"*60)
        print(" BENCHMARK PYTHON")
        print("-"*60)
        for log in py_logs:
            metrics = parse_benchmark_log(log)
            print(f"\n  {log.name}:")
            if 'elapsed_time' in metrics:
                print(f"    Tiempo total:     {metrics['elapsed_time']:.2f} s")
            if 'user_time' in metrics:
                print(f"    Tiempo usuario:   {metrics['user_time']:.2f} s")
            if 'inference_time' in metrics:
                print(f"    Tiempo inferencia:{metrics['inference_time']:.2f} s")
            if 'max_memory_mb' in metrics:
                print(f"    Memoria máxima:   {metrics['max_memory_mb']:.1f} MB")
    
    # Comparar latentes si hay pares
    if c_bins and py_bins:
        print("\n" + "-"*60)
        print(" COMPARACIÓN DE SALIDAS")
        print("-"*60)
        
        # Tomar el par más reciente
        c_bin = c_bins[-1]
        py_bin = py_bins[-1]
        
        print(f"\nComparando:")
        print(f"  C:      {c_bin.name} ({c_bin.stat().st_size:,} bytes)")
        print(f"  Python: {py_bin.name} ({py_bin.stat().st_size:,} bytes)")
        
        data_c = read_compressed_bin(c_bin)
        data_py = read_compressed_bin(py_bin)
        
        # Verificar headers
        print(f"\nHeaders:")
        print(f"  C:      {data_c['header']}")
        print(f"  Python: {data_py['header']}")
        
        headers_match = data_c['header'] == data_py['header']
        print(f"  Match:  {'✅' if headers_match else '❌'}")
        
        # Comparar latentes
        compare_latents(data_c, data_py)
    
    # Calcular speedup si hay métricas de ambos
    if c_logs and py_logs:
        c_metrics = parse_benchmark_log(c_logs[-1])
        py_metrics = parse_benchmark_log(py_logs[-1])
        
        if 'elapsed_time' in c_metrics and 'elapsed_time' in py_metrics:
            speedup = py_metrics['elapsed_time'] / c_metrics['elapsed_time']
            print("\n" + "="*60)
            print(" SPEEDUP C vs Python")
            print("="*60)
            print(f"  Python: {py_metrics['elapsed_time']:.2f} s")
            print(f"  C:      {c_metrics['elapsed_time']:.2f} s")
            print(f"  Speedup: {speedup:.2f}x")
            
            if 'max_memory_mb' in c_metrics and 'max_memory_mb' in py_metrics:
                mem_ratio = py_metrics['max_memory_mb'] / c_metrics['max_memory_mb']
                print(f"\n  Memoria Python: {py_metrics['max_memory_mb']:.1f} MB")
                print(f"  Memoria C:      {c_metrics['max_memory_mb']:.1f} MB")
                print(f"  Ratio:          {mem_ratio:.2f}x")


def main():
    if len(sys.argv) < 2:
        print("Uso: python analyze_benchmark_results.py <directorio_resultados> [imagen_original.raw]")
        print("\nEjemplo:")
        print("  python analyze_benchmark_results.py benchmark_results/")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    original_image = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyze_results(results_dir, original_image)


if __name__ == "__main__":
    main()
