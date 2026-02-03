#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descompresión y comparación de resultados SORTENY
Ejecutar en PC local con TensorFlow instalado

Uso:
    python decompress_and_compare.py <archivo.bin> [imagen_original.raw] [--model_dir models/ieec050]
"""

import os
import sys
import argparse
import numpy as np

# Intentar importar TensorFlow
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    print("⚠️  TensorFlow no disponible. Solo análisis de latentes.")
    HAS_TF = False

# Parámetros del modelo
MIN_LAMBDA = 0.0
MAX_LAMBDA = 0.125


def read_compressed_bin(filepath):
    """Lee un archivo .bin comprimido."""
    with open(filepath, 'rb') as f:
        header = np.fromfile(f, dtype=np.uint16, count=5)
        bands = int(header[0])
        height = int(header[1])
        width = int(header[2])
        datatype = int(header[3])
        num_filters = int(header[4])
        
        qmap_size = (height // 16) * (width // 16)
        qmap = np.fromfile(f, dtype=np.uint8, count=qmap_size)
        
        latents = np.fromfile(f, dtype=np.int32)
        
    info = {
        'bands': bands,
        'height': height,
        'width': width,
        'datatype': datatype,
        'num_filters': num_filters,
        'qmap': qmap,
        'latents': latents
    }
    
    print(f"Archivo: {filepath}")
    print(f"  Header: bands={bands}, height={height}, width={width}, dtype={datatype}, filters={num_filters}")
    print(f"  Q[0]: {qmap[0]}")
    print(f"  Latentes: {latents.shape} ({latents.size} elementos)")
    print(f"  Rango latentes: [{latents.min()}, {latents.max()}]")
    
    return info


def decompress(data, model_dir):
    """Descomprime latentes usando el modelo TensorFlow."""
    if not HAS_TF:
        print("ERROR: TensorFlow requerido para descompresión")
        return None
    
    bands = data['bands']
    height = data['height']
    width = data['width']
    num_filters = data['num_filters']
    qmap = data['qmap']
    latents = data['latents']
    
    # Cargar modelos
    print(f"\nCargando modelos desde {model_dir}...")
    synthesis = tf.saved_model.load(os.path.join(model_dir, 'synthesis'))
    modulating = tf.saved_model.load(os.path.join(model_dir, 'modulating'))
    spectral = tf.saved_model.load(os.path.join(model_dir, 'spectral'))
    
    # Reshape Q-map y calcular lambda
    Q = tf.convert_to_tensor(np.reshape(qmap, (1, height//16, width//16, 1)), dtype=tf.float32)
    q = (Q / 255.0) * (MAX_LAMBDA - MIN_LAMBDA) + MIN_LAMBDA
    
    # Calcular modulador inverso
    mod = modulating.signatures['serving_default'](q)['dense_2']
    
    # Reshape latentes: (bands * num_filters * h_lat * w_lat) -> (1, h_lat, w_lat, bands*num_filters)
    h_lat = height // 16
    w_lat = width // 16
    y_hat_planar = np.reshape(latents, (1, bands * num_filters, h_lat, w_lat))
    y_hat_3D = tf.convert_to_tensor(np.transpose(y_hat_planar, (0, 2, 3, 1)), dtype=tf.float32)
    
    print(f"  Latentes reshape: {y_hat_3D.shape}")
    
    # Demodular
    y_hat_3D = y_hat_3D / mod
    y_hat_3D = tf.transpose(y_hat_3D, (0, 3, 1, 2))
    y_hat_3D = tf.reshape(y_hat_3D, (bands, num_filters, h_lat, w_lat))
    y_hat = tf.transpose(y_hat_3D, (0, 2, 3, 1))
    
    # Síntesis
    print("  Ejecutando síntesis...")
    x_hat_1D = synthesis.signatures['serving_default'](y_hat)['lambda_5']
    
    # Transformación espectral inversa
    I = tf.expand_dims(tf.expand_dims(tf.eye(bands), axis=1), axis=1)
    A = tf.squeeze(spectral.signatures['serving_default'](I)['dense'])
    B = tf.linalg.inv(A)
    
    x_hat_1D = tf.transpose(x_hat_1D, (0, 3, 1, 2))
    x_hat_1D = tf.reshape(x_hat_1D, (1, bands, height, width))
    x_hat_1D = tf.transpose(x_hat_1D, (0, 2, 3, 1))
    x_hat = tf.linalg.matvec(tf.linalg.matrix_transpose(B), x_hat_1D)
    
    # Convertir a uint16
    x_hat = tf.saturate_cast(tf.round(x_hat), tf.uint16)
    
    print(f"  Imagen reconstruida: {x_hat.shape}")
    
    return x_hat.numpy()


def read_raw_image(filepath, bands, height, width, dtype='uint16'):
    """Lee imagen RAW original."""
    dt = np.uint16 if dtype == 'uint16' else np.float32
    data = np.fromfile(filepath, dtype=dt)
    # Formato BSQ: (bands, height, width) -> (1, height, width, bands)
    img = np.reshape(data, (bands, height, width))
    img = np.transpose(img, (1, 2, 0))
    return img[np.newaxis, ...]


def calculate_metrics(original, reconstructed):
    """Calcula métricas de calidad."""
    orig = original.astype(np.float64)
    recon = reconstructed.astype(np.float64)
    
    # MSE
    mse = np.mean((orig - recon) ** 2)
    
    # PSNR (para uint16, max_val = 65535)
    max_val = 65535.0
    psnr = 10 * np.log10((max_val ** 2) / mse) if mse > 0 else float('inf')
    
    # MAE
    mae = np.mean(np.abs(orig - recon))
    
    # Per-band metrics
    print("\nMétricas por banda:")
    for b in range(orig.shape[-1]):
        band_mse = np.mean((orig[..., b] - recon[..., b]) ** 2)
        band_psnr = 10 * np.log10((max_val ** 2) / band_mse) if band_mse > 0 else float('inf')
        print(f"  Banda {b}: MSE={band_mse:.2f}, PSNR={band_psnr:.2f} dB")
    
    return {'mse': mse, 'psnr': psnr, 'mae': mae}


def main():
    parser = argparse.ArgumentParser(description='Descomprime y compara resultados SORTENY')
    parser.add_argument('compressed_file', help='Archivo .bin comprimido')
    parser.add_argument('--original', '-o', help='Imagen original .raw para comparación')
    parser.add_argument('--model_dir', '-m', default='models/ieec050', help='Directorio del modelo')
    parser.add_argument('--output', help='Archivo de salida para imagen reconstruida')
    parser.add_argument('--compare', '-c', help='Segundo archivo .bin para comparar latentes')
    
    args = parser.parse_args()
    
    # Leer archivo comprimido
    print("="*60)
    print(" ANÁLISIS DE ARCHIVO COMPRIMIDO")
    print("="*60)
    data = read_compressed_bin(args.compressed_file)
    
    # Comparar con otro archivo si se especifica
    if args.compare:
        print("\n" + "="*60)
        print(" COMPARACIÓN CON SEGUNDO ARCHIVO")
        print("="*60)
        data2 = read_compressed_bin(args.compare)
        
        lat1 = data['latents']
        lat2 = data2['latents']
        
        if lat1.size == lat2.size:
            diff = lat1.astype(np.int64) - lat2.astype(np.int64)
            num_diff = np.sum(diff != 0)
            print(f"\nDiferencias: {num_diff:,} / {lat1.size:,} ({100*num_diff/lat1.size:.4f}%)")
            if num_diff > 0:
                print(f"  Diff media: {np.mean(np.abs(diff)):.6f}")
                print(f"  Diff máx:   {np.max(np.abs(diff))}")
        else:
            print(f"⚠️  Tamaños diferentes: {lat1.size} vs {lat2.size}")
    
    # Descomprimir si hay TensorFlow y modelo
    if HAS_TF and os.path.isdir(args.model_dir):
        print("\n" + "="*60)
        print(" DESCOMPRESIÓN")
        print("="*60)
        reconstructed = decompress(data, args.model_dir)
        
        if reconstructed is not None:
            # Guardar si se especifica
            if args.output:
                recon_bsq = np.transpose(reconstructed[0], (2, 0, 1))  # (H,W,B) -> (B,H,W)
                recon_bsq.astype(np.uint16).tofile(args.output)
                print(f"\nImagen reconstruida guardada: {args.output}")
            
            # Comparar con original si se especifica
            if args.original and os.path.exists(args.original):
                print("\n" + "="*60)
                print(" MÉTRICAS DE CALIDAD")
                print("="*60)
                original = read_raw_image(args.original, data['bands'], data['height'], data['width'])
                metrics = calculate_metrics(original, reconstructed)
                print(f"\nMétricas globales:")
                print(f"  MSE:  {metrics['mse']:.2f}")
                print(f"  PSNR: {metrics['psnr']:.2f} dB")
                print(f"  MAE:  {metrics['mae']:.2f}")
    else:
        if not HAS_TF:
            print("\n⚠️  Instala TensorFlow para descompresión: pip install tensorflow")
        if not os.path.isdir(args.model_dir):
            print(f"\n⚠️  Modelo no encontrado: {args.model_dir}")


if __name__ == "__main__":
    main()
