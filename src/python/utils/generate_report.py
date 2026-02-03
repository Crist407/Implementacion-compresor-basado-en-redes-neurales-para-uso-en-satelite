#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SORTENY - Generador de informes visuales de análisis.
"""

import os
import sys
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Configuración de matplotlib para mejor calidad
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Parámetros del modelo
BANDS = 8
H_IN, W_IN = 512, 512
NUM_FILTERS = 384
H_LATENT, W_LATENT = 32, 32

# Nombres de las bandas Sentinel-2 (aproximadas para 8 bandas)
BAND_NAMES = ['B2 (Blue)', 'B3 (Green)', 'B4 (Red)', 'B5 (RE1)', 
              'B6 (RE2)', 'B7 (RE3)', 'B8 (NIR)', 'B8A (NIR2)']


def read_raw_image(filepath, bands=8, height=512, width=512):
    """Lee imagen RAW uint16 en formato BSQ."""
    data = np.fromfile(filepath, dtype=np.uint16)
    return data.reshape(bands, height, width)


def read_compressed_bin(filepath):
    """Lee archivo .bin comprimido."""
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
        
    return {
        'header': {'bands': bands, 'height': height, 'width': width, 
                   'datatype': datatype, 'num_filters': num_filters},
        'qmap': qmap,
        'latents': latents.reshape(bands, num_filters, height//16, width//16)
    }


def parse_benchmark_log(filepath):
    """Extrae métricas de un log de benchmark."""
    metrics = {}
    if not os.path.exists(filepath):
        return metrics
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    patterns = {
        'user_time': r'User time \(seconds\): ([\d.]+)',
        'system_time': r'System time \(seconds\): ([\d.]+)',
        'elapsed_time': r'Elapsed \(wall clock\) time.*: ([\d:]+\.?\d*)',
        'max_memory_kb': r'Maximum resident set size \(kbytes\): (\d+)',
        'cpu_percent': r'Percent of CPU this job got: (\d+)%',
        'inference_time': r'\[BENCHMARK\] Tiempo Inferencia.*: ([\d.]+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            value = match.group(1)
            if key == 'elapsed_time' and ':' in value:
                parts = value.split(':')
                if len(parts) == 2:
                    value = float(parts[0]) * 60 + float(parts[1])
                elif len(parts) == 3:
                    value = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            metrics[key] = float(value) if '.' in str(value) or key != 'max_memory_kb' else int(value)
    
    if 'max_memory_kb' in metrics:
        metrics['max_memory_mb'] = metrics['max_memory_kb'] / 1024
    
    return metrics


def calculate_metrics(original, reconstructed):
    """Calcula métricas de calidad por banda y global."""
    metrics = {'per_band': [], 'global': {}}
    
    orig = original.astype(np.float64)
    recon = reconstructed.astype(np.float64)
    max_val = 65535.0
    
    for b in range(orig.shape[0]):
        mse = np.mean((orig[b] - recon[b]) ** 2)
        psnr = 10 * np.log10((max_val ** 2) / mse) if mse > 0 else float('inf')
        mae = np.mean(np.abs(orig[b] - recon[b]))
        metrics['per_band'].append({'band': b, 'mse': mse, 'psnr': psnr, 'mae': mae})
    
    global_mse = np.mean((orig - recon) ** 2)
    global_psnr = 10 * np.log10((max_val ** 2) / global_mse) if global_mse > 0 else float('inf')
    global_mae = np.mean(np.abs(orig - recon))
    
    metrics['global'] = {'mse': global_mse, 'psnr': global_psnr, 'mae': global_mae}
    
    return metrics


# =============================================================================
# FIGURAS
# =============================================================================

def fig_bandas_comparacion(original, recon_c, recon_py, output_dir):
    """Figura 1: Comparación visual de las 8 bandas (Original vs C vs Python)."""
    fig, axes = plt.subplots(3, 8, figsize=(20, 8))
    
    titles = ['Original', 'Reconstruida (C)', 'Reconstruida (Python)']
    images = [original, recon_c, recon_py]
    
    for row, (title, img) in enumerate(zip(titles, images)):
        for b in range(8):
            ax = axes[row, b]
            vmin, vmax = np.percentile(img[b], (1, 99))
            ax.imshow(img[b], cmap='gray', vmin=vmin, vmax=vmax)
            if row == 0:
                ax.set_title(f'B{b+1}', fontsize=10)
            if b == 0:
                ax.set_ylabel(title, fontsize=11, fontweight='bold')
            ax.axis('off')
    
    plt.suptitle('Comparación Visual por Banda: Original vs Reconstrucciones', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_bandas_comparacion.png'), bbox_inches='tight')
    plt.close()
    print("  ✓ fig1_bandas_comparacion.png")


def fig_mapa_error(original, recon_c, recon_py, output_dir):
    """Figura 2: Mapas de error absoluto por banda."""
    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    
    error_c = np.abs(original.astype(np.float32) - recon_c.astype(np.float32))
    error_py = np.abs(original.astype(np.float32) - recon_py.astype(np.float32))
    
    # Encontrar el máximo error para escala común
    vmax = max(np.percentile(error_c, 99), np.percentile(error_py, 99))
    
    for b in range(8):
        # Fila 1: Error C
        ax1 = axes[0, b]
        im1 = ax1.imshow(error_c[b], cmap='hot', vmin=0, vmax=vmax)
        ax1.set_title(f'B{b+1}', fontsize=10)
        if b == 0:
            ax1.set_ylabel('Error C', fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        # Fila 2: Error Python
        ax2 = axes[1, b]
        im2 = ax2.imshow(error_py[b], cmap='hot', vmin=0, vmax=vmax)
        if b == 0:
            ax2.set_ylabel('Error Python', fontsize=11, fontweight='bold')
        ax2.axis('off')
    
    # Colorbar
    cbar = fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Error Absoluto (niveles)', fontsize=10)
    
    plt.suptitle('Mapas de Error Absoluto: |Original - Reconstruida|', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'fig2_mapa_error.png'), bbox_inches='tight')
    plt.close()
    print("  ✓ fig2_mapa_error.png")


def fig_diferencia_c_vs_py(recon_c, recon_py, output_dir):
    """Figura 3: Diferencia entre reconstrucciones C y Python."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    diff = recon_c.astype(np.float32) - recon_py.astype(np.float32)
    vmax = max(abs(np.percentile(diff, 1)), abs(np.percentile(diff, 99)))
    
    for b in range(8):
        ax = axes[b//4, b%4]
        im = ax.imshow(diff[b], cmap='RdBu', vmin=-vmax, vmax=vmax)
        ax.set_title(f'{BAND_NAMES[b]}\nμ={np.mean(diff[b]):.3f}, σ={np.std(diff[b]):.3f}', fontsize=9)
        ax.axis('off')
    
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Diferencia (C - Python)', fontsize=10)
    
    plt.suptitle('Diferencia entre Reconstrucciones: C vs Python', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'fig3_diferencia_c_vs_py.png'), bbox_inches='tight')
    plt.close()
    print("  ✓ fig3_diferencia_c_vs_py.png")


def fig_metricas_calidad(metrics_c, metrics_py, output_dir):
    """Figura 4: Gráficas de métricas de calidad por banda."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    bands = [f'B{i+1}' for i in range(8)]
    x = np.arange(len(bands))
    width = 0.35
    
    # PSNR
    ax1 = axes[0]
    psnr_c = [m['psnr'] for m in metrics_c['per_band']]
    psnr_py = [m['psnr'] for m in metrics_py['per_band']]
    bars1 = ax1.bar(x - width/2, psnr_c, width, label='C', color='#2196F3')
    bars2 = ax1.bar(x + width/2, psnr_py, width, label='Python', color='#4CAF50')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_xlabel('Banda')
    ax1.set_title('PSNR por Banda')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bands)
    ax1.legend()
    ax1.axhline(y=metrics_c['global']['psnr'], color='#2196F3', linestyle='--', alpha=0.5, label='Global C')
    ax1.axhline(y=metrics_py['global']['psnr'], color='#4CAF50', linestyle='--', alpha=0.5, label='Global Py')
    ax1.set_ylim([min(psnr_c + psnr_py) - 2, max(psnr_c + psnr_py) + 2])
    
    # MSE
    ax2 = axes[1]
    mse_c = [m['mse'] for m in metrics_c['per_band']]
    mse_py = [m['mse'] for m in metrics_py['per_band']]
    ax2.bar(x - width/2, mse_c, width, label='C', color='#2196F3')
    ax2.bar(x + width/2, mse_py, width, label='Python', color='#4CAF50')
    ax2.set_ylabel('MSE')
    ax2.set_xlabel('Banda')
    ax2.set_title('MSE por Banda')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bands)
    ax2.legend()
    
    # MAE
    ax3 = axes[2]
    mae_c = [m['mae'] for m in metrics_c['per_band']]
    mae_py = [m['mae'] for m in metrics_py['per_band']]
    ax3.bar(x - width/2, mae_c, width, label='C', color='#2196F3')
    ax3.bar(x + width/2, mae_py, width, label='Python', color='#4CAF50')
    ax3.set_ylabel('MAE')
    ax3.set_xlabel('Banda')
    ax3.set_title('MAE por Banda')
    ax3.set_xticks(x)
    ax3.set_xticklabels(bands)
    ax3.legend()
    
    plt.suptitle('Métricas de Calidad: Original vs Reconstrucciones', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_metricas_calidad.png'), bbox_inches='tight')
    plt.close()
    print("  ✓ fig4_metricas_calidad.png")


def fig_rendimiento(metrics_c, metrics_py, output_dir):
    """Figura 5: Comparación de rendimiento (tiempo y memoria)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Colores
    colors = {'C': '#2196F3', 'Python': '#4CAF50'}
    
    # Gráfica 1: Tiempo de ejecución
    ax1 = axes[0]
    times = [metrics_c.get('elapsed_time', 0), metrics_py.get('elapsed_time', 0)]
    bars = ax1.bar(['C', 'Python'], times, color=[colors['C'], colors['Python']], edgecolor='black')
    ax1.set_ylabel('Tiempo (segundos)')
    ax1.set_title('Tiempo de Ejecución Total')
    for bar, t in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02, 
                f'{t:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Añadir speedup
    if times[1] > 0 and times[0] > 0:
        speedup = times[1] / times[0]
        ax1.text(0.5, 0.5, f'Speedup:\n{speedup:.2f}x', 
                ha='center', va='center', fontsize=14, fontweight='bold', color='red',
                transform=ax1.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Gráfica 2: Memoria máxima
    ax2 = axes[1]
    memory = [metrics_c.get('max_memory_mb', 0), metrics_py.get('max_memory_mb', 0)]
    bars = ax2.bar(['C', 'Python'], memory, color=[colors['C'], colors['Python']], edgecolor='black')
    ax2.set_ylabel('Memoria (MB)')
    ax2.set_title('Memoria RAM Máxima')
    for bar, m in zip(bars, memory):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(memory)*0.02, 
                f'{m:.0f} MB', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Añadir ratio
    if memory[0] > 0:
        ratio = memory[1] / memory[0]
        ax2.text(0.5, 0.5, f'Reducción:\n{ratio:.1f}x', 
                ha='center', va='center', fontsize=14, fontweight='bold', color='red',
                transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Gráfica 3: % CPU (limitado a valores razonables)
    ax3 = axes[2]
    cpu_c = min(metrics_c.get('cpu_percent', 0), 500)  # Limitar a 500%
    cpu_py = min(metrics_py.get('cpu_percent', 0), 500)
    cpu = [cpu_c, cpu_py]
    bars = ax3.bar(['C', 'Python'], cpu, color=[colors['C'], colors['Python']], edgecolor='black')
    ax3.set_ylabel('% CPU')
    ax3.set_title('Uso de CPU')
    ax3.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='1 core')
    ax3.axhline(y=400, color='gray', linestyle=':', alpha=0.5, label='4 cores')
    ax3.set_ylim(0, max(cpu) * 1.2 if max(cpu) > 0 else 100)
    for bar, c in zip(bars, cpu):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cpu)*0.02, 
                f'{c:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right')
    
    plt.suptitle('Comparación de Rendimiento: C vs Python (Raspberry Pi 3B+)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_rendimiento.png'), bbox_inches='tight')
    plt.close()
    print("  ✓ fig5_rendimiento.png")


def fig_histograma_latentes(latents_c, latents_py, output_dir):
    """Figura 6: Histograma de diferencias en latentes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    diff = latents_c.flatten().astype(np.int64) - latents_py.flatten().astype(np.int64)
    
    # Histograma de diferencias
    ax1 = axes[0]
    bins = np.arange(diff.min() - 0.5, diff.max() + 1.5, 1)
    ax1.hist(diff, bins=bins, color='#FF5722', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Diferencia (C - Python)')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Histograma de Diferencias en Latentes')
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    # Estadísticas
    num_diff = np.sum(diff != 0)
    pct_diff = 100 * num_diff / diff.size
    ax1.text(0.95, 0.95, f'Diferentes: {num_diff:,}\n({pct_diff:.4f}%)', 
            transform=ax1.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Histograma de latentes C
    ax2 = axes[1]
    ax2.hist(latents_c.flatten(), bins=100, color='#2196F3', alpha=0.7, label='C')
    ax2.hist(latents_py.flatten(), bins=100, color='#4CAF50', alpha=0.5, label='Python')
    ax2.set_xlabel('Valor del Latente')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución de Latentes')
    ax2.legend()
    ax2.set_yscale('log')
    
    # Scatter plot de primeros N latentes
    ax3 = axes[2]
    n_points = min(10000, latents_c.size)
    idx = np.random.choice(latents_c.size, n_points, replace=False)
    ax3.scatter(latents_c.flatten()[idx], latents_py.flatten()[idx], 
               alpha=0.1, s=1, c='#9C27B0')
    lims = [min(latents_c.min(), latents_py.min()), max(latents_c.max(), latents_py.max())]
    ax3.plot(lims, lims, 'r--', alpha=0.8, label='y=x')
    ax3.set_xlabel('Latentes C')
    ax3.set_ylabel('Latentes Python')
    ax3.set_title('Correlación de Latentes (muestreo)')
    ax3.legend()
    ax3.set_aspect('equal')
    
    plt.suptitle('Análisis de Latentes: C vs Python', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_histograma_latentes.png'), bbox_inches='tight')
    plt.close()
    print("  ✓ fig6_histograma_latentes.png")


def fig_resumen_tabla(metrics_c, metrics_py, bench_c, bench_py, output_dir):
    """Figura 7: Tabla resumen con todas las métricas."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Datos para la tabla
    data = [
        ['RENDIMIENTO', '', ''],
        ['Tiempo total (s)', f"{bench_c.get('elapsed_time', 'N/A'):.2f}", f"{bench_py.get('elapsed_time', 'N/A'):.2f}"],
        ['Memoria máxima (MB)', f"{bench_c.get('max_memory_mb', 'N/A'):.1f}", f"{bench_py.get('max_memory_mb', 'N/A'):.1f}"],
        ['Uso CPU (%)', f"{bench_c.get('cpu_percent', 'N/A'):.0f}", f"{bench_py.get('cpu_percent', 'N/A'):.0f}"],
        ['', '', ''],
        ['CALIDAD (vs Original)', '', ''],
        ['PSNR global (dB)', f"{metrics_c['global']['psnr']:.2f}", f"{metrics_py['global']['psnr']:.2f}"],
        ['MSE global', f"{metrics_c['global']['mse']:.2f}", f"{metrics_py['global']['mse']:.2f}"],
        ['MAE global', f"{metrics_c['global']['mae']:.2f}", f"{metrics_py['global']['mae']:.2f}"],
    ]
    
    columns = ['Métrica', 'C', 'Python']
    
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Estilo
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Filas de sección
    for row in [1, 6]:
        for col in range(len(columns)):
            table[(row, col)].set_facecolor('#E3F2FD')
            table[(row, col)].set_text_props(fontweight='bold')
    
    plt.title('Resumen de Métricas: SORTENY C vs Python\n(Raspberry Pi 3B+)', 
             fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'fig7_resumen_tabla.png'), bbox_inches='tight')
    plt.close()
    print("  ✓ fig7_resumen_tabla.png")


def fig_detalle_banda(original, recon_c, recon_py, band_idx, output_dir):
    """Figura 8: Análisis detallado de una banda específica."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 0.8])
    
    band = band_idx
    orig = original[band]
    rec_c = recon_c[band]
    rec_py = recon_py[band]
    
    vmin, vmax = np.percentile(orig, (1, 99))
    
    # Fila 1: Imágenes
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig, cmap='gray', vmin=vmin, vmax=vmax)
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(rec_c, cmap='gray', vmin=vmin, vmax=vmax)
    ax2.set_title('Reconstruida (C)')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(rec_py, cmap='gray', vmin=vmin, vmax=vmax)
    ax3.set_title('Reconstruida (Python)')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    diff_c_py = rec_c.astype(np.float32) - rec_py.astype(np.float32)
    vmax_diff = max(abs(diff_c_py.min()), abs(diff_c_py.max()))
    im = ax4.imshow(diff_c_py, cmap='RdBu', vmin=-vmax_diff, vmax=vmax_diff)
    ax4.set_title('Diferencia C - Python')
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, fraction=0.046)
    
    # Fila 2: Mapas de error
    error_c = np.abs(orig.astype(np.float32) - rec_c.astype(np.float32))
    error_py = np.abs(orig.astype(np.float32) - rec_py.astype(np.float32))
    vmax_err = max(np.percentile(error_c, 99), np.percentile(error_py, 99))
    
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(error_c, cmap='hot', vmin=0, vmax=vmax_err)
    ax5.set_title(f'Error C (MAE={np.mean(error_c):.2f})')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(error_py, cmap='hot', vmin=0, vmax=vmax_err)
    ax6.set_title(f'Error Python (MAE={np.mean(error_py):.2f})')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    # Histogramas
    ax7 = fig.add_subplot(gs[1, 2:])
    ax7.hist(error_c.flatten(), bins=50, alpha=0.7, label='Error C', color='#2196F3')
    ax7.hist(error_py.flatten(), bins=50, alpha=0.7, label='Error Python', color='#4CAF50')
    ax7.set_xlabel('Error Absoluto')
    ax7.set_ylabel('Frecuencia')
    ax7.set_title('Distribución del Error')
    ax7.legend()
    ax7.set_yscale('log')
    
    # Fila 3: Perfiles
    ax8 = fig.add_subplot(gs[2, :2])
    row = H_IN // 2
    ax8.plot(orig[row, :], label='Original', linewidth=1)
    ax8.plot(rec_c[row, :], label='C', linewidth=1, alpha=0.8)
    ax8.plot(rec_py[row, :], label='Python', linewidth=1, alpha=0.8)
    ax8.set_xlabel('Columna')
    ax8.set_ylabel('Valor')
    ax8.set_title(f'Perfil Horizontal (fila {row})')
    ax8.legend()
    
    ax9 = fig.add_subplot(gs[2, 2:])
    col = W_IN // 2
    ax9.plot(orig[:, col], label='Original', linewidth=1)
    ax9.plot(rec_c[:, col], label='C', linewidth=1, alpha=0.8)
    ax9.plot(rec_py[:, col], label='Python', linewidth=1, alpha=0.8)
    ax9.set_xlabel('Fila')
    ax9.set_ylabel('Valor')
    ax9.set_title(f'Perfil Vertical (columna {col})')
    ax9.legend()
    
    plt.suptitle(f'Análisis Detallado: {BAND_NAMES[band]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig8_detalle_banda{band+1}.png'), bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig8_detalle_banda{band+1}.png")


def generate_report(results_dir, original_path, output_dir):
    """Genera el informe completo."""
    
    print("=" * 60)
    print(" GENERADOR DE INFORME SORTENY")
    print("=" * 60)
    
    results_path = Path(results_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Buscar archivos
    c_bins = sorted(results_path.glob("output_c_*.bin"))
    py_bins = sorted(results_path.glob("output_py_*.bin"))
    c_logs = sorted(results_path.glob("benchmark_c_*.log"))
    py_logs = sorted(results_path.glob("benchmark_py_*.log"))
    recon_c_files = sorted(results_path.glob("reconstructed_from_c*.raw"))
    recon_py_files = sorted(results_path.glob("reconstructed_from_py*.raw"))
    
    print(f"\nArchivos encontrados en {results_dir}:")
    print(f"  - Binarios C: {len(c_bins)}")
    print(f"  - Binarios Python: {len(py_bins)}")
    print(f"  - Logs C: {len(c_logs)}")
    print(f"  - Logs Python: {len(py_logs)}")
    print(f"  - Reconstruidas C: {len(recon_c_files)}")
    print(f"  - Reconstruidas Py: {len(recon_py_files)}")
    
    # Cargar datos
    print("\nCargando datos...")
    
    original = read_raw_image(original_path)
    print(f"  ✓ Original: {original.shape}")
    
    recon_c = read_raw_image(recon_c_files[-1]) if recon_c_files else None
    recon_py = read_raw_image(recon_py_files[-1]) if recon_py_files else None
    
    if recon_c is not None:
        print(f"  ✓ Reconstruida C: {recon_c.shape}")
    if recon_py is not None:
        print(f"  ✓ Reconstruida Py: {recon_py.shape}")
    
    data_c = read_compressed_bin(c_bins[-1]) if c_bins else None
    data_py = read_compressed_bin(py_bins[-1]) if py_bins else None
    
    bench_c = parse_benchmark_log(c_logs[-1]) if c_logs else {}
    bench_py = parse_benchmark_log(py_logs[-1]) if py_logs else {}
    
    # Calcular métricas
    print("\nCalculando métricas...")
    metrics_c = calculate_metrics(original, recon_c) if recon_c is not None else None
    metrics_py = calculate_metrics(original, recon_py) if recon_py is not None else None
    
    # Generar figuras
    print(f"\nGenerando figuras en {output_dir}/...")
    
    if recon_c is not None and recon_py is not None:
        fig_bandas_comparacion(original, recon_c, recon_py, output_dir)
        fig_mapa_error(original, recon_c, recon_py, output_dir)
        fig_diferencia_c_vs_py(recon_c, recon_py, output_dir)
    
    if metrics_c and metrics_py:
        fig_metricas_calidad(metrics_c, metrics_py, output_dir)
    
    if bench_c and bench_py:
        fig_rendimiento(bench_c, bench_py, output_dir)
    
    if data_c and data_py:
        fig_histograma_latentes(data_c['latents'], data_py['latents'], output_dir)
    
    if metrics_c and metrics_py and bench_c and bench_py:
        fig_resumen_tabla(metrics_c, metrics_py, bench_c, bench_py, output_dir)
    
    # Generar análisis detallado para bandas seleccionadas
    if recon_c is not None and recon_py is not None:
        for band in [0, 2, 6]:  # Blue, Red, NIR
            fig_detalle_banda(original, recon_c, recon_py, band, output_dir)
    
    print("\n" + "=" * 60)
    print(f" ✅ Informe generado en: {output_dir}/")
    print("=" * 60)
    print("\nFiguras generadas:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.png'):
            print(f"  - {f}")


def main():
    parser = argparse.ArgumentParser(description='Genera informe visual de resultados SORTENY')
    parser.add_argument('results_dir', help='Directorio con resultados del benchmark')
    parser.add_argument('--original', '-o', required=True, help='Imagen original .raw')
    parser.add_argument('--output_dir', '-d', default='figuras_informe', help='Directorio de salida')
    
    args = parser.parse_args()
    generate_report(args.results_dir, args.original, args.output_dir)


if __name__ == "__main__":
    main()
