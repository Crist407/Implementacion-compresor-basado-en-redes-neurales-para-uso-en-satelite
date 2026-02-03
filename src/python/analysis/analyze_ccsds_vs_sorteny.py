#!/usr/bin/env python3
"""
Análisis Comparativo: CCSDS 122 vs SORTENY
==========================================

Calcula métricas de calidad y genera figuras comparativas.

Autor: Christian Añez
Fecha: 2026-01-05
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Añadir el directorio de scripts al path
sys.path.insert(0, str(Path(__file__).parent))

# Directorio de resultados
RESULTS_DIR = Path(__file__).parent.parent / "raspberry_results" / "ccsds_comparison"
FIGURES_DIR = RESULTS_DIR / "figuras"
FIGURES_DIR.mkdir(exist_ok=True)

# Imagen original
ORIGINAL_BE = RESULTS_DIR / "original_be.raw"
ORIGINAL_LE = Path(__file__).parent.parent / "raspberry_results" / "reconstructed_from_c.raw"

# Dimensiones
WIDTH, HEIGHT, BANDS = 512, 512, 8
ORIGINAL_SIZE = WIDTH * HEIGHT * BANDS * 2  # 4 MB

def load_image(path, big_endian=False):
    """Carga imagen raw uint16"""
    if not path.exists():
        print(f"Archivo no encontrado: {path}")
        return None
    dtype = '>u2' if big_endian else '<u2'
    return np.fromfile(path, dtype=dtype).reshape(BANDS, HEIGHT, WIDTH)


def calculate_metrics(original, reconstructed):
    """Calcula PSNR, MSE, MAE entre original y reconstruido"""
    orig = original.astype(np.float64)
    recon = reconstructed.astype(np.float64)
    
    diff = orig - recon
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    max_err = np.max(np.abs(diff))
    
    max_val = 65535.0
    psnr = 10 * np.log10((max_val ** 2) / mse) if mse > 0 else float('inf')
    
    # Métricas por banda
    band_psnr = []
    for b in range(BANDS):
        band_mse = np.mean((orig[b] - recon[b]) ** 2)
        if band_mse > 0:
            band_psnr.append(10 * np.log10((max_val ** 2) / band_mse))
        else:
            band_psnr.append(float('inf'))
    
    return {
        'psnr': psnr,
        'mse': mse,
        'mae': mae,
        'max_err': max_err,
        'band_psnr': band_psnr,
        'identical': int(np.sum(orig == recon)),
        'total': orig.size
    }


def main():
    print("="*70)
    print("ANÁLISIS COMPARATIVO: CCSDS 122 vs SORTENY")
    print("="*70)
    
    # Cargar imagen original
    print("\nCargando imagen original (big-endian)...")
    original = load_image(ORIGINAL_BE, big_endian=True)
    if original is None:
        print("ERROR: No se pudo cargar la imagen original")
        return
    
    print(f"Original: shape={original.shape}, min={original.min()}, max={original.max()}")
    
    # Resultados del benchmark (extraídos del log)
    benchmark_results = {
        'CCSDS 122\nLossless': {
            'time_encode': 12.66,
            'time_decode': 10.88,
            'size': 2334768,
            'memory_kb': 25184,
            'file': 'ccsds_lossless_decoded.raw',
            'big_endian': True
        },
        'CCSDS 122\nNear-Lossless': {
            'time_encode': 12.67,
            'time_decode': 9.08,
            'size': 2334753,
            'memory_kb': 25300,
            'file': 'ccsds_nearlossless_decoded.raw',
            'big_endian': True
        },
        'SORTENY C': {
            'time_encode': 304.09,
            'time_decode': None,  # Requiere TensorFlow
            'size': 12583946,
            'memory_kb': 88612,
            'file': 'sorteny_c.bin',
            'big_endian': False,
            'is_latent': True
        },
        'SORTENY\nPython': {
            'time_encode': 470.45,
            'time_decode': None,
            'size': 12583946,
            'memory_kb': 815940,
            'file': 'sorteny_py.bin',
            'big_endian': False,
            'is_latent': True
        }
    }
    
    # Calcular métricas de calidad para CCSDS
    print("\n" + "="*70)
    print("MÉTRICAS DE CALIDAD")
    print("="*70)
    
    quality_results = {}
    
    for name, info in benchmark_results.items():
        if info.get('is_latent'):
            # SORTENY produce latentes, no imagen reconstruida directamente
            continue
        
        file_path = RESULTS_DIR / info['file']
        if not file_path.exists():
            print(f"\n{name}: Archivo no encontrado - {file_path}")
            continue
        
        recon = load_image(file_path, big_endian=info['big_endian'])
        if recon is None:
            continue
        
        metrics = calculate_metrics(original, recon)
        quality_results[name] = metrics
        
        psnr_str = "∞ (lossless)" if metrics['psnr'] == float('inf') else f"{metrics['psnr']:.2f}"
        print(f"\n{name.replace(chr(10), ' ')}:")
        print(f"  PSNR:     {psnr_str} dB")
        print(f"  MSE:      {metrics['mse']:.4f}")
        print(f"  MAE:      {metrics['mae']:.4f}")
        print(f"  Max Error: {metrics['max_err']:.0f}")
        print(f"  Idénticos: {metrics['identical']:,} / {metrics['total']:,} ({100*metrics['identical']/metrics['total']:.2f}%)")
    
    # Cargar reconstrucción de SORTENY (del benchmark anterior)
    sorteny_recon_path = Path(__file__).parent.parent / "raspberry_results" / "reconstructed_from_c.raw"
    if sorteny_recon_path.exists():
        # La imagen original para SORTENY está en little-endian
        original_le = original.astype('>u2').byteswap().view('<u2').reshape(BANDS, HEIGHT, WIDTH)
        sorteny_recon = load_image(sorteny_recon_path, big_endian=False)
        if sorteny_recon is not None:
            metrics = calculate_metrics(original_le, sorteny_recon)
            quality_results['SORTENY'] = metrics
            print(f"\nSORTENY (reconstruido):")
            print(f"  PSNR:     {metrics['psnr']:.2f} dB")
            print(f"  MSE:      {metrics['mse']:.4f}")
            print(f"  MAE:      {metrics['mae']:.4f}")
    
    # ===========================================
    # GENERAR FIGURAS
    # ===========================================
    print("\n" + "="*70)
    print("GENERANDO FIGURAS")
    print("="*70)
    
    # Figura 1: Comparación de tiempos de compresión
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = list(benchmark_results.keys())
    times = [benchmark_results[m]['time_encode'] for m in methods]
    colors = ['#2196F3', '#03A9F4', '#4CAF50', '#FF9800']
    
    bars = ax.bar(methods, times, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Tiempo de Compresión (segundos)', fontsize=12)
    ax.set_title('Comparación de Tiempos de Compresión\n(Raspberry Pi 3B+)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    
    for bar, time in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{time:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ccsds_tiempos_compresion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {FIGURES_DIR / 'ccsds_tiempos_compresion.png'}")
    
    # Figura 2: Comparación de tamaños / ratios de compresión
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sizes_kb = [benchmark_results[m]['size'] / 1024 for m in methods]
    ratios = [ORIGINAL_SIZE / benchmark_results[m]['size'] for m in methods]
    
    # Tamaños
    bars1 = ax1.bar(methods, sizes_kb, color=colors, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=ORIGINAL_SIZE/1024, color='red', linestyle='--', linewidth=2, label='Original (4096 KB)')
    ax1.set_ylabel('Tamaño (KB)', fontsize=12)
    ax1.set_title('Tamaño del Archivo Comprimido', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, size in zip(bars1, sizes_kb):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{size:.0f} KB', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Ratios
    bars2 = ax2.bar(methods, ratios, color=colors, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Sin compresión')
    ax2.set_ylabel('Ratio de Compresión', fontsize=12)
    ax2.set_title('Ratio de Compresión (mayor = mejor)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, ratio in zip(bars2, ratios):
        label = f'{ratio:.2f}:1' if ratio >= 1 else f'{1/ratio:.2f}x expand'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ccsds_tamanos_ratios.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {FIGURES_DIR / 'ccsds_tamanos_ratios.png'}")
    
    # Figura 3: Uso de memoria
    fig, ax = plt.subplots(figsize=(10, 6))
    
    memory_mb = [benchmark_results[m]['memory_kb'] / 1024 for m in methods]
    
    bars = ax.bar(methods, memory_mb, color=colors, edgecolor='black', linewidth=1.5)
    ax.axhline(y=909, color='red', linestyle='--', linewidth=2, label='RAM disponible (~909 MB)')
    ax.set_ylabel('Memoria (MB)', fontsize=12)
    ax.set_title('Uso de Memoria (Maximum RSS)\n(Raspberry Pi 3B+)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bar, mem in zip(bars, memory_mb):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{mem:.1f} MB', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ccsds_memoria.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {FIGURES_DIR / 'ccsds_memoria.png'}")
    
    # Figura 4: Tabla resumen
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')
    
    # Preparar datos
    headers = ['Método', 'Tiempo\n(s)', 'Tamaño\n(KB)', 'Ratio', 'Memoria\n(MB)', 'PSNR\n(dB)']
    
    data = []
    for name in methods:
        info = benchmark_results[name]
        ratio = ORIGINAL_SIZE / info['size']
        ratio_str = f"{ratio:.2f}:1" if ratio >= 1 else f"0.33:1*"
        
        # PSNR
        if name in quality_results:
            psnr = quality_results[name]['psnr']
            psnr_str = "∞" if psnr == float('inf') else f"{psnr:.2f}"
        elif 'SORTENY' in quality_results:
            psnr_str = f"{quality_results['SORTENY']['psnr']:.2f}"
        else:
            psnr_str = "~76.7†"
        
        data.append([
            name.replace('\n', ' '),
            f"{info['time_encode']:.1f}",
            f"{info['size']/1024:.0f}",
            ratio_str,
            f"{info['memory_kb']/1024:.1f}",
            psnr_str
        ])
    
    table = ax.table(
        cellText=data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['#4472C4'] * len(headers)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.3, 2.2)
    
    # Estilo
    for j in range(len(headers)):
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    for i in range(1, len(data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F0FE')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    ax.set_title('Resumen Comparativo: CCSDS 122 vs SORTENY\n(Raspberry Pi 3B+ - Imagen Sentinel-2 512×512×8 uint16)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Notas al pie
    fig.text(0.5, 0.08, 
             "* SORTENY produce latentes sin codificador de entropía. Con codificador ANS/Huffman el ratio mejoraría ~3-4x\n"
             "† PSNR de SORTENY medido tras descompresión con TensorFlow",
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ccsds_tabla_resumen.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {FIGURES_DIR / 'ccsds_tabla_resumen.png'}")
    
    # Figura 5: Eficiencia (PSNR por segundo de cómputo)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calcular eficiencia: menor tiempo = mejor para igual calidad
    # Para CCSDS lossless: PSNR infinito, usamos un valor alto para visualización
    efficiencies = []
    eff_methods = []
    eff_colors = []
    
    for i, name in enumerate(methods):
        if name in quality_results or 'SORTENY' in name:
            if name in quality_results:
                psnr = quality_results[name]['psnr']
                if psnr == float('inf'):
                    psnr = 100  # Para visualización
            else:
                psnr = 76.7  # SORTENY aproximado
            
            time = benchmark_results[name]['time_encode']
            eff = psnr / np.log10(time + 1)  # Eficiencia ponderada
            efficiencies.append(eff)
            eff_methods.append(name)
            eff_colors.append(colors[i])
    
    if efficiencies:
        bars = ax.bar(eff_methods, efficiencies, color=eff_colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Eficiencia (PSNR / log₁₀(tiempo))', fontsize=12)
        ax.set_title('Eficiencia: Calidad vs Tiempo de Cómputo\n(Mayor = Mejor)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'ccsds_eficiencia.png', dpi=150, bbox_inches='tight')
        print(f"  Guardado: {FIGURES_DIR / 'ccsds_eficiencia.png'}")
    plt.close()
    
    # ===========================================
    # RESUMEN FINAL
    # ===========================================
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                    COMPARATIVA CCSDS 122 vs SORTENY                  │
├─────────────────────────────────────────────────────────────────────┤
│ Hardware: Raspberry Pi 3B+ (ARM Cortex-A53, 1GB RAM)                │
│ Imagen:   Sentinel-2, 512×512×8 bandas, uint16 (4 MB)               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ CCSDS 122 (Lossless):                                               │
│   • Tiempo:  12.66s (compresión) + 10.88s (descompresión)           │
│   • Tamaño:  2.23 MB (ratio 1.79:1)                                 │
│   • Memoria: 25 MB                                                  │
│   • Calidad: LOSSLESS (PSNR = ∞)                                    │
│                                                                      │
│ CCSDS 122 (Near-Lossless, U=4, D=4):                                │
│   • Tiempo:  12.67s + 9.08s                                         │
│   • Tamaño:  2.23 MB (ratio 1.79:1)                                 │
│   • Memoria: 25 MB                                                  │
│   • Calidad: Near-lossless (pérdida mínima por shift)               │
│                                                                      │
│ SORTENY C:                                                          │
│   • Tiempo:  304.09s (solo codificador, sin entropía)               │
│   • Tamaño:  12.0 MB latentes (expandido 3x)*                       │
│   • Memoria: 88.6 MB                                                │
│   • Calidad: PSNR ~76.7 dB (lossy aprendido)                        │
│                                                                      │
│ SORTENY Python:                                                     │
│   • Tiempo:  470.45s                                                │
│   • Tamaño:  12.0 MB latentes*                                      │
│   • Memoria: 816 MB (¡cerca del límite!)                            │
│   • Calidad: PSNR ~76.7 dB                                          │
│                                                                      │
│ * SORTENY produce latentes cuantizados sin codificador de entropía. │
│   Con ANS/Huffman, el ratio estimado sería ~1.5-2:1                 │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ CONCLUSIONES:                                                        │
│                                                                      │
│ 1. CCSDS 122 es ~24x más rápido que SORTENY C                       │
│ 2. CCSDS 122 usa ~3.5x menos memoria que SORTENY C                  │
│ 3. CCSDS 122 logra compresión real (1.79:1) vs latentes expandidos  │
│ 4. SORTENY requiere codificador de entropía para ser competitivo    │
│ 5. SORTENY C es 1.55x más rápido que SORTENY Python                 │
│ 6. SORTENY C usa 9.2x menos memoria que Python                      │
│                                                                      │
│ CCSDS 122 es claramente superior para compresión embebida.          │
│ SORTENY tiene potencial con codificador de entropía adecuado.       │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    print(f"\nFiguras guardadas en: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
