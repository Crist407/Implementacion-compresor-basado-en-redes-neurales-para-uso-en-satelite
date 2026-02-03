#!/usr/bin/env python3
"""
Re-exportar los pesos del modelo ieec050/analysis aplicando IRDFT2D
para obtener los kernels espaciales efectivos.
"""
import tensorflow as tf
import numpy as np
import os

# Usar rutas relativas desde la raíz del proyecto
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
output_dir = os.path.join(ROOT, "weights", "pesos_ieec050_spatial")
os.makedirs(output_dir, exist_ok=True)

# Cargar modelos
analysis = tf.saved_model.load(os.path.join(ROOT, "models", "ieec050", "analysis"))
spectral = tf.saved_model.load(os.path.join(ROOT, "models", "ieec050", "spectral"))
modulating = tf.saved_model.load(os.path.join(ROOT, "models", "ieec050", "modulating"))

def rdft_to_spatial(kernel_real, kernel_imag, fft_length):
    """Convierte kernels de dominio RDFT a dominio espacial."""
    rdft = tf.dtypes.complex(kernel_real, kernel_imag)
    # El código de TFC usa norm = sqrt(kH * kW) donde kH, kW son las dimensiones espaciales
    # del kernel efectivo, NO las dimensiones del RDFT.
    # fft_length = [kH, kW] así que norm = sqrt(fft_length[0] * fft_length[1])
    norm = np.sqrt(fft_length[0] * fft_length[1]).astype(np.float32)
    rdft_scaled = rdft * norm
    kernel = tf.signal.irfft2d(rdft_scaled, fft_length=fft_length)
    # Transponer de (C_in, C_out, kH, kW) a (kH, kW, C_in, C_out)
    kernel = tf.transpose(kernel, (2, 3, 0, 1))
    return kernel.numpy()

# Extraer y convertir pesos de cada capa
layers_info = [
    ("layer_0", 1, 128, 5, 5),   # C_in=1, C_out=128, kH=5, kW=5
    ("layer_1", 128, 128, 5, 5),
    ("layer_2", 128, 128, 5, 5),
    ("layer_3", 128, 384, 5, 5),
]

weights_index = []

for layer_name, c_in, c_out, kh, kw in layers_info:
    kernel_real = None
    kernel_imag = None
    bias = None
    gdn_beta = None
    gdn_gamma = None
    
    for v in analysis.variables:
        if f'{layer_name}/kernel_real' in v.name:
            kernel_real = v.numpy()
        elif f'{layer_name}/kernel_imag' in v.name:
            kernel_imag = v.numpy()
        elif f'{layer_name}/bias' in v.name:
            bias = v.numpy()
        elif f'{layer_name}/' in v.name and 'reparam_beta' in v.name:
            gdn_beta = v.numpy()
        elif f'{layer_name}/' in v.name and 'reparam_gamma' in v.name:
            gdn_gamma = v.numpy()
    
    if kernel_real is not None:
        print(f"\n{layer_name}:")
        print(f"  kernel_real: {kernel_real.shape}")
        
        # Convertir a kernel espacial
        kernel_spatial = rdft_to_spatial(kernel_real, kernel_imag, [kh, kw])
        print(f"  kernel_spatial: {kernel_spatial.shape}")
        
        # Guardar kernel en formato (kH, kW, C_in, C_out)
        layer_idx = int(layer_name.split('_')[1])
        fname = f"analysis_conv_{layer_idx}_kernel.bin"
        kernel_spatial.astype(np.float32).tofile(f"{output_dir}/{fname}")
        weights_index.append((fname, "float32", kernel_spatial.nbytes, 
                              f"{kh}x{kw}x{c_in}x{c_out}"))
        print(f"  Guardado: {fname} ({kernel_spatial.nbytes} bytes)")
        
        # Guardar bias si existe
        if bias is not None:
            fname = f"analysis_conv_{layer_idx}_bias.bin"
            bias.astype(np.float32).tofile(f"{output_dir}/{fname}")
            weights_index.append((fname, "float32", bias.nbytes, str(c_out)))
            print(f"  Guardado: {fname}")
        
        # Guardar GDN params si existen (reparametrizar: beta = reparam^2, gamma = reparam^2)
        if gdn_beta is not None:
            # La reparametrización de GDN es: value = variable^2 - offset^2 + minimum
            # Para simplificar, calculamos el valor efectivo
            offset = 2 ** -18
            beta_eff = gdn_beta ** 2 - offset ** 2
            beta_eff = np.maximum(beta_eff, 0)  # Asegurar no negativo
            
            fname = f"analysis_gdn_{layer_idx}_beta.bin"
            beta_eff.astype(np.float32).tofile(f"{output_dir}/{fname}")
            weights_index.append((fname, "float32", beta_eff.nbytes, str(c_out)))
            print(f"  Guardado: {fname}")
        
        if gdn_gamma is not None:
            offset = 2 ** -18
            gamma_eff = gdn_gamma ** 2 - offset ** 2
            gamma_eff = np.maximum(gamma_eff, 0)
            
            fname = f"analysis_gdn_{layer_idx}_gamma.bin"
            gamma_eff.astype(np.float32).tofile(f"{output_dir}/{fname}")
            weights_index.append((fname, "float32", gamma_eff.nbytes, f"{c_out}x{c_out}"))
            print(f"  Guardado: {fname}")

# Copiar spectral (ya está en formato correcto - matriz densa 8x8)
for v in spectral.variables:
    if 'kernel' in v.name.lower() or 'dense' in v.name.lower():
        kernel = v.numpy()
        fname = "spectral_analysis_kernel.bin"
        kernel.astype(np.float32).tofile(f"{output_dir}/{fname}")
        weights_index.append((fname, "float32", kernel.nbytes, f"{kernel.shape[0]}x{kernel.shape[1]}"))
        print(f"\nSpectral: {kernel.shape} -> {fname}")

# Copiar modulating
mod_vars = {}
for v in modulating.variables:
    mod_vars[v.name] = v.numpy()

# Dense 1 (entrada -> hidden)
for name, val in mod_vars.items():
    if 'dense_1/kernel' in name:
        fname = "mod_dense_1_kernel.bin"
        val.astype(np.float32).tofile(f"{output_dir}/{fname}")
        weights_index.append((fname, "float32", val.nbytes, f"{val.shape[0]}x{val.shape[1]}"))
        print(f"Modulating dense_1 kernel: {val.shape}")
    elif 'dense_1/bias' in name:
        fname = "mod_dense_1_bias.bin"
        val.astype(np.float32).tofile(f"{output_dir}/{fname}")
        weights_index.append((fname, "float32", val.nbytes, str(val.shape[0])))
        print(f"Modulating dense_1 bias: {val.shape}")
    elif 'dense_2/kernel' in name:
        fname = "mod_dense_2_kernel.bin"
        val.astype(np.float32).tofile(f"{output_dir}/{fname}")
        weights_index.append((fname, "float32", val.nbytes, f"{val.shape[0]}x{val.shape[1]}"))
        print(f"Modulating dense_2 kernel: {val.shape}")
    elif 'dense_2/bias' in name:
        fname = "mod_dense_2_bias.bin"
        val.astype(np.float32).tofile(f"{output_dir}/{fname}")
        weights_index.append((fname, "float32", val.nbytes, str(val.shape[0])))
        print(f"Modulating dense_2 bias: {val.shape}")

# Escribir índice
with open(f"{output_dir}/weights_index.tsv", "w") as f:
    f.write("filename\tdtype\tsize_bytes\tshape\n")
    for row in weights_index:
        f.write("\t".join(str(x) for x in row) + "\n")

print(f"\n=== Pesos exportados a {output_dir} ===")
print(f"Total archivos: {len(weights_index)}")
