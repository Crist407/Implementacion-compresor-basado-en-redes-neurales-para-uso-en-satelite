import numpy as np
import matplotlib.pyplot as plt
import os

# Configuración
filename = 'T31TCG_20230907T104629_5.8_512_512_2_1_0.raw'
width = 512
height = 512
num_bandas = 8
bytes_per_pixel = 2  # uint16

# Leer archivo
if not os.path.exists(filename):
    raise FileNotFoundError(f"No se encuentra el archivo {filename}")

with open(filename, 'rb') as f:
    data = np.frombuffer(f.read(), dtype='<u2')

# Comprobar tamaño
expected_size = width * height * num_bandas
if data.size != expected_size:
    raise ValueError(f"Tamaño inesperado: {data.size} elementos, esperado {expected_size}")

# Reshape a (8, 512, 512)
bandas = data.reshape((num_bandas, height, width))

# Visualización
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(num_bandas):
    ax = axes[i//4, i%4]
    vmin, vmax = np.percentile(bandas[i], (1, 99))
    ax.imshow(bandas[i], cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f'Banda {i+1}')
    ax.axis('off')
plt.tight_layout()
plt.show()
