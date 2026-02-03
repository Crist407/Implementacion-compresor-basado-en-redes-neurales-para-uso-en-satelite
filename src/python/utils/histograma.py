import numpy as np
import matplotlib.pyplot as plt

# Dimensiones del latente (por defecto)
H, W, C = 32, 32, 3072
N = H * W * C

# Cargar archivo binario en numpy array
def load_bin_file(path: str, shape_flat: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != shape_flat:
        raise ValueError(f"Unexpected size for {path}: got {arr.size}, expected {shape_flat}")
    return arr

# Cargar arrays
YhatPY = load_bin_file("debug_dumps/python_ground_truth.bin", N)
YhatC = load_bin_file("debug_dumps/Y_hat_c_even.bin", N)


# Crear figura con dos subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histograma Python
ax1.hist(YhatPY, bins=100, edgecolor='black', alpha=0.7, color='blue')
ax1.set_title('Histograma Python Ground Truth')
ax1.set_xlabel('Valor')
ax1.set_ylabel('Frecuencia')
ax1.grid(True, alpha=0.3)

# Histograma C
ax2.hist(YhatC, bins=100, edgecolor='black', alpha=0.7, color='green')
ax2.set_title('Histograma C Y_hat_c_even')
ax2.set_xlabel('Valor')
ax2.set_ylabel('Frecuencia')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('histogramas_comparacion.png', dpi=150)
print("Histogramas guardados en 'histogramas_comparacion.png'")
plt.show()  # Mostrar ventana con los histogramas

# Estadísticas básicas
print(f"\nPython - Min: {YhatPY.min():.3f}, Max: {YhatPY.max():.3f}, Mean: {YhatPY.mean():.3f}, Std: {YhatPY.std():.3f}")
print(f"C      - Min: {YhatC.min():.3f}, Max: {YhatC.max():.3f}, Mean: {YhatC.mean():.3f}, Std: {YhatC.std():.3f}")
print(f"\nDiferencia - Max: {np.abs(YhatPY - YhatC).max():.6f}, Mean: {np.abs(YhatPY - YhatC).mean():.6f}")











