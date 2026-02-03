import numpy as np
import sys
import os

def compare_binaries(file1, file2):
    print(f"Comparing {file1} and {file2}...")
    
    if not os.path.exists(file1):
        print(f"Error: {file1} does not exist.")
        return
    if not os.path.exists(file2):
        print(f"Error: {file2} does not exist.")
        return

    size1 = os.path.getsize(file1)
    size2 = os.path.getsize(file2)

    if size1 != size2:
        print(f"Warning: File sizes differ! {file1}: {size1}, {file2}: {size2}")
    
    data1 = np.fromfile(file1, dtype=np.float32)
    data2 = np.fromfile(file2, dtype=np.float32)

    if data1.shape != data2.shape:
        print(f"Error: Shapes differ! {data1.shape} vs {data2.shape}")
        # Truncate to smaller size for comparison if needed, or just fail
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]

    diff = np.abs(data1 - data2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    mse = np.mean(diff**2)

    print(f"Max Difference: {max_diff}")
    print(f"Mean Difference: {mean_diff}")
    print(f"MSE: {mse}")

    if max_diff == 0:
        print("SUCCESS: Files are identical.")
    elif max_diff < 1e-5:
        print("SUCCESS: Files are virtually identical (within float precision).")
    else:
        print("WARNING: Significant differences found.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_outputs.py <file1> <file2>")
        sys.exit(1)
    
    compare_binaries(sys.argv[1], sys.argv[2])
