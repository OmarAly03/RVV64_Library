import numpy as np
import os
import sys

# --- Configuration ---
N, C, H, W = 1, 3, 32, 32

if len(sys.argv) >= 3:
    H = int(sys.argv[1])
    W = int(sys.argv[2])

# --- Ensure output directory exists ---
os.makedirs("output_files", exist_ok=True)

# --- Generate and Save Input Data ---
print("Generating input tensor with NumPy...")
# Use a fixed seed for reproducibility
np.random.seed(42)
X = np.random.rand(N, C, H, W).astype(np.float32)
X.tofile("./output_files/X.bin")
print("Input tensor saved to ./output_files/X.bin")
