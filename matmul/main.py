import numpy as np
import onnx
import onnxruntime as ort
import sys
import os

# Get the absolute path to the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============== Import Utility Functions ===============
from src.onnx_utils import max_abs_error, snr_db

# =========== Import Implementations Functions ===========
from src.matmult import matmul_py_scalar

# ==== Loading ONNX Model ====
onnx_model = onnx.load(os.path.join(SCRIPT_DIR, "./output_files/matrix_multiply.onnx"))
onnx.checker.check_model(onnx_model)
session = ort.InferenceSession(os.path.join(SCRIPT_DIR, "./output_files/matrix_multiply.onnx"))

# --- HANDLE ARGUMENTS ---
M, N, K = 4, 4, 4  # defaults
tilesize = 8  # default tile size

if len(sys.argv) >= 5:
    # Format: python main.py M N K tilesize
    try:
        M = int(sys.argv[1])
        N = int(sys.argv[2])
        K = int(sys.argv[3])
        tilesize = int(sys.argv[4])
        if M <= 0 or N <= 0 or K <= 0 or tilesize <= 0:
            raise ValueError("All arguments must be positive")
    except (ValueError, IndexError):
        print("Invalid arguments. Usage: python main.py M N K tilesize")
        print(f"Using defaults: {M}x{N} (K={K}), tile={tilesize}")
        M, N, K, tilesize = 4, 4, 4, 8

elif len(sys.argv) == 4:
    # Format: python main.py M N K (use default tile size)
    try:
        M = int(sys.argv[1])
        N = int(sys.argv[2])
        K = int(sys.argv[3])
        tilesize = min(32, min(M, min(N, K)))  # Auto tile size
    except (ValueError, IndexError):
        print("Invalid arguments. Using defaults.")
        M, N, K, tilesize = 4, 4, 4, 8

elif len(sys.argv) == 3:
    # Format: python main.py size tilesize (square matrices)
    try:
        size = int(sys.argv[1])
        tilesize = int(sys.argv[2])
        if size <= 0 or tilesize <= 0:
            raise ValueError("All arguments must be positive")
        M = N = K = size
    except (ValueError, IndexError):
        print("Invalid arguments. Usage: python main.py size tilesize")
        print(f"Using defaults: {M}x{N}, tile={tilesize}")
        M, N, K, tilesize = 4, 4, 4, 8

elif len(sys.argv) == 2:
    # Format: python main.py size (square matrices, auto tile size)
    try:
        size = int(sys.argv[1])
        if size <= 0:
            raise ValueError("Size must be positive")
        M = N = K = size
        tilesize = min(32, size)  # Auto tile size
    except (ValueError, IndexError):
        print("Invalid argument. Using defaults.")
        M, N, K, tilesize = 4, 4, 4, 8

# Validate tile size
if tilesize > M or tilesize > N or tilesize > K:
    print(f"Warning: Tile size ({tilesize}) is larger than matrix dimensions.")
    tilesize = min(M, min(N, K))
    print(f"Adjusting tile size to: {tilesize}")

# Load matrices
A = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/A.bin"), dtype=np.float32).reshape(M, K)
B = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/B.bin"), dtype=np.float32).reshape(K, N)

print(f"\nMatrix multiplication: A({M}x{K}) @ B({K}x{N}) -> C({M}x{N})")
print(f"Total operations: {2 * M * K * N:,} FLOPs")

# ==== ONNX Golden Reference (using ONNXRuntime) ====
input_names = [input.name for input in onnx_model.graph.input]
onnx_ref = session.run(None, {input_names[0]: A, input_names[1]: B})[0]

# ==== C Scalar ====
c_scalar = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/c_scalar.bin"), dtype=np.float32).reshape(M, N)

# ==== C Scalar Tiled Version ====
c_tiled_scalar = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/c_tiled_scalar.bin"), dtype=np.float32).reshape(M, N)

c_tiled_e32m8 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/c_tiled_e32m8.bin"), dtype=np.float32).reshape(M, N)

# ONNX --> golden reference
c_ref = onnx_ref

# ==== Results Table ====
implementations = [
	("ONNX Golden Ref", onnx_ref),
	("C Scalar", c_scalar),
	("C Tiled Scalar", c_tiled_scalar),
	("C Tiled e32m8", c_tiled_e32m8),
]

print(f"\n{'Implementation':<25}{'Max Abs Error':<20}{'SNR (dB)':<20}")
print("-" * 60)

for name, result in implementations:
    mae = max_abs_error(c_ref, result)
    snr = snr_db(c_ref, result)
    print(f"{name:<25}{mae:<20.6g}{snr:<20.6g}")
