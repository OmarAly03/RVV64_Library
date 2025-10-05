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

# Default size
M, N, K = 4, 4, 4

if len(sys.argv) == 2:
    M = N = K = int(sys.argv[1])
elif len(sys.argv) >= 4:
    M = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])

# Load matrices
A = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/A.bin"), dtype=np.float32).reshape(M, K)
B = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/B.bin"), dtype=np.float32).reshape(K, N)

print(f"\nMatrix multiplication: A({M}x{K}) @ B({K}x{N}) -> C({M}x{N})")
print(f"Total operations: {2 * M * K * N:,} FLOPs")

# ==== ONNX Golden Reference (using ONNXRuntime) ====
input_names = [input.name for input in onnx_model.graph.input]
onnx_ref = session.run(None, {input_names[0]: A, input_names[1]: B})[0]

# ==== Python Scalar ====
py_scalar = matmul_py_scalar(A, B)

# ==== Python NumPy dot ====
py_numpy = np.dot(A, B)

# ==== C Scalar ====
c_scalar = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/c_scalar.bin"), dtype=np.float32).reshape(M, N)

# ==== C Vectorized (e32m1) ====
c_e32m1 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/c_e32m1.bin"), dtype=np.float32).reshape(M, N)

# ==== C Vectorized (e32m2) ====
c_e32m2 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/c_e32m2.bin"), dtype=np.float32).reshape(M, N)

# ==== C Vectorized (e32m4) ====
c_e32m4 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/c_e32m4.bin"), dtype=np.float32).reshape(M, N)

# ==== C Vectorized (e32m8) ====
c_e32m8 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/c_e32m8.bin"), dtype=np.float32).reshape(M, N)

# ONNX --> golden reference
c_ref = onnx_ref

# ==== Results Table ====
implementations = [
    ("ONNX Golden Ref", onnx_ref),
    ("Python Scalar", py_scalar),
    ("NumPy dot", py_numpy),
    ("C Scalar", c_scalar),
    ("C Vectorized (e32m1)", c_e32m1),
    ("C Vectorized (e32m2)", c_e32m2),
    ("C Vectorized (e32m4)", c_e32m4),
    ("C Vectorized (e32m8)", c_e32m8),  
]

print(f"\n{'Implementation':<25}{'Max Abs Error':<20}{'SNR (dB)':<20}")
print("-" * 60)

for name, result in implementations:
    mae = max_abs_error(c_ref, result)
    snr = snr_db(c_ref, result)
    print(f"{name:<25}{mae:<20.6g}{snr:<20.6g}")
