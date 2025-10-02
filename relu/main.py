import numpy as np
import onnx
import onnxruntime as ort
import sys

# =============== Import Utility Functions ===============
from src.onnx_utils import max_abs_error, snr_db

# =========== Import Implementations Functions ===========
from src.relu_scalar import relu_py_scalar

# ==== Loading ONNX Model ====
onnx_model = onnx.load("./output_files/relu.onnx")
onnx.checker.check_model(onnx_model)
session = ort.InferenceSession("./output_files/relu.onnx")

# Default size
N = 16

if len(sys.argv) == 2:
    N = int(sys.argv[1])

# Load matrices
input = np.fromfile("./output_files/input.bin", dtype=np.float32).reshape(N)


print(f"\nReLU activation on {N} elements")

# ==== ONNX Golden Reference (using ONNXRuntime) ====
input_names = [input.name for input in onnx_model.graph.input]
onnx_ref = session.run(None, {input_names[0]: input})[0]

# ==== Python Scalar ====
py_scalar = relu_py_scalar(input) 

# ==== Python NumPy dot ====
py_numpy = np.maximum(input, 0)

# ==== C Scalar ====
c_scalar = np.fromfile("./output_files/relu_scalar.bin", dtype=np.float32).reshape(N)

# ==== C Vectorized (e32m1) ====
c_e32m1 = np.fromfile("./output_files/relu_e32m1.bin", dtype=np.float32).reshape(N)

# ==== C Vectorized (e32m2) ====
c_e32m2 = np.fromfile("./output_files/relu_e32m2.bin", dtype=np.float32).reshape(N)

# ==== C Vectorized (e32m4) ====
c_e32m4 = np.fromfile("./output_files/relu_e32m4.bin", dtype=np.float32).reshape(N)

# ==== C Vectorized (e32m8) ====
c_e32m8 = np.fromfile("./output_files/relu_e32m8.bin", dtype=np.float32).reshape(N)

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
