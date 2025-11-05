import numpy as np
import onnx
import onnxruntime as ort
import sys
import os

# Get the absolute path to the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============== Import Utility Functions ===============
from src.onnx_utils import max_abs_error, snr_db

# ==== Loading ONNX Model ====
onnx_model = onnx.load(os.path.join(SCRIPT_DIR, "./output_files/relu.onnx"))
onnx.checker.check_model(onnx_model)
session = ort.InferenceSession(os.path.join(SCRIPT_DIR, "./output_files/relu.onnx"))

# Default size
N = 16

if len(sys.argv) == 2:
    N = int(sys.argv[1])

# Load matrices
input = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/input.bin"), dtype=np.float32).reshape(N)

print(f"\nReLU activation on {N} elements")

# ==== ONNX Golden Reference (using ONNXRuntime) ====
input_names = [input.name for input in onnx_model.graph.input]
onnx_ref = session.run(None, {input_names[0]: input})[0]

# ==== C Scalar ====
c_scalar = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/relu_scalar.bin"), dtype=np.float32).reshape(N)

# ==== C Tiled Scalar ====
c_tiled_scalar = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/relu_tiled_scalar.bin"), dtype=np.float32).reshape(N)

# ==== C Vectorized (e32mx) ====
c_e32m1 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/relu_e32m1.bin"), dtype=np.float32).reshape(N)
c_e32m2 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/relu_e32m2.bin"), dtype=np.float32).reshape(N)
c_e32m4 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/relu_e32m4.bin"), dtype=np.float32).reshape(N)
c_e32m8 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/relu_e32m8.bin"), dtype=np.float32).reshape(N)

# ==== C Tiled Vectorized (e32mx) ====
c_tiled_e32m1 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/relu_tiled_e32m1.bin"), dtype=np.float32).reshape(N)
c_tiled_e32m2 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/relu_tiled_e32m2.bin"), dtype=np.float32).reshape(N)
c_tiled_e32m4 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/relu_tiled_e32m4.bin"), dtype=np.float32).reshape(N)
c_tiled_e32m8 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/relu_tiled_e32m8.bin"), dtype=np.float32).reshape(N)

# ONNX --> golden reference
c_ref = onnx_ref

# ==== Results Table ====
implementations = [
    ("ONNX Golden Ref", onnx_ref),
    ("C Scalar", c_scalar),
    ("C Tiled Scalar", c_tiled_scalar),
    ("C Vectorized (e32m1)", c_e32m1),
    ("C Vectorized (e32m2)", c_e32m2),
    ("C Vectorized (e32m4)", c_e32m4),
    ("C Vectorized (e32m8)", c_e32m8),
    ("C Tiled Vectorized (e32m1)", c_tiled_e32m1),
    ("C Tiled Vectorized (e32m2)", c_tiled_e32m2),
    ("C Tiled Vectorized (e32m4)", c_tiled_e32m4),
    ("C Tiled Vectorized (e32m8)", c_tiled_e32m8),  
]

print(f"\n{'Implementation':<30}{'Max Abs Error':<20}{'SNR (dB)':<20}")
print("-" * 60)

for name, result in implementations:
    mae = max_abs_error(c_ref, result)
    snr = snr_db(c_ref, result)
    print(f"{name:<30}{mae:<20.6g}{snr:<20.6g}")
