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
# Note: Ensure your ONNX model matches the KH=2, KW=2, S=2 configuration used in C
onnx_path = os.path.join(SCRIPT_DIR, "./output_files/maxpool.onnx")
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
session = ort.InferenceSession(onnx_path)

# Default size (Matching the C main.cpp defaults)
N, C, H, W = 16, 1, 4, 4
KH, KW = 2, 2
SH, SW = 2, 2
PH, PW = 0, 0

if len(sys.argv) == 5:
    N = int(sys.argv[1])
    C = int(sys.argv[2])
    H = int(sys.argv[3])
    W = int(sys.argv[4])

# Calculate output dimensions
OH = (H + 2 * PH - KH) // SH + 1
OW = (W + 2 * PW - KW) // SW + 1

input_shape = (N, C, H, W)
output_shape = (N, C, OH, OW)

print(f"\nMaxPool Verification")
print(f"Input Shape:  {input_shape}")
print(f"Output Shape: {output_shape}")

# Load input matrix
input_data = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/input.bin"), dtype=np.float32).reshape(input_shape)

# ==== ONNX Golden Reference ====
input_names = [i.name for i in onnx_model.graph.input]
onnx_ref = session.run(None, {input_names[0]: input_data})[0]

# ==== Load C Results ====
# We use the two kernels we just built
c_scalar = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/maxpool_scalar.bin"), dtype=np.float32).reshape(output_shape)
c_e32m1 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/maxpool_e32m1.bin"), dtype=np.float32).reshape(output_shape)
c_e32m2 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/maxpool_e32m2.bin"), dtype=np.float32).reshape(output_shape)
c_e32m4 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/maxpool_e32m4.bin"), dtype=np.float32).reshape(output_shape)
c_e32m8 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/maxpool_e32m8.bin"), dtype=np.float32).reshape(output_shape)

# ==== Results Table ====
implementations = [
    ("ONNX Golden Ref", onnx_ref),
    ("C Scalar Reference", c_scalar),
    ("C RVV e32m1", c_e32m1),
    ("C RVV e32m2", c_e32m2),
    ("C RVV e32m4", c_e32m4),
    ("C RVV e32m8", c_e32m8),
]

print(f"\n{'Implementation':<30}{'Max Abs Error':<20}{'SNR (dB)':<20}")
print("-" * 70)

# We compare everything against the ONNX Golden Reference
golden = onnx_ref

for name, result in implementations:
    mae = max_abs_error(golden, result)
    snr = snr_db(golden, result)
    print(f"{name:<30}{mae:<20.6g}{snr:<20.6g}")
    
print("-" * 70)
