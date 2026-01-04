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
onnx_model = onnx.load(os.path.join(SCRIPT_DIR, "./output_files/batch_norm.onnx"))
onnx.checker.check_model(onnx_model)
session = ort.InferenceSession(os.path.join(SCRIPT_DIR, "./output_files/batch_norm.onnx"))

# Default tensor size: 2x3x4x4 (batch=2, channels=3, height=4, width=4)
N, C, H, W = 2, 3, 4, 4

if len(sys.argv) == 5:
    N = int(sys.argv[1])
    C = int(sys.argv[2]) 
    H = int(sys.argv[3])
    W = int(sys.argv[4])

input_size = N * C * H * W
output_size = input_size  # BatchNorm doesn't change tensor size

# Load matrices
input_data = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/input.bin"), dtype=np.float32).reshape(N, C, H, W)

print(f"\nBatchNormalization on {N}x{C}x{H}x{W} tensor")

# ==== ONNX Golden Reference (using ONNXRuntime) ====
input_names = [input.name for input in onnx_model.graph.input]
onnx_ref = session.run(None, {input_names[0]: input_data})[0]
onnx_ref_flat = onnx_ref.flatten()

# ==== C Scalar ====
c_scalar = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/batch_norm_scalar.bin"), dtype=np.float32).reshape(output_size)

# ==== C Tiled Scalar ====
c_tiled_scalar = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/batch_norm_tiled_scalar.bin"), dtype=np.float32).reshape(output_size)

# ==== C Vectorized (e32mx) ====
c_e32m1 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/batch_norm_e32m1.bin"), dtype=np.float32).reshape(output_size)
c_e32m2 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/batch_norm_e32m2.bin"), dtype=np.float32).reshape(output_size)
c_e32m4 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/batch_norm_e32m4.bin"), dtype=np.float32).reshape(output_size)
c_e32m8 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/batch_norm_e32m8.bin"), dtype=np.float32).reshape(output_size)

# ==== C Tiled Vectorized (e32mx) ====
c_tiled_e32m1 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/batch_norm_tiled_e32m1.bin"), dtype=np.float32).reshape(output_size)
c_tiled_e32m2 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/batch_norm_tiled_e32m2.bin"), dtype=np.float32).reshape(output_size)
c_tiled_e32m4 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/batch_norm_tiled_e32m4.bin"), dtype=np.float32).reshape(output_size)
c_tiled_e32m8 = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/batch_norm_tiled_e32m8.bin"), dtype=np.float32).reshape(output_size)

# ONNX --> golden reference
c_ref = onnx_ref_flat

# ==== Results Table ====
implementations = [
    ("ONNX Golden Ref", onnx_ref_flat),
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
print("-" * 70)

for name, result in implementations:
    mae = max_abs_error(c_ref, result)
    snr = snr_db(c_ref, result)
    print(f"{name:<30}{mae:<20.6g}{snr:<20.6g}")