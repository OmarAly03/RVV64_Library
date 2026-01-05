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
onnx_path = os.path.join(SCRIPT_DIR, "./output_files/maxpool.onnx")
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
session = ort.InferenceSession(onnx_path)

# Default parameters
N, C, H, W = 16, 1, 4, 4
KH, KW = 2, 2
SH, SW = 1, 1  # Default stride should be 2, not 1
PH, PW = 0, 0

# Parse command line arguments: N C H W KH KW SH SW PH PW
if len(sys.argv) >= 5:
    N = int(sys.argv[1])
    C = int(sys.argv[2])
    H = int(sys.argv[3])
    W = int(sys.argv[4])
    
if len(sys.argv) >= 7:
    KH = int(sys.argv[5])
    KW = int(sys.argv[6])
    
if len(sys.argv) >= 9:
    SH = int(sys.argv[7])
    SW = int(sys.argv[8])
    
if len(sys.argv) >= 11:
    PH = int(sys.argv[9])
    PW = int(sys.argv[10])

# Calculate output dimensions
OH = (H + 2 * PH - KH) // SH + 1
OW = (W + 2 * PW - KW) // SW + 1

input_shape = (N, C, H, W)
output_shape = (N, C, OH, OW)

print(f"\nMaxPool Verification")
print(f"Input Shape:  {input_shape}")
print(f"Kernel: {KH}x{KW}, Stride: {SH}x{SW}, Padding: {PH}x{PW}")
print(f"Output Shape: {output_shape}")

# Load input matrix
input_data = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/input.bin"), dtype=np.float32).reshape(input_shape)

# ==== ONNX Golden Reference ====
input_names = [i.name for i in onnx_model.graph.input]
onnx_ref = session.run(None, {input_names[0]: input_data})[0]

# ==== Load C Results ====
def safe_load(filename, expected_shape):
    """Safely load binary file and check size"""
    try:
        data = np.fromfile(os.path.join(SCRIPT_DIR, f"./output_files/{filename}"), dtype=np.float32)
        expected_size = np.prod(expected_shape)
        if data.size != expected_size:
            print(f"Warning: {filename} has {data.size} elements, expected {expected_size}")
            return None
        return data.reshape(expected_shape)
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return None

c_scalar = safe_load("maxpool_scalar.bin", output_shape)
c_e32m1 = safe_load("maxpool_e32m1.bin", output_shape)
c_e32m2 = safe_load("maxpool_e32m2.bin", output_shape)
c_e32m4 = safe_load("maxpool_e32m4.bin", output_shape)
c_e32m8 = safe_load("maxpool_e32m8.bin", output_shape)

maxpool_tiled_m1 = safe_load("maxpool_tiled_m1.bin", output_shape)
maxpool_tiled_m2 = safe_load("maxpool_tiled_m2.bin", output_shape)
maxpool_tiled_m4 = safe_load("maxpool_tiled_m4.bin", output_shape)
maxpool_tiled_m8 = safe_load("maxpool_tiled_m8.bin", output_shape)

# ==== Results Table ====
implementations = [
    ("ONNX Golden Ref", onnx_ref),
    ("C Scalar Reference", c_scalar),
    ("C RVV e32m1", c_e32m1),
    ("C RVV e32m2", c_e32m2),
    ("C RVV e32m4", c_e32m4),
    ("C RVV e32m8", c_e32m8),
    ("C RVV tiled_m1", maxpool_tiled_m1),
    ("C RVV tiled_m2", maxpool_tiled_m2),
    ("C RVV tiled_m4", maxpool_tiled_m4), 
    ("C RVV tiled_m8", maxpool_tiled_m8),
]

print(f"\n{'Implementation':<30}{'Max Abs Error':<20}{'SNR (dB)':<20}")
print("-" * 70)

# We compare everything against the ONNX Golden Reference
golden = onnx_ref

for name, result in implementations:
    if result is not None:
        mae = max_abs_error(golden, result)
        snr = snr_db(golden, result)
        print(f"{name:<30}{mae:<20.6g}{snr:<20.6g}")
    else:
        print(f"{name:<30}{'FILE NOT FOUND':<20}{'N/A':<20}")
    
print("-" * 70)