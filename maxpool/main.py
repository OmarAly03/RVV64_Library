import numpy as np
import onnxruntime as ort
import sys
import os

# --- Utility and Implementation Imports ---
from src.onnx_utils import max_abs_error, snr_db

# --- Configuration ---
N, C, H, W = 1, 3, 32, 32
KERNEL_SIZE, STRIDE = 3, 2
CEIL_MODE = 0 # Set to 1 to test ceil_mode

if len(sys.argv) >= 3:
    H = int(sys.argv[1])
    W = int(sys.argv[2])

# Calculate output shape
if CEIL_MODE:
    OH = (H + STRIDE - KERNEL_SIZE + STRIDE - 1) // STRIDE
    OW = (W + STRIDE - KERNEL_SIZE + STRIDE - 1) // STRIDE
else:
    OH = (H - KERNEL_SIZE) // STRIDE + 1
    OW = (W - KERNEL_SIZE) // STRIDE + 1
output_shape = (N, C, OH, OW)

# --- Load All Data and Results from Files ---
print("Loading input tensor and all C++ results...")
X = np.fromfile("./output_files/X.bin", dtype=np.float32).reshape(N, C, H, W)
y_scalar = np.fromfile("./output_files/Y_scalar.bin", dtype=np.float32).reshape(output_shape)
i_scalar = np.fromfile("./output_files/I_scalar.bin", dtype=np.int64).reshape(output_shape)
y_e32m1 = np.fromfile("./output_files/Y_e32m1.bin", dtype=np.float32).reshape(output_shape)
i_e32m1 = np.fromfile("./output_files/I_e32m1.bin", dtype=np.int64).reshape(output_shape)
y_e32m2 = np.fromfile("./output_files/Y_e32m2.bin", dtype=np.float32).reshape(output_shape)
i_e32m2 = np.fromfile("./output_files/I_e32m2.bin", dtype=np.int64).reshape(output_shape)
y_e32m4 = np.fromfile("./output_files/Y_e32m4.bin", dtype=np.float32).reshape(output_shape)
i_e32m4 = np.fromfile("./output_files/I_e32m4.bin", dtype=np.int64).reshape(output_shape)
y_e32m8 = np.fromfile("./output_files/Y_e32m8.bin", dtype=np.float32).reshape(output_shape)
i_e32m8 = np.fromfile("./output_files/I_e32m8.bin", dtype=np.int64).reshape(output_shape)

# --- Run ONNX Golden Reference ---
session = ort.InferenceSession("./output_files/maxpool.onnx")
input_name = session.get_inputs()[0].name
onnx_ref_y, onnx_ref_i = session.run(None, {input_name: X})

# --- Display The Final, Combined Results Table ---
print(f"\nMaxPool Verification: X({N}x{C}x{H}x{W}) -> Y({N}x{C}x{OH}x{OW})")
print(f"{'Implementation':<28}{'Max Abs Error':<20}{'SNR (dB)':<20}{'Indices Match?':<20}")
print("-" * 88)

implementations = [
    ("C++ Scalar", y_scalar, i_scalar),
    ("C++ Vectorized (e32m1)", y_e32m1, i_e32m1),
    ("C++ Vectorized (e32m2)", y_e32m2, i_e32m2),
    ("C++ Vectorized (e32m4)", y_e32m4, i_e32m4),
    ("C++ Vectorized (e32m8)", y_e32m8, i_e32m8),
]

for name, y_result, i_result in implementations:
    mae = max_abs_error(onnx_ref_y, y_result)
    snr = snr_db(onnx_ref_y, y_result)
    indices_match = "CORRECT" if np.array_equal(i_result, onnx_ref_i) else "INCORRECT"
    print(f"{name:<28}{mae:<20.6g}{snr:<20.6g}{indices_match:<20}")
