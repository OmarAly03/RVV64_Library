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
onnx_model = onnx.load(os.path.join(SCRIPT_DIR, "./output_files/conv_transpose.onnx"))
onnx.checker.check_model(onnx_model)
session = ort.InferenceSession(os.path.join(SCRIPT_DIR, "./output_files/conv_transpose.onnx"))

# Default parameters
input_size = 4
kernel_size = 3
stride = 2
padding = 1
in_channels = 1
out_channels = 1

# Parse command line arguments
if len(sys.argv) >= 7:
    input_size = int(sys.argv[1])
    kernel_size = int(sys.argv[2])
    stride = int(sys.argv[3])
    padding = int(sys.argv[4])
    in_channels = int(sys.argv[5])
    out_channels = int(sys.argv[6])
elif len(sys.argv) >= 2:
    input_size = int(sys.argv[1])

# Calculate output dimensions
out_height = (input_size - 1) * stride - 2 * padding + kernel_size
out_width = (input_size - 1) * stride - 2 * padding + kernel_size

print(f"\nTransposed Convolution: {input_size}x{input_size} input, {kernel_size}x{kernel_size} kernel")
print(f"Stride: {stride}, Padding: {padding}, Channels: {in_channels}->{out_channels}")
print(f"Output: {out_height}x{out_width}")

# Load input and kernel data
input_data = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/input.bin"), dtype=np.float32).reshape(1, in_channels, input_size, input_size)
kernel_data = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/kernel.bin"), dtype=np.float32).reshape(in_channels, out_channels, kernel_size, kernel_size)

# ==== ONNX Golden Reference (using ONNXRuntime) ====
input_names = [input.name for input in onnx_model.graph.input]
onnx_ref = session.run(None, {input_names[0]: input_data, input_names[1]: kernel_data})[0]

# ==== C Scalar (optional if present) ====
c_scalar = None
scalar_path = os.path.join(SCRIPT_DIR, "./output_files/output_scalar.bin")
if os.path.exists(scalar_path):
    c_scalar = np.fromfile(scalar_path, dtype=np.float32).reshape(1, out_channels, out_height, out_width)

# ==== C Vectorized (e32m1) (optional if present) ====
c_e32m1 = None
e32m1_path = os.path.join(SCRIPT_DIR, "./output_files/output_e32m1.bin")
if os.path.exists(e32m1_path):
    c_e32m1 = np.fromfile(e32m1_path, dtype=np.float32).reshape(1, out_channels, out_height, out_width)

# ONNX --> golden reference
c_ref = onnx_ref

# ==== Results Table ====
implementations = [("ONNX Golden Ref", onnx_ref)]
if c_scalar is not None:
    implementations.append(("C Scalar", c_scalar))
if c_e32m1 is not None:
    implementations.append(("C Vectorized (e32m1)", c_e32m1))

print(f"\n{'Implementation':<25}{'Max Abs Error':<20}{'SNR (dB)':<20}")
print("-" * 60)

for name, result in implementations:
    mae = max_abs_error(c_ref, result)
    snr = snr_db(c_ref, result)
    print(f"{name:<25}{mae:<20.6g}{snr:<20.6g}")
