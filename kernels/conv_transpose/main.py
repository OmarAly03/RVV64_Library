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

# Fixed parameters: 3x3 kernel, configurable stride, no padding
input_size = 4
kernel_size = 3  # Fixed
stride = 1       # Default
in_channels = 1
out_channels = 1

# Parse command line arguments (input_size in_channels out_channels [stride])
if len(sys.argv) >= 5:
    input_size = int(sys.argv[1])
    in_channels = int(sys.argv[2])
    out_channels = int(sys.argv[3])
    stride = int(sys.argv[4])
elif len(sys.argv) >= 4:
    input_size = int(sys.argv[1])
    in_channels = int(sys.argv[2])
    out_channels = int(sys.argv[3])
elif len(sys.argv) >= 2:
    input_size = int(sys.argv[1])

# Calculate output dimensions based on stride
out_height = (input_size - 1) * stride + kernel_size
out_width = (input_size - 1) * stride + kernel_size

print(f"\nTransposed Convolution: {input_size}x{input_size} input, {kernel_size}x{kernel_size} kernel (fixed)")
print(f"Stride: {stride}, No Padding, Channels: {in_channels}->{out_channels}")
print(f"Output: {out_height}x{out_width}")

# Load input and kernel data
# All implementations use kernel layout: [in_channels, out_channels, kernel_h, kernel_w]
input_data = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/input.bin"), dtype=np.float32).reshape(1, in_channels, input_size, input_size)
kernel_data = np.fromfile(os.path.join(SCRIPT_DIR, "./output_files/kernel.bin"), dtype=np.float32).reshape(in_channels, out_channels, kernel_size, kernel_size)

# ==== ONNX Golden Reference (using ONNXRuntime) ====
input_names = [input.name for input in onnx_model.graph.input]
onnx_ref = session.run(None, {input_names[0]: input_data, input_names[1]: kernel_data})[0]

# Helper function to load output file
def load_output(filename):
    path = os.path.join(SCRIPT_DIR, f"./output_files/{filename}")
    if os.path.exists(path):
        return np.fromfile(path, dtype=np.float32).reshape(1, out_channels, out_height, out_width)
    return None

# ==== Load all implementations ====
c_scalar = load_output("output_scalar.bin")

# 3x3 Specialized RVV implementations
c_3x3_m1 = load_output("output_3x3_m1.bin")
c_3x3_m2 = load_output("output_3x3_m2.bin")
c_3x3_m4 = load_output("output_3x3_m4.bin")
c_3x3_m8 = load_output("output_3x3_m8.bin")

# General RVV implementations
c_general_m1 = load_output("output_general_m1.bin")
c_general_m2 = load_output("output_general_m2.bin")
c_general_m4 = load_output("output_general_m4.bin")
c_general_m8 = load_output("output_general_m8.bin")

# ONNX --> golden reference
c_ref = onnx_ref

# ==== Results Table ====
implementations = [("ONNX Golden Ref", onnx_ref)]

if c_scalar is not None:
    implementations.append(("C Scalar", c_scalar))

# 3x3 Specialized
if c_3x3_m1 is not None:
    implementations.append(("RVV 3x3 (m1)", c_3x3_m1))
if c_3x3_m2 is not None:
    implementations.append(("RVV 3x3 (m2)", c_3x3_m2))
if c_3x3_m4 is not None:
    implementations.append(("RVV 3x3 (m4)", c_3x3_m4))
if c_3x3_m8 is not None:
    implementations.append(("RVV 3x3 (m8)", c_3x3_m8))

# General
if c_general_m1 is not None:
    implementations.append(("RVV General (m1)", c_general_m1))
if c_general_m2 is not None:
    implementations.append(("RVV General (m2)", c_general_m2))
if c_general_m4 is not None:
    implementations.append(("RVV General (m4)", c_general_m4))
if c_general_m8 is not None:
    implementations.append(("RVV General (m8)", c_general_m8))

print(f"\n{'Implementation':<25}{'Max Abs Error':<20}{'SNR (dB)':<20}")
print("-" * 65)

for name, result in implementations:
    mae = max_abs_error(c_ref, result)
    snr = snr_db(c_ref, result)
    print(f"{name:<25}{mae:<20.6g}{snr:<20.6g}")
