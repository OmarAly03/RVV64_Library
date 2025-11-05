import numpy as np
import onnx
import onnxruntime as ort
import sys
import os

# Get the absolute path to the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR == "":
    SCRIPT_DIR = "."

# =============== Import Utility Functions ===============
sys.path.append(os.path.join(SCRIPT_DIR, "src"))
try:
    from src.onnx_utils import max_abs_error, snr_db
except ImportError:
    print("Error: Could not import from src.onnx_utils")
    print("Please ensure onnx_utils.py is in the 'src' directory.")
    sys.exit(1)


# ==== Loading ONNX Model ====
onnx_path = os.path.join(SCRIPT_DIR, "./output_files/dense.onnx")
try:
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(onnx_path)
except FileNotFoundError:
    print(f"Error: ONNX model not found at {onnx_path}")
    print("Please ensure 'make' successfully ran src/onnx_dense.py")
    sys.exit(1)


# Default sizes
B, IN, OUT = 1, 16, 16

if len(sys.argv) == 4:
    try:
        B = int(sys.argv[1])
        IN = int(sys.argv[2])
        OUT = int(sys.argv[3])
    except ValueError:
        print(f"Invalid size arguments. Using defaults B=1, IN=16, OUT=16.")
        B, IN, OUT = 1, 16, 16

# ==== Load All Input Data ====
def load_data(filename, shape):
    path = os.path.join(SCRIPT_DIR, f"./output_files/{filename}")
    try:
        return np.fromfile(path, dtype=np.float32).reshape(shape)
    except FileNotFoundError:
        print(f"Error: Input data not found at {path}")
        print("Please ensure QEMU simulation (run_dense) ran successfully.")
        sys.exit(1)

input_data = load_data("input.bin", (B, IN))
weights_data = load_data("weights.bin", (OUT, IN))
bias_data = load_data("bias.bin", (OUT))

print(f"\nDense (GEMM) validation with B={B}, IN={IN}, OUT={OUT}")

# ==== ONNX Golden Reference (using ONNXRuntime) ====
# Note: Input names "A", "B", "C" must match onnx_dense.py
onnx_ref = session.run(None, {
    "A": input_data,
    "B": weights_data,
    "C": bias_data
})[0]

# ONNX --> golden reference
c_ref = onnx_ref

# ==== Load C Kernel Results ====
def load_result(filename, shape):
    path = os.path.join(SCRIPT_DIR, f"./output_files/{filename}")
    try:
        return np.fromfile(path, dtype=np.float32).reshape(shape)
    except FileNotFoundError:
        print(f"Warning: Output file not found: {path}. Skipping.")
        return None

output_shape = (B, OUT)
implementations = [
    ("ONNX Golden Ref", onnx_ref),
    ("C Scalar", load_result("dense_scalar.bin", output_shape)),
    ("C Vectorized (e32m1)", load_result("dense_e32m1.bin", output_shape)),
    ("C Vectorized (e32m2)", load_result("dense_e32m2.bin", output_shape)),
    ("C Vectorized (e32m4)", load_result("dense_e32m4.bin", output_shape)),
    ("C VectorZ Vectorized (e32m8)", load_result("dense_e32m8.bin", output_shape)),
]

# ==== Results Table ====
print(f"\n{'Implementation':<30}{'Max Abs Error':<20}{'SNR (dB)':<20}")
print("-" * 70)

for name, result in implementations:
    if result is None:
        print(f"{name:<30}{'SKIPPED':<20}{'SKIPPED':<20}")
        continue
    
    mae = max_abs_error(c_ref, result)
    snr = snr_db(c_ref, result)
    print(f"{name:<30}{mae:<20.6g}{snr:<20.6g}")