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
    from onnx_utils import max_abs_error, snr_db
except ImportError:
    print("Error: Could not import from src.onnx_utils")
    print("Please ensure onnx_utils.py is in the 'src' directory.")
    sys.exit(1)


# ==== Loading ONNX Model ====
onnx_path = os.path.join(SCRIPT_DIR, "./output_files/tensor_add.onnx")
try:
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(onnx_path)
except FileNotFoundError:
    print(f"Error: ONNX model not found at {onnx_path}")
    print("Please ensure 'make' successfully ran src/onnx_tensor_add.py")
    sys.exit(1)


# Default size
N = 16

if len(sys.argv) == 2:
    try:
        N = int(sys.argv[1])
    except ValueError:
        print(f"Invalid size argument '{sys.argv[1]}'. Using default N=16.")
        N = 16

# ==== Load All Input Data ====
def load_data(filename, shape):
    path = os.path.join(SCRIPT_DIR, f"./output_files/{filename}")
    try:
        return np.fromfile(path, dtype=np.float32).reshape(shape)
    except FileNotFoundError:
        print(f"Error: Input data not found at {path}")
        print("Please ensure QEMU simulation (run_tensor_add) ran successfully.")
        sys.exit(1)

input_data_a = load_data("input_a.bin", (N))
input_data_b = load_data("input_b.bin", (N))


print(f"\nTensorAdd validation on {N} elements")

# ==== ONNX Golden Reference (using ONNXRuntime) ====
# Input names "A" and "B" must match onnx_tensor_add.py
onnx_ref = session.run(None, {
    "A": input_data_a,
    "B": input_data_b
})[0]

c_ref = onnx_ref

# ==== Load C Kernel Results ====
output_shape = (N)
implementations = [
    ("ONNX Golden Ref", onnx_ref),
    ("C Scalar", load_data("tensor_add_scalar.bin", output_shape)),
    ("C Vectorized (e32m1)", load_data("tensor_add_e32m1.bin", output_shape)),
    ("C Vectorized (e32m2)", load_data("tensor_add_e32m2.bin", output_shape)),
    ("C Vectorized (e32m4)", load_data("tensor_add_e32m4.bin", output_shape)),
    ("C Vectorized (e32m8)", load_data("tensor_add_e32m8.bin", output_shape)),
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