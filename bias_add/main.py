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
onnx_path = os.path.join(SCRIPT_DIR, "./output_files/bias_add.onnx")
try:
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(onnx_path)
except FileNotFoundError:
    print(f"Error: ONNX model not found at {onnx_path}")
    print("Please ensure 'make' successfully ran src/onnx_bias_add.py")
    sys.exit(1)


# Default sizes
B, C, H, W = 1, 16, 14, 14

if len(sys.argv) == 5:
    try:
        B = int(sys.argv[1])
        C = int(sys.argv[2])
        H = int(sys.argv[3])
        W = int(sys.argv[4])
    except ValueError:
        print(f"Invalid size arguments. Using defaults B=1, C=16, H=14, W=14.")
        B, C, H, W = 1, 16, 14, 14

# ==== Load All Input Data ====
def load_data(filename, shape):
    path = os.path.join(SCRIPT_DIR, f"./output_files/{filename}")
    try:
        return np.fromfile(path, dtype=np.float32).reshape(shape)
    except FileNotFoundError:
        print(f"Error: Input data not found at {path}")
        print("Please ensure QEMU simulation (run_bias_add) ran successfully.")
        sys.exit(1)

input_data = load_data("input.bin", (B, C, H, W))
bias_data_1d = load_data("bias.bin", (C))

# --- Crucial Step: Reshape 1D bias to 4D for ONNX broadcasting ---
bias_data_4d = bias_data_1d.reshape(1, C, 1, 1)

print(f"\nBiasAdd validation with B={B}, C={C}, H={H}, W={W}")

# ==== ONNX Golden Reference (using ONNXRuntime) ====
# Input names "A" and "B" must match onnx_bias_add.py
onnx_ref = session.run(None, {
    "A": input_data,
    "B": bias_data_4d
})[0]

c_ref = onnx_ref

# ==== Load C Kernel Results ====
output_shape = (B, C, H, W)
implementations = [
    ("ONNX Golden Ref", onnx_ref),
    ("C Scalar", load_data("bias_add_scalar.bin", output_shape)),
    ("C Vectorized (e32m1)", load_data("bias_add_e32m1.bin", output_shape)),
    ("C Vectorized (e32m2)", load_data("bias_add_e32m2.bin", output_shape)),
    ("C Vectorized (e32m4)", load_data("bias_add_e32m4.bin", output_shape)),
    ("C Vectorized (e32m8)", load_data("bias_add_e32m8.bin", output_shape)),
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