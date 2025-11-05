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
onnx_path = os.path.join(SCRIPT_DIR, "./output_files/softmax.onnx")
try:
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(onnx_path)
except FileNotFoundError:
    print(f"Error: ONNX model not found at {onnx_path}")
    print("Please ensure 'make' successfully ran src/onnx_softmax.py")
    sys.exit(1)


# ==== Parse Arguments ====
CHANNELS = 4
INNER_SIZE = 128

if len(sys.argv) == 3:
    try:
        CHANNELS = int(sys.argv[1])
        INNER_SIZE = int(sys.argv[2])
        if CHANNELS <= 0 or INNER_SIZE <= 0:
            raise ValueError
    except ValueError:
        print(f"Invalid arguments '{sys.argv[1]}', '{sys.argv[2]}'. Using defaults.")
        CHANNELS = 4
        INNER_SIZE = 128
else:
    print(f"Usage: python3 main.py <channels> <innerSize>. Using defaults.")

N = CHANNELS * INNER_SIZE
shape = (CHANNELS, INNER_SIZE)

# Load input data
input_path = os.path.join(SCRIPT_DIR, "./output_files/input.bin")
try:
    # Reshape the flat input data to the correct 2D shape
    input_data = np.fromfile(input_path, dtype=np.float32).reshape(shape)
except FileNotFoundError:
    print(f"Error: Input data not found at {input_path}")
    print("Please ensure QEMU simulation (run_softmax) ran successfully.")
    sys.exit(1)
except ValueError:
    print(f"Error: Input data size mismatch. Expected {N} elements for shape {shape}.")
    sys.exit(1)


print(f"\nSoftmax validation on shape {shape} (Total {N} elements)")

# ==== ONNX Golden Reference (using ONNXRuntime) ====
input_names = [input_node.name for input_node in onnx_model.graph.input]
# Feed the 2D [CHANNELS, INNER_SIZE] array to ONNX
onnx_ref = session.run(None, {input_names[0]: input_data})[0]

# ONNX --> golden reference
c_ref = onnx_ref

# ==== Load C Kernel Results ====
def load_result(filename, shape):
    path = os.path.join(SCRIPT_DIR, f"./output_files/{filename}")
    try:
        # Reshape the flat output data to the correct 2D shape
        return np.fromfile(path, dtype=np.float32).reshape(shape)
    except FileNotFoundError:
        print(f"Warning: Output file not found: {path}. Skipping.")
        return None
    except ValueError:
        print(f"Warning: Output file {filename} size mismatch. Skipping.")
        return None

implementations = [
    ("ONNX Golden Ref", onnx_ref),
    ("C Scalar", load_result("softmax_scalar.bin", shape)),
    ("C Vectorized", load_result("softmax_vector.bin", shape)),
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