import numpy as np
import onnx
import onnxruntime as ort
from src.onnx_utils import max_abs_error, snr_db
from src.relu_scalar import relu_py_scalar
from src.relu_wrapper import relu_scalar

# ==== Load ONNX Model ====
onnx_model = onnx.load("relu.onnx")
onnx.checker.check_model(onnx_model)
session = ort.InferenceSession("relu.onnx")
print("ONNX model loaded and validated successfully")
print(f"Model inputs: {[input.name for input in onnx_model.graph.input]}")
print(f"Model outputs: {[output.name for output in onnx_model.graph.output]}")

# ==== Input Size ====
N = 1024 * 1024  

np.random.seed(42)  # For reproducible results
# Data with both positive and negative values to test ReLU
x_np = (np.random.rand(N).astype(np.float32) - 0.5) * 10.0  # Range [-5, 5]

print(f"\nReLU activation: Input tensor size: {N} elements")
print(f"Input range: [{np.min(x_np):.3f}, {np.max(x_np):.3f}]")

# ==== ONNX Golden Reference (using ONNXRuntime) ====
input_names = [input.name for input in onnx_model.graph.input]
onnx_ref = session.run(None, {input_names[0]: x_np})[0]
print(f"ONNX reference shape: {onnx_ref.shape}")

# ==== Python Scalar ====
py_scalar = relu_py_scalar(x_np)

# ==== Python NumPy ====
py_numpy = np.maximum(0, x_np)

# ==== C Scalar ====
c_scalar = relu_scalar(x_np)

# Use ONNX as golden reference
c_ref = onnx_ref

# ==== Results Table ====
implementations = [
    ("ONNX Golden Ref", onnx_ref),
    ("Python Scalar", py_scalar),
    ("NumPy maximum", py_numpy),
    ("C Scalar", c_scalar),  
]

print(f"\n{'Implementation':<20}{'Max Abs Error':<20}{'SNR (dB)':<20}")
print("-" * 60)

for name, result in implementations:
    mae = max_abs_error(c_ref, result)
    snr = snr_db(c_ref, result)
    print(f"{name:<20}{mae:<20.6g}{snr:<20.6g}")

# ==== Verification against ONNX Golden Reference ====
print(f"\nVerification against ONNX Golden Reference:")
print(f"NumPy maximum matches ONNX: {np.allclose(onnx_ref, py_numpy, rtol=1e-6, atol=1e-6)}")
print(f"Python scalar matches ONNX: {np.allclose(onnx_ref, py_scalar, rtol=1e-5, atol=1e-6)}")
print(f"C scalar matches ONNX:      {np.allclose(onnx_ref, c_scalar, rtol=1e-5, atol=1e-6)}")
