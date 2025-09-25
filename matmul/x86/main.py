import numpy as np
import onnx
import onnxruntime as ort
from src.onnx_utils import max_abs_error, snr_db
from src.matmult import matmul_py_scalar
from src.matmul_wrapper import matmul_scalar

# ==== Loading ONNX Model ====
onnx_model = onnx.load("matrix_multiply.onnx")
onnx.checker.check_model(onnx_model)
session = ort.InferenceSession("matrix_multiply.onnx")
print("ONNX model loaded and validated successfully")
print(f"Model inputs: {[input.name for input in onnx_model.graph.input]}")
print(f"Model outputs: {[output.name for output in onnx_model.graph.output]}")

# ==== Input Size ====
M, K, N = 128, 64, 128  

np.random.seed(42)  # For reproducible results
a_np = np.random.rand(M, K).astype(np.float32)
b_np = np.random.rand(K, N).astype(np.float32)

print(f"\nMatrix multiplication: A({M}x{K}) @ B({K}x{N}) -> C({M}x{N})")
print(f"Total operations: {2 * M * K * N:,} FLOPs")

# ==== ONNX Golden Reference (using ONNXRuntime) ====
input_names = [input.name for input in onnx_model.graph.input]
onnx_ref = session.run(None, {input_names[0]: a_np, input_names[1]: b_np})[0]
print(f"ONNX reference shape: {onnx_ref.shape}")

# ==== Python Scalar ====
py_scalar = matmul_py_scalar(a_np, b_np)

# ==== Python NumPy dot ====
py_numpy = np.dot(a_np, b_np)

# ==== Python NumPy @ operator ====
py_at_operator = a_np @ b_np

# ==== C Scalar ====
c_scalar = matmul_scalar(a_np, b_np)

# ONNX --> golden reference
c_ref = onnx_ref

# ==== Results Table ====
implementations = [
    ("ONNX Golden Ref", onnx_ref),
    ("Python Scalar", py_scalar),
    ("NumPy dot", py_numpy),
    ("NumPy @ operator", py_at_operator),
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
print(f"NumPy @ matches ONNX:     {np.allclose(onnx_ref, py_at_operator, rtol=1e-6, atol=1e-6)}")
print(f"NumPy dot matches ONNX:   {np.allclose(onnx_ref, py_numpy, rtol=1e-6, atol=1e-6)}")
print(f"Python scalar matches ONNX: {np.allclose(onnx_ref, py_scalar, rtol=1e-5, atol=1e-6)}")
print(f"C scalar matches ONNX:    {np.allclose(onnx_ref, c_scalar, rtol=1e-5, atol=1e-6)}")