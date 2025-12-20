import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyv.kernels import tensor_add

# Test tensor_add
print("\nTesting tensor_add...")
# Create two tensors of the same shape for addition
D = np.random.randn(128, 32).astype(np.float32)
E = np.random.randn(128, 32).astype(np.float32)

print("Tensor D shape:", D.shape)
print("Tensor E shape:", E.shape)

add_scalar = tensor_add(D, E, variant="scalar")
add_rvv = tensor_add(D, E, variant="rvv")

print("Tensor add results match:", np.allclose(add_scalar, add_rvv))
print("Tensor add output shape:", add_scalar.shape)

# Verify correctness with numpy
numpy_add = D + E
print("Results match NumPy:", np.allclose(add_scalar, numpy_add))
