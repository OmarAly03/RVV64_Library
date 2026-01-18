import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyv.kernels import tensor_add

print("\nTesting tensor_add (128 x 32)...")
# Create two tensors of the same shape for addition
D = np.random.randn(128, 32).astype(np.float32)
E = np.random.randn(128, 32).astype(np.float32)

add_scalar = np.add(D, E)
add_rvv = tensor_add(D, E, variant="M1")

# Verify correctness with numpy
print("\nResults match NumPy:", np.allclose(add_rvv, add_scalar))
