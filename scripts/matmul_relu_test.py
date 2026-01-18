import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyv.kernels import matmul, relu

# Create test matrices
A = np.random.randn(128, 64).astype(np.float32)
B = np.random.randn(64, 32).astype(np.float32)

print("Matrix A shape:", A.shape)
print("Matrix B shape:", B.shape)

# Test matmul with both variants
print("\nTesting matmul...")
C_scalar = matmul(A, B, variant="scalar")

C_rvv = matmul(A, B, variant="M1")

print("Result shape:", C_scalar.shape)
print("Results match:", np.allclose(C_scalar, C_rvv, atol=1e-5))


# Apply ReLU to the matmul result
print("\nTesting relu on matmul result...")

C_with_negatives = C_scalar - 0.5

relu_scalar = relu(C_with_negatives, variant="scalar")

relu_rvv = relu(C_with_negatives, variant="M1")

print("ReLU results match:", np.allclose(relu_scalar, relu_rvv))
print("ReLU output shape:", relu_scalar.shape)
