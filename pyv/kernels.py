import numpy as np
from .backend import ptr_f32
from .wrappers import relu as relu_wrapper
from .wrappers import matmul as matmul_wrapper
from .wrappers import tensor_add as tensor_add_wrapper

def relu(x: np.ndarray, variant="rvv"):
    assert x.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"]

    y = np.zeros_like(x)
    size = x.size

    if variant == "scalar":
        relu_wrapper.relu_scalar(ptr_f32(x), ptr_f32(y), size)
    elif variant == "rvv":
        relu_wrapper.relu_e32m8(ptr_f32(x), ptr_f32(y), size)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return y

def matmul(A: np.ndarray, B: np.ndarray, variant="rvv"):
    assert A.dtype == np.float32
    assert B.dtype == np.float32
    assert A.flags["C_CONTIGUOUS"]
    assert B.flags["C_CONTIGUOUS"]
    
    # Assuming A is M x K and B is K x N
    assert A.ndim == 2 and B.ndim == 2
    assert A.shape[1] == B.shape[0]  # K dimension must match
    
    M, K = A.shape
    K_B, N = B.shape
    
    C = np.zeros((M, N), dtype=np.float32)

    if variant == "scalar":
        matmul_wrapper.matmul_scalar(ptr_f32(A), ptr_f32(B), ptr_f32(C), M, N, K)
    elif variant == "rvv":
        matmul_wrapper.matmul_e32m8(ptr_f32(A), ptr_f32(B), ptr_f32(C), M, N, K)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return C

def tensor_add(A: np.ndarray, B: np.ndarray, variant="rvv"):
    assert A.dtype == np.float32
    assert B.dtype == np.float32
    assert A.flags["C_CONTIGUOUS"]
    assert B.flags["C_CONTIGUOUS"]
    assert A.shape == B.shape  # Tensors must have the same shape
    
    C = np.zeros_like(A)
    size = A.size
    
    if variant == "scalar":
        tensor_add_wrapper.tensor_add_scalar(ptr_f32(A), ptr_f32(B), ptr_f32(C), size)
    elif variant == "rvv":
        tensor_add_wrapper.tensor_add_e32m8(ptr_f32(A), ptr_f32(B), ptr_f32(C), size)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    return C