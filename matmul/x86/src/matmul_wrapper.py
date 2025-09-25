import ctypes
import numpy as np
from pathlib import Path

LIB_PATH = Path(__file__).parent / "matmul.so"
lib = ctypes.CDLL(str(LIB_PATH))

lib.matmul_scalar.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # src1
    ctypes.POINTER(ctypes.c_float),  # src2
    ctypes.POINTER(ctypes.c_float),  # dst
    ctypes.c_size_t,                 # M
    ctypes.c_size_t,                 # N
    ctypes.c_size_t                  # K
]
lib.matmul_scalar.restype = None

def matmul_scalar(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    A Pythonic wrapper around the C matmul_scalar function.

    Args:
        A (np.ndarray): 2D array of float32, shape (M, K)
        B (np.ndarray): 2D array of float32, shape (K, N)
    Returns:
        np.ndarray: Result of matrix multiplication, shape (M, N)
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Both A and B must be 2D arrays")
    if A.dtype != np.float32 or B.dtype != np.float32:
        raise TypeError("Arrays must be of dtype np.float32") 
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Matrix dimensions incompatible: A{A.shape} @ B{B.shape}")

    M, K = A.shape
    K_B, N = B.shape
    
    # Output matrix
    C = np.zeros((M, N), dtype=np.float32)
    
    # Ensuring arrays are contiguous
    A_contig = np.ascontiguousarray(A)
    B_contig = np.ascontiguousarray(B)

    lib.matmul_scalar(
        A_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, N, K
    )

    return C