import ctypes
import os

_lib = ctypes.CDLL(
    os.path.join(os.path.dirname(__file__), "../../libso/libmatmul.so")
)

# ---- Signatures ----
_lib.matmul_scalar.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_lib.matmul_scalar.restype = None

_lib.matmul_e32m1.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_lib.matmul_e32m1.restype = None

_lib.matmul_e32m2.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_lib.matmul_e32m2.restype = None

_lib.matmul_e32m4.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_lib.matmul_e32m4.restype = None

_lib.matmul_e32m8.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_lib.matmul_e32m8.restype = None

# ---- Python-facing API ----
def matmul_scalar(A, B, C, M, N, K):
    _lib.matmul_scalar(A, B, C, M, N, K)

def matmul_e32m1(A, B, C, M, N, K):
    _lib.matmul_e32m1(A, B, C, M, N, K)
    
def matmul_e32m2(A, B, C, M, N, K):
    _lib.matmul_e32m2(A, B, C, M, N, K)
    
def matmul_e32m4(A, B, C, M, N, K):
    _lib.matmul_e32m4(A, B, C, M, N, K)
    
def matmul_e32m8(A, B, C, M, N, K):
    _lib.matmul_e32m8(A, B, C, M, N, K)

