import ctypes
import os

_lib = ctypes.CDLL(
    os.path.join(os.path.dirname(__file__), "../../libso/libleakyrelu.so")
)

# ---- Signatures ----
_lib.leaky_relu_scalar.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_float,
]
_lib.leaky_relu_scalar.restype = None

for name in ("e32m1", "e32m2", "e32m4", "e32m8"):
    getattr(_lib, f"leaky_relu_{name}").argtypes = _lib.leaky_relu_scalar.argtypes
    getattr(_lib, f"leaky_relu_{name}").restype = None

_lib.leaky_relu_tiled_scalar.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_float,
    ctypes.c_size_t,
]
_lib.leaky_relu_tiled_scalar.restype = None

for name in ("tiled_e32m1", "tiled_e32m2", "tiled_e32m4", "tiled_e32m8"):
    getattr(_lib, f"leaky_relu_{name}").argtypes = _lib.leaky_relu_tiled_scalar.argtypes
    getattr(_lib, f"leaky_relu_{name}").restype = None

# ---- Python-facing API ----
def leaky_relu_scalar(src_ptr, dest_ptr, n, alpha):
    _lib.leaky_relu_scalar(src_ptr, dest_ptr, n, alpha)

def leaky_relu_e32m1(src_ptr, dest_ptr, n, alpha):
    _lib.leaky_relu_e32m1(src_ptr, dest_ptr, n, alpha)

def leaky_relu_e32m2(src_ptr, dest_ptr, n, alpha):
    _lib.leaky_relu_e32m2(src_ptr, dest_ptr, n, alpha)

def leaky_relu_e32m4(src_ptr, dest_ptr, n, alpha):
    _lib.leaky_relu_e32m4(src_ptr, dest_ptr, n, alpha)

def leaky_relu_e32m8(src_ptr, dest_ptr, n, alpha):
    _lib.leaky_relu_e32m8(src_ptr, dest_ptr, n, alpha)

def leaky_relu_tiled_scalar(input_ptr, output_ptr, size, alpha, tile_size):
    _lib.leaky_relu_tiled_scalar(input_ptr, output_ptr, size, alpha, tile_size)

def leaky_relu_tiled_e32m1(input_ptr, output_ptr, size, alpha, tile_size):
    _lib.leaky_relu_tiled_e32m1(input_ptr, output_ptr, size, alpha, tile_size)

def leaky_relu_tiled_e32m2(input_ptr, output_ptr, size, alpha, tile_size):
    _lib.leaky_relu_tiled_e32m2(input_ptr, output_ptr, size, alpha, tile_size)

def leaky_relu_tiled_e32m4(input_ptr, output_ptr, size, alpha, tile_size):
    _lib.leaky_relu_tiled_e32m4(input_ptr, output_ptr, size, alpha, tile_size)

def leaky_relu_tiled_e32m8(input_ptr, output_ptr, size, alpha, tile_size):
    _lib.leaky_relu_tiled_e32m8(input_ptr, output_ptr, size, alpha, tile_size)
