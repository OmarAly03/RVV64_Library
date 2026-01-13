import ctypes
import os

_lib = ctypes.CDLL(
    os.path.join(os.path.dirname(__file__), "../../libso/libmaxpool.so")
)

# ---- Signatures ----
_lib.maxpool_scalar.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
_lib.maxpool_scalar.restype = None

for name in ("e32m1", "e32m2", "e32m4", "e32m8"):
    getattr(_lib, f"maxpool_{name}").argtypes = _lib.maxpool_scalar.argtypes
    getattr(_lib, f"maxpool_{name}").restype = None

for name in ("rvv_tiled_m1", "rvv_tiled_m2", "rvv_tiled_m4", "rvv_tiled_m8"):
    func = getattr(_lib, f"maxpool_{name}")
    func.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    func.restype = None

# ---- Python-facing API ----
def maxpool_scalar(input_ptr, output_ptr, batch, channels, in_h, in_w, k_h, k_w, stride_h, stride_w, pad_h, pad_w):
    _lib.maxpool_scalar(input_ptr, output_ptr, batch, channels, in_h, in_w, k_h, k_w, stride_h, stride_w, pad_h, pad_w)

def maxpool_e32m1(input_ptr, output_ptr, batch, channels, in_h, in_w, k_h, k_w, stride_h, stride_w, pad_h, pad_w):
    _lib.maxpool_e32m1(input_ptr, output_ptr, batch, channels, in_h, in_w, k_h, k_w, stride_h, stride_w, pad_h, pad_w)

def maxpool_e32m2(input_ptr, output_ptr, batch, channels, in_h, in_w, k_h, k_w, stride_h, stride_w, pad_h, pad_w):
    _lib.maxpool_e32m2(input_ptr, output_ptr, batch, channels, in_h, in_w, k_h, k_w, stride_h, stride_w, pad_h, pad_w)

def maxpool_e32m4(input_ptr, output_ptr, batch, channels, in_h, in_w, k_h, k_w, stride_h, stride_w, pad_h, pad_w):
    _lib.maxpool_e32m4(input_ptr, output_ptr, batch, channels, in_h, in_w, k_h, k_w, stride_h, stride_w, pad_h, pad_w)

def maxpool_e32m8(input_ptr, output_ptr, batch, channels, in_h, in_w, k_h, k_w, stride_h, stride_w, pad_h, pad_w):
    _lib.maxpool_e32m8(input_ptr, output_ptr, batch, channels, in_h, in_w, k_h, k_w, stride_h, stride_w, pad_h, pad_w)

def maxpool_rvv_tiled_m1(input_ptr, output_ptr, batch, channels, in_h, in_w, k_h, k_w, stride_h, stride_w, pad_h, pad_w, tile_h, tile_w):
    _lib.maxpool_rvv_tiled_m1(input_ptr, output_ptr, batch, channels, in_h, in_w, k_h, k_w, stride_h, stride_w, pad_h, pad_w, tile_h, tile_w)

def maxpool_rvv_tiled_m2(input_ptr, output_ptr, batch, channels, in_h, in_w, k_h, k_w, stride_h, stride_w, pad_h, pad_w, tile_h, tile_w):
    _lib.maxpool_rvv_tiled_m2(input_ptr, output_ptr, batch, channels, in_h, in_w, k_h, k_w, stride_h, stride_w, pad_h, pad_w, tile_h, tile_w)

def maxpool_rvv_tiled_m4(input_ptr, output_ptr, batch, channels, in_h, in_w, k_h, k_w, stride_h, stride_w, pad_h, pad_w, tile_h, tile_w):
    _lib.maxpool_rvv_tiled_m4(input_ptr, output_ptr, batch, channels, in_h, in_w, k_h, k_w, stride_h, stride_w, pad_h, pad_w, tile_h, tile_w)

def maxpool_rvv_tiled_m8(input_ptr, output_ptr, batch, channels, in_h, in_w, k_h, k_w, stride_h, stride_w, pad_h, pad_w, tile_h, tile_w):
    _lib.maxpool_rvv_tiled_m8(input_ptr, output_ptr, batch, channels, in_h, in_w, k_h, k_w, stride_h, stride_w, pad_h, pad_w, tile_h, tile_w)
