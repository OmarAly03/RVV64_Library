import ctypes
import os

_lib = ctypes.CDLL(
    os.path.join(os.path.dirname(__file__), "../../libso/libconv_transpose.so")
)

# ---- Signatures ----
_lib.conv2d_transpose_scalar.argtypes = [
    ctypes.POINTER(ctypes.c_float),
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
_lib.conv2d_transpose_scalar.restype = None

for name in ("e32m1", "e32m2", "e32m4", "e32m8"):
    getattr(_lib, f"conv2d_transpose_{name}").argtypes = _lib.conv2d_transpose_scalar.argtypes
    getattr(_lib, f"conv2d_transpose_{name}").restype = None

for name in ("3x3_rvv_m1", "3x3_rvv_m2", "3x3_rvv_m4", "3x3_rvv_m8"):
    func = getattr(_lib, f"conv2d_transpose_{name}")
    func.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    func.restype = None

# ---- Python-facing API ----
def conv2d_transpose_scalar(input_ptr, kernel_ptr, output_ptr, batch_size, in_channels, out_channels, input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w):
    _lib.conv2d_transpose_scalar(input_ptr, kernel_ptr, output_ptr, batch_size, in_channels, out_channels, input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)

def conv2d_transpose_e32m1(input_ptr, kernel_ptr, output_ptr, batch_size, in_channels, out_channels, input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w):
    _lib.conv2d_transpose_e32m1(input_ptr, kernel_ptr, output_ptr, batch_size, in_channels, out_channels, input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)

def conv2d_transpose_e32m2(input_ptr, kernel_ptr, output_ptr, batch_size, in_channels, out_channels, input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w):
    _lib.conv2d_transpose_e32m2(input_ptr, kernel_ptr, output_ptr, batch_size, in_channels, out_channels, input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)

def conv2d_transpose_e32m4(input_ptr, kernel_ptr, output_ptr, batch_size, in_channels, out_channels, input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w):
    _lib.conv2d_transpose_e32m4(input_ptr, kernel_ptr, output_ptr, batch_size, in_channels, out_channels, input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)

def conv2d_transpose_e32m8(input_ptr, kernel_ptr, output_ptr, batch_size, in_channels, out_channels, input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w):
    _lib.conv2d_transpose_e32m8(input_ptr, kernel_ptr, output_ptr, batch_size, in_channels, out_channels, input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)

def conv2d_transpose_3x3_rvv_m1(input_ptr, kernel_ptr, output_ptr, in_channels, in_h, in_w, out_channels, stride_h, stride_w):
    _lib.conv2d_transpose_3x3_rvv_m1(input_ptr, kernel_ptr, output_ptr, in_channels, in_h, in_w, out_channels, stride_h, stride_w)

def conv2d_transpose_3x3_rvv_m2(input_ptr, kernel_ptr, output_ptr, in_channels, in_h, in_w, out_channels, stride_h, stride_w):
    _lib.conv2d_transpose_3x3_rvv_m2(input_ptr, kernel_ptr, output_ptr, in_channels, in_h, in_w, out_channels, stride_h, stride_w)

def conv2d_transpose_3x3_rvv_m4(input_ptr, kernel_ptr, output_ptr, in_channels, in_h, in_w, out_channels, stride_h, stride_w):
    _lib.conv2d_transpose_3x3_rvv_m4(input_ptr, kernel_ptr, output_ptr, in_channels, in_h, in_w, out_channels, stride_h, stride_w)

def conv2d_transpose_3x3_rvv_m8(input_ptr, kernel_ptr, output_ptr, in_channels, in_h, in_w, out_channels, stride_h, stride_w):
    _lib.conv2d_transpose_3x3_rvv_m8(input_ptr, kernel_ptr, output_ptr, in_channels, in_h, in_w, out_channels, stride_h, stride_w)
