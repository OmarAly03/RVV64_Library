import ctypes
import os

_lib = ctypes.CDLL(
    os.path.join(os.path.dirname(__file__), "../../libso/libbatch_norm.so")
)

# ---- Signatures ----
_lib.batch_norm_scalar.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
]
_lib.batch_norm_scalar.restype = None

_lib.batch_norm_tiled_scalar.argtypes = _lib.batch_norm_scalar.argtypes
_lib.batch_norm_tiled_scalar.restype = None

for name in ("e32m1", "e32m2", "e32m4", "e32m8"):
    getattr(_lib, f"batch_norm_{name}").argtypes = _lib.batch_norm_scalar.argtypes
    getattr(_lib, f"batch_norm_{name}").restype = None

for name in ("tiled_e32m1", "tiled_e32m2", "tiled_e32m4", "tiled_e32m8"):
    func_name = f"batch_norm_{name}"
    getattr(_lib, func_name).argtypes = _lib.batch_norm_scalar.argtypes
    getattr(_lib, func_name).restype = None

# ---- Python-facing API ----
def batch_norm_scalar(input_ptr, output_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, channels, height, width, epsilon):
    _lib.batch_norm_scalar(input_ptr, output_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, channels, height, width, epsilon)

def batch_norm_tiled_scalar(input_ptr, output_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, channels, height, width, epsilon):
    _lib.batch_norm_tiled_scalar(input_ptr, output_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, channels, height, width, epsilon)

def batch_norm_e32m1(input_ptr, output_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, channels, height, width, epsilon):
    _lib.batch_norm_e32m1(input_ptr, output_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, channels, height, width, epsilon)

def batch_norm_e32m2(input_ptr, output_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, channels, height, width, epsilon):
    _lib.batch_norm_e32m2(input_ptr, output_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, channels, height, width, epsilon)

def batch_norm_e32m4(input_ptr, output_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, channels, height, width, epsilon):
    _lib.batch_norm_e32m4(input_ptr, output_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, channels, height, width, epsilon)

def batch_norm_e32m8(input_ptr, output_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, channels, height, width, epsilon):
    _lib.batch_norm_e32m8(input_ptr, output_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, channels, height, width, epsilon)

def batch_norm_tiled_e32m1(input_ptr, output_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, channels, height, width, epsilon):
    _lib.batch_norm_tiled_e32m1(input_ptr, output_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, channels, height, width, epsilon)

def batch_norm_tiled_e32m2(input_ptr, output_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, channels, height, width, epsilon):
    _lib.batch_norm_tiled_e32m2(input_ptr, output_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, channels, height, width, epsilon)

def batch_norm_tiled_e32m4(input_ptr, output_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, channels, height, width, epsilon):
    _lib.batch_norm_tiled_e32m4(input_ptr, output_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, channels, height, width, epsilon)

def batch_norm_tiled_e32m8(input_ptr, output_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, channels, height, width, epsilon):
    _lib.batch_norm_tiled_e32m8(input_ptr, output_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, channels, height, width, epsilon)
