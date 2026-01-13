import ctypes
import os

_lib = ctypes.CDLL(
    os.path.join(os.path.dirname(__file__), "../../libso/libdense.so")
)

# ---- Signatures ----
_lib.dense_scalar.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_lib.dense_scalar.restype = None

for name in ("e32m1", "e32m2", "e32m4", "e32m8"):
    getattr(_lib, f"dense_{name}").argtypes = _lib.dense_scalar.argtypes
    getattr(_lib, f"dense_{name}").restype = None

# ---- Python-facing API ----
def dense_scalar(input_ptr, weights_ptr, bias_ptr, output_ptr, in_features, out_features):
    _lib.dense_scalar(input_ptr, weights_ptr, bias_ptr, output_ptr, in_features, out_features)

def dense_e32m1(input_ptr, weights_ptr, bias_ptr, output_ptr, in_features, out_features):
    _lib.dense_e32m1(input_ptr, weights_ptr, bias_ptr, output_ptr, in_features, out_features)

def dense_e32m2(input_ptr, weights_ptr, bias_ptr, output_ptr, in_features, out_features):
    _lib.dense_e32m2(input_ptr, weights_ptr, bias_ptr, output_ptr, in_features, out_features)

def dense_e32m4(input_ptr, weights_ptr, bias_ptr, output_ptr, in_features, out_features):
    _lib.dense_e32m4(input_ptr, weights_ptr, bias_ptr, output_ptr, in_features, out_features)

def dense_e32m8(input_ptr, weights_ptr, bias_ptr, output_ptr, in_features, out_features):
    _lib.dense_e32m8(input_ptr, weights_ptr, bias_ptr, output_ptr, in_features, out_features)
