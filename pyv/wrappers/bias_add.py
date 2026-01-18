import ctypes
import os

_lib = ctypes.CDLL(
    os.path.join(os.path.dirname(__file__), "../../libso/libbiasadd.so")
)

# ---- Signatures ----
_lib.bias_add_scalar.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_lib.bias_add_scalar.restype = None

for name in ("e32m1", "e32m2", "e32m4", "e32m8"):
    getattr(_lib, f"bias_add_{name}").argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    getattr(_lib, f"bias_add_{name}").restype = None

# ---- Python-facing API ----
def bias_add_scalar(input_ptr, bias_ptr, output_ptr, batch_size, channels, height, width):
    _lib.bias_add_scalar(input_ptr, bias_ptr, output_ptr, batch_size, channels, height, width)

def bias_add_e32m1(input_ptr, bias_ptr, output_ptr, channels, channel_size):
    _lib.bias_add_e32m1(input_ptr, bias_ptr, output_ptr, channels, channel_size)

def bias_add_e32m2(input_ptr, bias_ptr, output_ptr, channels, channel_size):
    _lib.bias_add_e32m2(input_ptr, bias_ptr, output_ptr, channels, channel_size)

def bias_add_e32m4(input_ptr, bias_ptr, output_ptr, channels, channel_size):
    _lib.bias_add_e32m4(input_ptr, bias_ptr, output_ptr, channels, channel_size)

def bias_add_e32m8(input_ptr, bias_ptr, output_ptr, channels, channel_size):
    _lib.bias_add_e32m8(input_ptr, bias_ptr, output_ptr, channels, channel_size)
