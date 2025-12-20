import ctypes
import os

_lib = ctypes.CDLL(
    os.path.join(os.path.dirname(__file__), "../../libso/libtensor_add.so")
)

# ---- Signatures ----
_lib.tensor_add_scalar.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
_lib.tensor_add_scalar.restype = None

_lib.tensor_add_e32m8.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
_lib.tensor_add_e32m8.restype = None

# ---- Python-facing API ----
def tensor_add_scalar(input_a, input_b, output, size):
    _lib.tensor_add_scalar(input_a, input_b, output, size)

def tensor_add_e32m8(input_a, input_b, output, size):
    _lib.tensor_add_e32m8(input_a, input_b, output, size)