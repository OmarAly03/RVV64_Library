import ctypes
import os

_lib = ctypes.CDLL(
    os.path.join(os.path.dirname(__file__), "../../libso/librelu.so")
)

# ---- Signatures ----
_lib.relu_scalar.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
_lib.relu_scalar.restype = None

_lib.relu_e32m1.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
_lib.relu_e32m1.restype = None

_lib.relu_e32m2.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
_lib.relu_e32m2.restype = None

_lib.relu_e32m4.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
_lib.relu_e32m4.restype = None

_lib.relu_e32m8.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
_lib.relu_e32m8.restype = None

# ---- Python-facing API ----
def relu_scalar(inp, out, size):
    _lib.relu_scalar(inp, out, size)

def relu_e32m1(inp, out, size):
    _lib.relu_e32m1(inp, out, size)
    
def relu_e32m2(inp, out, size):
    _lib.relu_e32m2(inp, out, size)
    
def relu_e32m4(inp, out, size):
    _lib.relu_e32m4(inp, out, size)
    
def relu_e32m8(inp, out, size):
    _lib.relu_e32m8(inp, out, size)

