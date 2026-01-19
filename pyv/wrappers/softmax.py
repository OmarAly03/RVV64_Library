import ctypes
import os

_lib = ctypes.CDLL(
    os.path.join(os.path.dirname(__file__), "../../libso/libsoftmax.so")
)

# ---- Signatures ----
# C: void softmax(const float* input, float* output, size_t n);
_lib.softmax.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_size_t,                 # n
]
_lib.softmax.restype = None

# ---- Python-facing API ----
def softmax(input_ptr, output_ptr, n):
    _lib.softmax(input_ptr, output_ptr, n)