import ctypes
import numpy as np
from pathlib import Path

LIB_PATH = Path(__file__).parent / "relu.so"
lib = ctypes.CDLL(str(LIB_PATH))

lib.relu_scalar.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_size_t,                 # size
]
lib.relu_scalar.restype = None

def relu_scalar(x: np.ndarray) -> np.ndarray:
    """
    A Pythonic wrapper around the C relu_scalar function.

    Args:
        x (np.ndarray): Input tensor of dtype np.float32
    Returns:
        np.ndarray: Result of ReLU activation, same shape as input
    """
    if x.dtype != np.float32:
        raise TypeError("Array must be of dtype np.float32")
    
	# Reshape (Flatten)
    original_shape = x.shape
    x_flat = x.flatten()
    size = len(x_flat)
    
    # Output array
    y = np.zeros(size, dtype=np.float32)
    
    # Ensuring input array is contiguous
    x_contig = np.ascontiguousarray(x_flat)

    lib.relu_scalar(
        x_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        size
    )

    # Reshape back to original shape
    return y.reshape(original_shape)