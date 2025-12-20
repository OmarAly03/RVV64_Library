import ctypes
import numpy as np

def ptr_f32(arr: np.ndarray):
    assert arr.dtype == np.float32
    assert arr.flags["C_CONTIGUOUS"]
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
