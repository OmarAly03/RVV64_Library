import ctypes
import os

_lib = ctypes.CDLL(
    os.path.join(os.path.dirname(__file__), "../../libso/libconv.so")
)

# -------------------------------------------------------------------------
# 1. Standard Conv2D (Scalar + Vectorized)
# -------------------------------------------------------------------------
# void conv2d_*(const float* input, const float* kernel, float* output,
#               int batch_size, int in_channels, int out_channels,
#               int input_h, int input_w, int kernel_h, int kernel_w,
#               int stride_h, int stride_w, int pad_h, int pad_w);

_lib.conv2d_scalar.argtypes = [
    ctypes.POINTER(ctypes.c_float), # input
    ctypes.POINTER(ctypes.c_float), # kernel
    ctypes.POINTER(ctypes.c_float), # output
    ctypes.c_int, # batch_size
    ctypes.c_int, # in_channels
    ctypes.c_int, # out_channels
    ctypes.c_int, # input_h
    ctypes.c_int, # input_w
    ctypes.c_int, # kernel_h
    ctypes.c_int, # kernel_w
    ctypes.c_int, # stride_h
    ctypes.c_int, # stride_w
    ctypes.c_int, # pad_h
    ctypes.c_int  # pad_w
]
_lib.conv2d_scalar.restype = None

# Apply same signature to vectorized standard kernels
for name in ("e32m1", "e32m2", "e32m4", "e32m8"):
    try:
        func = getattr(_lib, f"conv2d_{name}")
        func.argtypes = _lib.conv2d_scalar.argtypes
        func.restype = None
    except AttributeError:
        pass

# -------------------------------------------------------------------------
# 2. Im2Col Struct (Im2Col+GEMM)
# -------------------------------------------------------------------------
# void conv2d_im2col_gemm_*(const float* input, const float* weights, const float* bias,
#                           float* output, float* col_buf, float* gemm_buf,
#                           int C, int H, int W, int M, int KH, int KW,
#                           int pad_h, int pad_w, int stride_h, int stride_w, int has_bias);

im2col_argtypes = [
    ctypes.POINTER(ctypes.c_float), # input
    ctypes.POINTER(ctypes.c_float), # weights/kernel
    ctypes.POINTER(ctypes.c_float), # bias
    ctypes.POINTER(ctypes.c_float), # output
    ctypes.POINTER(ctypes.c_float), # col_buf
    ctypes.POINTER(ctypes.c_float), # gemm_buf
    ctypes.c_int, # in_channels (C)
    ctypes.c_int, # input_h
    ctypes.c_int, # input_w
    ctypes.c_int, # out_channels (M)
    ctypes.c_int, # kernel_h
    ctypes.c_int, # kernel_w
    ctypes.c_int, # pad_h
    ctypes.c_int, # pad_w
    ctypes.c_int, # stride_h
    ctypes.c_int, # stride_w
    ctypes.c_int  # has_bias
]

for name in ("conv2d_im2col_gemm_scalar", "conv2d_im2col_gemm_vector", "conv2d_im2col_gemm_m8"):
    try:
        func = getattr(_lib, name)
        func.argtypes = im2col_argtypes
        func.restype = None
    except AttributeError:
        pass

# -------------------------------------------------------------------------
# 3. 3x3 Specialized RVV & Batched/RGB
# -------------------------------------------------------------------------
# void conv2d_3x3_m*(const float* input, const float* kernel, float* output,
#                    int H, int W, bool use_padding);

spec_3x3_argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, # H
    ctypes.c_int, # W
    ctypes.c_bool # use_padding
]

for name in ("m1", "m2", "m4", "m8"):
    # Standard 3x3
    try:
        func = getattr(_lib, f"conv2d_3x3_{name}")
        func.argtypes = spec_3x3_argtypes
        func.restype = None
    except AttributeError: pass
    
    # RGB 3x3
    try:
        func = getattr(_lib, f"conv2d_3x3_{name}_rgb")
        func.argtypes = spec_3x3_argtypes
        func.restype = None
    except AttributeError: pass

# Batched 3x3 (extra batch_rows arg)
for name in ("m2", "m4", "m8"):
    try:
        func = getattr(_lib, f"conv2d_3x3_{name}_batched")
        func.argtypes = spec_3x3_argtypes + [ctypes.c_int]
        func.restype = None
    except AttributeError: pass


# ---- Python-facing API ----

def conv2d_scalar(input_ptr, kernel_ptr, output_ptr, batch, in_c, out_c, in_h, in_w, k_h, k_w, s_h, s_w, p_h, p_w):
    _lib.conv2d_scalar(input_ptr, kernel_ptr, output_ptr, batch, in_c, out_c, in_h, in_w, k_h, k_w, s_h, s_w, p_h, p_w)

def conv2d_e32m1(input_ptr, kernel_ptr, output_ptr, batch, in_c, out_c, in_h, in_w, k_h, k_w, s_h, s_w, p_h, p_w):
    _lib.conv2d_e32m1(input_ptr, kernel_ptr, output_ptr, batch, in_c, out_c, in_h, in_w, k_h, k_w, s_h, s_w, p_h, p_w)

def conv2d_e32m2(input_ptr, kernel_ptr, output_ptr, batch, in_c, out_c, in_h, in_w, k_h, k_w, s_h, s_w, p_h, p_w):
    _lib.conv2d_e32m2(input_ptr, kernel_ptr, output_ptr, batch, in_c, out_c, in_h, in_w, k_h, k_w, s_h, s_w, p_h, p_w)

def conv2d_e32m4(input_ptr, kernel_ptr, output_ptr, batch, in_c, out_c, in_h, in_w, k_h, k_w, s_h, s_w, p_h, p_w):
    _lib.conv2d_e32m4(input_ptr, kernel_ptr, output_ptr, batch, in_c, out_c, in_h, in_w, k_h, k_w, s_h, s_w, p_h, p_w)

def conv2d_e32m8(input_ptr, kernel_ptr, output_ptr, batch, in_c, out_c, in_h, in_w, k_h, k_w, s_h, s_w, p_h, p_w):
    _lib.conv2d_e32m8(input_ptr, kernel_ptr, output_ptr, batch, in_c, out_c, in_h, in_w, k_h, k_w, s_h, s_w, p_h, p_w)

def conv2d_im2col_gemm_m8(input_ptr, kernel_ptr, bias_ptr, output_ptr, col_buf_ptr, gemm_buf_ptr, in_c, in_h, in_w, out_c, k_h, k_w, p_h, p_w, s_h, s_w, has_bias):
    _lib.conv2d_im2col_gemm_m8(input_ptr, kernel_ptr, bias_ptr, output_ptr, col_buf_ptr, gemm_buf_ptr, in_c, in_h, in_w, out_c, k_h, k_w, p_h, p_w, s_h, s_w, has_bias)

# 3x3 Specialized Wrappers
def conv2d_3x3_m1(input_ptr, kernel_ptr, output_ptr, h, w, use_padding):
    _lib.conv2d_3x3_m1(input_ptr, kernel_ptr, output_ptr, h, w, use_padding)

def conv2d_3x3_m2(input_ptr, kernel_ptr, output_ptr, h, w, use_padding):
    _lib.conv2d_3x3_m2(input_ptr, kernel_ptr, output_ptr, h, w, use_padding)

def conv2d_3x3_m4(input_ptr, kernel_ptr, output_ptr, h, w, use_padding):
    _lib.conv2d_3x3_m4(input_ptr, kernel_ptr, output_ptr, h, w, use_padding)

def conv2d_3x3_m8(input_ptr, kernel_ptr, output_ptr, h, w, use_padding):
    _lib.conv2d_3x3_m8(input_ptr, kernel_ptr, output_ptr, h, w, use_padding)