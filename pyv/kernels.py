import numpy as np
from .backend import ptr_f32
from .wrappers import relu as relu_wrapper
from .wrappers import matmul as matmul_wrapper
from .wrappers import tensor_add as tensor_add_wrapper
from .wrappers import batch_norm as batch_norm_wrapper
from .wrappers import bias_add as bias_add_wrapper
from .wrappers import conv_transpose as conv_transpose_wrapper
from .wrappers import dense as dense_wrapper
from .wrappers import leaky_relu as leaky_relu_wrapper
from .wrappers import maxpool as maxpool_wrapper
from .wrappers import conv as conv_wrapper
from .wrappers import softmax as softmax_wrapper 

def relu(x: np.ndarray, variant="rvv"):
    assert x.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"]

    y = np.zeros_like(x)
    size = x.size

    if variant == "scalar":
        relu_wrapper.relu_scalar(ptr_f32(x), ptr_f32(y), size)
    elif variant == "M1":
        relu_wrapper.relu_e32m1(ptr_f32(x), ptr_f32(y), size)
    elif variant == "M2":
        relu_wrapper.relu_e32m2(ptr_f32(x), ptr_f32(y), size)
    elif variant == "M4":
        relu_wrapper.relu_e32m4(ptr_f32(x), ptr_f32(y), size)
    elif variant == "M8":
        relu_wrapper.relu_e32m8(ptr_f32(x), ptr_f32(y), size)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return y

def matmul(A: np.ndarray, B: np.ndarray, variant="rvv"):
    assert A.dtype == np.float32
    assert B.dtype == np.float32
    assert A.flags["C_CONTIGUOUS"]
    assert B.flags["C_CONTIGUOUS"]
    
    # Assuming A is M x K and B is K x N
    assert A.ndim == 2 and B.ndim == 2
    assert A.shape[1] == B.shape[0]  # K dimension must match
    
    M, K = A.shape
    K_B, N = B.shape
    
    C = np.zeros((M, N), dtype=np.float32)

    if variant == "scalar":
        matmul_wrapper.matmul_scalar(ptr_f32(A), ptr_f32(B), ptr_f32(C), M, N, K)
    elif variant == "M1":
        matmul_wrapper.matmul_e32m1(ptr_f32(A), ptr_f32(B), ptr_f32(C), M, N, K)
    elif variant == "M2":
        matmul_wrapper.matmul_e32m2(ptr_f32(A), ptr_f32(B), ptr_f32(C), M, N, K)
    elif variant == "M4":
        matmul_wrapper.matmul_e32m4(ptr_f32(A), ptr_f32(B), ptr_f32(C), M, N, K)
    elif variant == "M8":
        matmul_wrapper.matmul_e32m8(ptr_f32(A), ptr_f32(B), ptr_f32(C), M, N, K)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return C

def tensor_add(A: np.ndarray, B: np.ndarray, variant="rvv"):
    assert A.dtype == np.float32
    assert B.dtype == np.float32
    assert A.flags["C_CONTIGUOUS"]
    assert B.flags["C_CONTIGUOUS"]
    assert A.shape == B.shape  # Tensors must have the same shape
    
    C = np.zeros_like(A)
    size = A.size
    
    if variant == "scalar":
        tensor_add_wrapper.tensor_add_scalar(ptr_f32(A), ptr_f32(B), ptr_f32(C), size)
    elif variant == "M1":
        tensor_add_wrapper.tensor_add_e32m1(ptr_f32(A), ptr_f32(B), ptr_f32(C), size)
    elif variant == "M2":
        tensor_add_wrapper.tensor_add_e32m2(ptr_f32(A), ptr_f32(B), ptr_f32(C), size)
    elif variant == "M4":
        tensor_add_wrapper.tensor_add_e32m4(ptr_f32(A), ptr_f32(B), ptr_f32(C), size)
    elif variant == "M8":
        tensor_add_wrapper.tensor_add_e32m8(ptr_f32(A), ptr_f32(B), ptr_f32(C), size)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    return C

def batch_norm(x: np.ndarray, scale: np.ndarray, bias: np.ndarray, mean: np.ndarray, variance: np.ndarray, epsilon: float = 1e-5, variant="rvv"):
    assert x.dtype == np.float32
    assert scale.dtype == np.float32
    assert bias.dtype == np.float32
    assert mean.dtype == np.float32
    assert variance.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"]

    # Expect N,C,H,W
    assert x.ndim == 4
    N, C, H, W = x.shape

    y = np.zeros_like(x)

    if variant == "scalar":
        batch_norm_wrapper.batch_norm_scalar(ptr_f32(x), ptr_f32(y), ptr_f32(scale), ptr_f32(bias), ptr_f32(mean), ptr_f32(variance), C, H, W, epsilon)
    elif variant == "tiled_scalar":
        batch_norm_wrapper.batch_norm_tiled_scalar(ptr_f32(x), ptr_f32(y), ptr_f32(scale), ptr_f32(bias), ptr_f32(mean), ptr_f32(variance), C, H, W, epsilon)
    elif variant == "M1":
        batch_norm_wrapper.batch_norm_e32m1(ptr_f32(x), ptr_f32(y), ptr_f32(scale), ptr_f32(bias), ptr_f32(mean), ptr_f32(variance), C, H, W, epsilon)
    elif variant == "M2":
        batch_norm_wrapper.batch_norm_e32m2(ptr_f32(x), ptr_f32(y), ptr_f32(scale), ptr_f32(bias), ptr_f32(mean), ptr_f32(variance), C, H, W, epsilon)
    elif variant == "M4":
        batch_norm_wrapper.batch_norm_e32m4(ptr_f32(x), ptr_f32(y), ptr_f32(scale), ptr_f32(bias), ptr_f32(mean), ptr_f32(variance), C, H, W, epsilon)
    elif variant == "M8":
        batch_norm_wrapper.batch_norm_e32m8(ptr_f32(x), ptr_f32(y), ptr_f32(scale), ptr_f32(bias), ptr_f32(mean), ptr_f32(variance), C, H, W, epsilon)
    elif variant == "tiled_M1":
        batch_norm_wrapper.batch_norm_tiled_e32m1(ptr_f32(x), ptr_f32(y), ptr_f32(scale), ptr_f32(bias), ptr_f32(mean), ptr_f32(variance), C, H, W, epsilon)
    elif variant == "tiled_M2":
        batch_norm_wrapper.batch_norm_tiled_e32m2(ptr_f32(x), ptr_f32(y), ptr_f32(scale), ptr_f32(bias), ptr_f32(mean), ptr_f32(variance), C, H, W, epsilon)
    elif variant == "tiled_M4":
        batch_norm_wrapper.batch_norm_tiled_e32m4(ptr_f32(x), ptr_f32(y), ptr_f32(scale), ptr_f32(bias), ptr_f32(mean), ptr_f32(variance), C, H, W, epsilon)
    elif variant == "tiled_M8":
        batch_norm_wrapper.batch_norm_tiled_e32m8(ptr_f32(x), ptr_f32(y), ptr_f32(scale), ptr_f32(bias), ptr_f32(mean), ptr_f32(variance), C, H, W, epsilon)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return y

def bias_add(input: np.ndarray, bias: np.ndarray, variant="rvv"):
    assert input.dtype == np.float32
    assert bias.dtype == np.float32
    assert input.flags["C_CONTIGUOUS"]

    # Expect N,C,H,W
    assert input.ndim == 4
    N, C, H, W = input.shape

    out = np.zeros_like(input)

    if variant == "scalar":
        bias_add_wrapper.bias_add_scalar(ptr_f32(input), ptr_f32(bias), ptr_f32(out), N, C, H, W)
    elif variant == "M1":
        channel_size = H * W
        bias_add_wrapper.bias_add_e32m1(ptr_f32(input), ptr_f32(bias), ptr_f32(out), C, channel_size)
    elif variant == "M2":
        channel_size = H * W
        bias_add_wrapper.bias_add_e32m2(ptr_f32(input), ptr_f32(bias), ptr_f32(out), C, channel_size)
    elif variant == "M4":
        channel_size = H * W
        bias_add_wrapper.bias_add_e32m4(ptr_f32(input), ptr_f32(bias), ptr_f32(out), C, channel_size)
    elif variant == "M8":
        channel_size = H * W
        bias_add_wrapper.bias_add_e32m8(ptr_f32(input), ptr_f32(bias), ptr_f32(out), C, channel_size)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return out

def conv_transpose(input: np.ndarray, kernel: np.ndarray, stride=(1,1), pad=(0,0), variant="rvv"):
    assert input.dtype == np.float32
    assert kernel.dtype == np.float32
    assert input.flags["C_CONTIGUOUS"]
    assert kernel.flags["C_CONTIGUOUS"]

    # input: N, C_in, H, W
    assert input.ndim == 4 and kernel.ndim == 4
    N, C_in, H, W = input.shape
    k0, k1, kH, kW = kernel.shape

    # Try to infer in/out channels from kernel layout
    if k0 == C_in:
        in_channels = k0
        out_channels = k1
    elif k1 == C_in:
        in_channels = k1
        out_channels = k0
    else:
        in_channels = C_in
        out_channels = k1

    stride_h, stride_w = stride
    pad_h, pad_w = pad

    out_h = (H - 1) * stride_h - 2 * pad_h + kH
    out_w = (W - 1) * stride_w - 2 * pad_w + kW

    out = np.zeros((N, out_channels, out_h, out_w), dtype=np.float32)

    if variant == "scalar":
        conv_transpose_wrapper.conv2d_transpose_scalar(ptr_f32(input), ptr_f32(kernel), ptr_f32(out), N, in_channels, out_channels, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w)
    elif variant == "M1":
        if kH == 3 and kW == 3:
            conv_transpose_wrapper.conv2d_transpose_3x3_rvv_m1(ptr_f32(input), ptr_f32(kernel), ptr_f32(out), in_channels, H, W, out_channels, stride_h, stride_w)
        else:
            conv_transpose_wrapper.conv2d_transpose_e32m1(ptr_f32(input), ptr_f32(kernel), ptr_f32(out), N, in_channels, out_channels, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w)
    elif variant == "M2":
        if kH == 3 and kW == 3:
            conv_transpose_wrapper.conv2d_transpose_3x3_rvv_m2(ptr_f32(input), ptr_f32(kernel), ptr_f32(out), in_channels, H, W, out_channels, stride_h, stride_w)
        else:
            conv_transpose_wrapper.conv2d_transpose_e32m2(ptr_f32(input), ptr_f32(kernel), ptr_f32(out), N, in_channels, out_channels, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w)
    elif variant == "M4":
        if kH == 3 and kW == 3:
            conv_transpose_wrapper.conv2d_transpose_3x3_rvv_m4(ptr_f32(input), ptr_f32(kernel), ptr_f32(out), in_channels, H, W, out_channels, stride_h, stride_w)
        else:
            conv_transpose_wrapper.conv2d_transpose_e32m4(ptr_f32(input), ptr_f32(kernel), ptr_f32(out), N, in_channels, out_channels, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w)
    elif variant == "M8":
        if kH == 3 and kW == 3:
            conv_transpose_wrapper.conv2d_transpose_3x3_rvv_m8(ptr_f32(input), ptr_f32(kernel), ptr_f32(out), in_channels, H, W, out_channels, stride_h, stride_w)
        else:
            conv_transpose_wrapper.conv2d_transpose_e32m8(ptr_f32(input), ptr_f32(kernel), ptr_f32(out), N, in_channels, out_channels, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return out

# def dense(x: np.ndarray, weights: np.ndarray, bias: np.ndarray = None, variant="rvv"):
#     assert x.dtype == np.float32
#     assert weights.dtype == np.float32
#     assert x.flags["C_CONTIGUOUS"]
#     assert weights.flags["C_CONTIGUOUS"]

#     # x: (in_features,) or (batch, in_features)
#     if x.ndim == 1:
#         in_features = x.shape[0]
#         out_features = weights.shape[1]
#         out = np.zeros((out_features,), dtype=np.float32)
#         if bias is None:
#             bias = np.zeros((out_features,), dtype=np.float32)
#         if variant == "scalar":
#             dense_wrapper.dense_scalar(ptr_f32(x), ptr_f32(weights), ptr_f32(bias), ptr_f32(out), in_features, out_features)
#         elif variant == "M1":
#             dense_wrapper.dense_e32m1(ptr_f32(x), ptr_f32(weights), ptr_f32(bias), ptr_f32(out), in_features, out_features)
#         elif variant == "M2":
#             dense_wrapper.dense_e32m2(ptr_f32(x), ptr_f32(weights), ptr_f32(bias), ptr_f32(out), in_features, out_features)
#         elif variant == "M4":
#             dense_wrapper.dense_e32m4(ptr_f32(x), ptr_f32(weights), ptr_f32(bias), ptr_f32(out), in_features, out_features)
#         elif variant == "M8":
#             dense_wrapper.dense_e32m8(ptr_f32(x), ptr_f32(weights), ptr_f32(bias), ptr_f32(out), in_features, out_features)
#         else:
#             raise ValueError(f"Unknown variant: {variant}")
#         return out
#     else:
#         raise ValueError("Only 1-D input supported for dense() in this wrapper")

def dense(x: np.ndarray, weights: np.ndarray, bias: np.ndarray, variant="M8"):
    """
    x: (in_features,)
    weights: (out_features, in_features)
    bias: (out_features,)
    """
    assert x.ndim == 1, f"x must be 1D, got {x.shape}"
    assert weights.ndim == 2, f"weights must be 2D, got {weights.shape}"
    assert bias.ndim == 1, f"bias must be 1D, got {bias.shape}"

    assert x.dtype == np.float32
    assert weights.dtype == np.float32
    assert bias.dtype == np.float32

    assert x.flags["C_CONTIGUOUS"]
    assert weights.flags["C_CONTIGUOUS"]
    assert bias.flags["C_CONTIGUOUS"]

    out_features, in_features = weights.shape
    assert x.shape[0] == in_features, f"x has {x.shape[0]} elems, expected {in_features}"
    assert bias.shape[0] == out_features, f"bias has {bias.shape[0]}, expected {out_features}"

    y = np.zeros((out_features,), dtype=np.float32)

    if variant == "scalar":
        dense_wrapper.dense_scalar(
            ptr_f32(x),
            ptr_f32(weights),
            ptr_f32(bias),
            ptr_f32(y),
            in_features,
            out_features,
        )
    elif variant == "M1":
        dense_wrapper.dense_e32m1(
            ptr_f32(x),
            ptr_f32(weights),
            ptr_f32(bias),
            ptr_f32(y),
            in_features,
            out_features,
        )
    elif variant == "M2":
        dense_wrapper.dense_e32m2(
            ptr_f32(x),
            ptr_f32(weights),
            ptr_f32(bias),
            ptr_f32(y),
            in_features,
            out_features,
        )
    elif variant == "M4":
        dense_wrapper.dense_e32m4(
            ptr_f32(x),
            ptr_f32(weights),
            ptr_f32(bias),
            ptr_f32(y),
            in_features,
            out_features,
        )
    elif variant == "M8":
        dense_wrapper.dense_e32m8(
            ptr_f32(x),
            ptr_f32(weights),
            ptr_f32(bias),
            ptr_f32(y),
            in_features,
            out_features,
        )
    else:
        raise ValueError(f"Unknown dense variant: {variant}")

    return y

def leaky_relu(x: np.ndarray, alpha: float = 0.01, variant="rvv"):
    assert x.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"]

    y = np.zeros_like(x)
    size = x.size

    if variant == "scalar":
        leaky_relu_wrapper.leaky_relu_scalar(ptr_f32(x), ptr_f32(y), size, alpha)
    elif variant == "M1":
        leaky_relu_wrapper.leaky_relu_e32m1(ptr_f32(x), ptr_f32(y), size, alpha)
    elif variant == "M2":
        leaky_relu_wrapper.leaky_relu_e32m2(ptr_f32(x), ptr_f32(y), size, alpha)
    elif variant == "M4":
        leaky_relu_wrapper.leaky_relu_e32m4(ptr_f32(x), ptr_f32(y), size, alpha)
    elif variant == "M8":
        leaky_relu_wrapper.leaky_relu_e32m8(ptr_f32(x), ptr_f32(y), size, alpha)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return y

def maxpool(input: np.ndarray, k_h: int, k_w: int, stride_h: int = 1, stride_w: int = 1, pad_h: int = 0, pad_w: int = 0, variant="rvv", tile_h: int = None, tile_w: int = None):
    assert input.dtype == np.float32
    assert input.flags["C_CONTIGUOUS"]
    # input: N, C, H, W
    assert input.ndim == 4
    N, C, H, W = input.shape

    out_h = (H + 2 * pad_h - k_h) // stride_h + 1
    out_w = (W + 2 * pad_w - k_w) // stride_w + 1
    out = np.zeros((N, C, out_h, out_w), dtype=np.float32)

    if variant == "scalar":
        maxpool_wrapper.maxpool_scalar(ptr_f32(input), ptr_f32(out), N, C, H, W, k_h, k_w, stride_h, stride_w, pad_h, pad_w)
    elif variant == "M1":
        maxpool_wrapper.maxpool_e32m1(ptr_f32(input), ptr_f32(out), N, C, H, W, k_h, k_w, stride_h, stride_w, pad_h, pad_w)
    elif variant == "M2":
        maxpool_wrapper.maxpool_e32m2(ptr_f32(input), ptr_f32(out), N, C, H, W, k_h, k_w, stride_h, stride_w, pad_h, pad_w)
    elif variant == "M4":
        maxpool_wrapper.maxpool_e32m4(ptr_f32(input), ptr_f32(out), N, C, H, W, k_h, k_w, stride_h, stride_w, pad_h, pad_w)
    elif variant == "M8":
        maxpool_wrapper.maxpool_e32m8(ptr_f32(input), ptr_f32(out), N, C, H, W, k_h, k_w, stride_h, stride_w, pad_h, pad_w)
    elif variant == "tiled_M1":
        maxpool_wrapper.maxpool_rvv_tiled_m1(ptr_f32(input), ptr_f32(out), N, C, H, W, k_h, k_w, stride_h, stride_w, pad_h, pad_w, tile_h, tile_w)
    elif variant == "tiled_M2":
        maxpool_wrapper.maxpool_rvv_tiled_m2(ptr_f32(input), ptr_f32(out), N, C, H, W, k_h, k_w, stride_h, stride_w, pad_h, pad_w, tile_h, tile_w)
    elif variant == "tiled_M4":
        maxpool_wrapper.maxpool_rvv_tiled_m4(ptr_f32(input), ptr_f32(out), N, C, H, W, k_h, k_w, stride_h, stride_w, pad_h, pad_w, tile_h, tile_w)
    elif variant == "tiled_M8":
        maxpool_wrapper.maxpool_rvv_tiled_m8(ptr_f32(input), ptr_f32(out), N, C, H, W, k_h, k_w, stride_h, stride_w, pad_h, pad_w, tile_h, tile_w)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return out

def conv2d(input: np.ndarray, kernel: np.ndarray, bias: np.ndarray = None, stride=(1,1), pad=(0,0), variant="rvv"):
    assert input.dtype == np.float32
    assert kernel.dtype == np.float32
    if bias is not None:
        assert bias.dtype == np.float32
    assert input.flags["C_CONTIGUOUS"]
    assert kernel.flags["C_CONTIGUOUS"]

    # input: N, C_in, H, W
    assert input.ndim == 4
    N, C_in, H, W = input.shape
    
    # kernel: C_out, C_in, kH, kW
    assert kernel.ndim == 4
    C_out, C_in_k, kH, kW = kernel.shape
    assert C_in == C_in_k, f"Input channels {C_in} != Kernel in-channels {C_in_k}"

    stride_h, stride_w = stride
    pad_h, pad_w = pad

    out_h = (H + 2 * pad_h - kH) // stride_h + 1
    out_w = (W + 2 * pad_w - kW) // stride_w + 1

    out = np.zeros((N, C_out, out_h, out_w), dtype=np.float32)

    has_bias = 1 if bias is not None else 0
    bias_ptr = ptr_f32(bias) if has_bias else None  # Will be nullptr if None

    if variant == "scalar":
        conv_wrapper.conv2d_scalar(ptr_f32(input), ptr_f32(kernel), ptr_f32(out), 
                                   N, C_in, C_out, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w)
    elif variant == "M1":
        conv_wrapper.conv2d_e32m1(ptr_f32(input), ptr_f32(kernel), ptr_f32(out), 
                                  N, C_in, C_out, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w)
    elif variant == "M2":
        conv_wrapper.conv2d_e32m2(ptr_f32(input), ptr_f32(kernel), ptr_f32(out), 
                                  N, C_in, C_out, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w)
    elif variant == "M4":
        conv_wrapper.conv2d_e32m4(ptr_f32(input), ptr_f32(kernel), ptr_f32(out), 
                                  N, C_in, C_out, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w)
    elif variant == "M8":
        conv_wrapper.conv2d_e32m8(ptr_f32(input), ptr_f32(kernel), ptr_f32(out), 
                                  N, C_in, C_out, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w)
    elif variant == "im2col_M8":
        # Calculate buffer sizes for im2col_gemm
        # col_buf: C_in * kH * kW * outH * outW
        # gemm_buf: out_size (N * C_out * outH * outW) - usually handled internally by wrapper logic or separate buffer
        
        # The C++ function signature for im2col asks for specific buffers:
        # float* col_buf, float* gemm_buf
        
        col_len = C_in * kH * kW * out_h * out_w
        gemm_len = N * C_out * out_h * out_w # Intermediate gemm output often same size as final output if no bias/transpose stuff
        
        col_buf = np.zeros(col_len, dtype=np.float32)
        gemm_buf = np.zeros(gemm_len, dtype=np.float32)
        
        # Ensure bias is valid pointer (if None, create dummy or handle in wrapper if wrapper supports nullptr)
        # The wrapper definition for im2col accepts bias pointer. 
        if bias is None:
             # Create a dummy bias of zeros if required by C kernel or pass nullptr if handled.
             # Based on common usage, passing None (nullptr) often works if has_bias=0.
             actual_bias = np.zeros(C_out, dtype=np.float32) # Safe fallback
             has_bias = 0
        else:
             actual_bias = bias

        conv_wrapper.conv2d_im2col_gemm_m8(
            ptr_f32(input), ptr_f32(kernel), ptr_f32(actual_bias), ptr_f32(out),
            ptr_f32(col_buf), ptr_f32(gemm_buf),
            C_in, H, W, C_out, kH, kW,
            pad_h, pad_w, stride_h, stride_w, has_bias
        )
    else:
        # Check for specialized 3x3 kernels
        if kH == 3 and kW == 3:
            # Note: These kernels might assume specific layouts or padding handling as per C++ impl
            # Usually they are N=1 or specialized. Assuming signatures match wrappers.
            use_padding = (pad_h > 0 or pad_w > 0)
            if variant == "3x3_M1":
                conv_wrapper.conv2d_3x3_m1(ptr_f32(input), ptr_f32(kernel), ptr_f32(out), H, W, use_padding)
            elif variant == "3x3_M2":
                conv_wrapper.conv2d_3x3_m2(ptr_f32(input), ptr_f32(kernel), ptr_f32(out), H, W, use_padding)
            elif variant == "3x3_M4":
                conv_wrapper.conv2d_3x3_m4(ptr_f32(input), ptr_f32(kernel), ptr_f32(out), H, W, use_padding)
            elif variant == "3x3_M8":
                conv_wrapper.conv2d_3x3_m8(ptr_f32(input), ptr_f32(kernel), ptr_f32(out), H, W, use_padding)
            else:
                 raise ValueError(f"Unknown variant: {variant}")
        else:
            raise ValueError(f"Unknown variant: {variant}")

    return out

def softmax(x: np.ndarray):
    """
    1D softmax over the last dimension.
    Supports 1D (n,) or 2D (batch, n) by applying per row.
    """
    assert x.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"]

    if x.ndim == 1:
        n = x.shape[0]
        y = np.zeros_like(x)
        softmax_wrapper.softmax(ptr_f32(x), ptr_f32(y), n)
        return y
    elif x.ndim == 2:
        batch, n = x.shape
        y = np.zeros_like(x)
        # call kernel per row
        for b in range(batch):
            xb = x[b]
            yb = y[b]
            softmax_wrapper.softmax(ptr_f32(xb), ptr_f32(yb), n)
        return y
    else:
        raise ValueError("softmax() currently supports only 1D or 2D arrays")
