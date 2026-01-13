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

def dense(x: np.ndarray, weights: np.ndarray, bias: np.ndarray = None, variant="rvv"):
    assert x.dtype == np.float32
    assert weights.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"]
    assert weights.flags["C_CONTIGUOUS"]

    # x: (in_features,) or (batch, in_features)
    if x.ndim == 1:
        in_features = x.shape[0]
        out_features = weights.shape[1]
        out = np.zeros((out_features,), dtype=np.float32)
        if bias is None:
            bias = np.zeros((out_features,), dtype=np.float32)
        if variant == "scalar":
            dense_wrapper.dense_scalar(ptr_f32(x), ptr_f32(weights), ptr_f32(bias), ptr_f32(out), in_features, out_features)
        elif variant == "M1":
            dense_wrapper.dense_e32m1(ptr_f32(x), ptr_f32(weights), ptr_f32(bias), ptr_f32(out), in_features, out_features)
        elif variant == "M2":
            dense_wrapper.dense_e32m2(ptr_f32(x), ptr_f32(weights), ptr_f32(bias), ptr_f32(out), in_features, out_features)
        elif variant == "M4":
            dense_wrapper.dense_e32m4(ptr_f32(x), ptr_f32(weights), ptr_f32(bias), ptr_f32(out), in_features, out_features)
        elif variant == "M8":
            dense_wrapper.dense_e32m8(ptr_f32(x), ptr_f32(weights), ptr_f32(bias), ptr_f32(out), in_features, out_features)
        else:
            raise ValueError(f"Unknown variant: {variant}")
        return out
    else:
        raise ValueError("Only 1-D input supported for dense() in this wrapper")

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