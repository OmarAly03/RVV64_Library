import numpy as np

def conv2d_scalar(input_arr, kernel, stride_h=1, stride_w=1, pad_h=1, pad_w=1):
    """
    Scalar 2D convolution using nested loops (similar to C++ version).
    Assumes channel-last layout (HWC for input, OIHW for kernel).
    
    Args:
    - input_arr: np.ndarray of shape (in_h, in_w, in_c)
    - kernel: np.ndarray of shape (out_c, in_c, k_h, k_w)
    - stride_h, stride_w: int
    - pad_h, pad_w: int
    
    Returns:
    - output: np.ndarray of shape (out_h, out_w, out_c)
    """
    in_h, in_w, in_c = input_arr.shape
    out_c, _, k_h, k_w = kernel.shape  # Assuming kernel shape (out_c, in_c, k_h, k_w)
    
    # Calculate output dimensions
    out_h = (in_h + 2 * pad_h - k_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - k_w) // stride_w + 1
    
    # Initialize output array
    output = np.zeros((out_h, out_w, out_c), dtype=np.float32)
    
    # Nested loops for convolution
    for oc in range(out_c):
        for oh in range(out_h):
            for ow in range(out_w):
                sum_val = 0.0
                for ic in range(in_c):
                    for kh in range(k_h):
                        for kw in range(k_w):
                            # Compute input positions with stride and padding
                            ih = oh * stride_h + kh - pad_h
                            iw = ow * stride_w + kw - pad_w
                            # Check bounds (padding is handled by skipping invalid ih/iw)
                            if 0 <= ih < in_h and 0 <= iw < in_w:
                                sum_val += input_arr[ih, iw, ic] * kernel[oc, ic, kh, kw]
                output[oh, ow, oc] = sum_val
    
    return output

# Example usage
in_h, in_w, in_c = 4, 4, 1
k_h, k_w, out_c = 3, 3, 1
stride_h, stride_w = 1, 1
pad_h, pad_w = 1, 1

# Input: 4x4x1, all 1s
input_arr = np.ones((in_h, in_w, in_c), dtype=np.float32)

# Kernel: 3x3, all 1/9, shaped as (1,1,3,3)
kernel = np.full((out_c, in_c, k_h, k_w), 1.0 / 9.0, dtype=np.float32)

# Perform convolution
output = conv2d_scalar(input_arr, kernel, stride_h, stride_w, pad_h, pad_w)

# Print output (should match the verified values)
print("Output:")
print(np.round(output[:, :, 0], 6))  # Round for clean display

def conv2d_scalar_flattened(input_flat, in_h, in_w, in_c, kernel_flat, k_h, k_w, out_c,
                            stride_h=1, stride_w=1, pad_h=1, pad_w=1):
    out_h = (in_h + 2 * pad_h - k_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - k_w) // stride_w + 1
    output_flat = np.zeros(out_h * out_w * out_c, dtype=np.float32)
    
    for oc in range(out_c):
        for oh in range(out_h):
            for ow in range(out_w):
                sum_val = 0.0
                for ic in range(in_c):
                    for kh in range(k_h):
                        for kw in range(k_w):
                            ih = oh * stride_h + kh - pad_h
                            iw = ow * stride_w + kw - pad_w
                            if 0 <= ih < in_h and 0 <= iw < in_w:
                                input_idx = ih * in_w * in_c + iw * in_c + ic
                                kernel_idx = oc * in_c * k_h * k_w + ic * k_h * k_w + kh * k_w + kw
                                sum_val += input_flat[input_idx] * kernel_flat[kernel_idx]
                output_idx = oh * out_w * out_c + ow * out_c + oc
                output_flat[output_idx] = sum_val
    
    return output_flat.reshape((out_h, out_w, out_c))

# Example
input_flat = np.ones(in_h * in_w * in_c, dtype=np.float32)
kernel_flat = np.full(out_c * in_c * k_h * k_w, 1.0 / 9.0, dtype=np.float32)
output_flat = conv2d_scalar_flattened(input_flat, in_h, in_w, in_c, kernel_flat, k_h, k_w, out_c,
                                      stride_h, stride_w, pad_h, pad_w)
print("Flattened Output:")
print(np.round(output_flat[:, :, 0], 6))