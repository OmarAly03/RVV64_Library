import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import pyv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyv.kernels import conv2d

def numpy_conv2d(input, kernel, stride=(1,1), pad=(0,0)):
    """
    Reference implementation of Conv2D using pure NumPy.
    Input: (N, Cin, H, W)
    Kernel: (Cout, Cin, kH, kW)
    Returns: (N, Cout, H_out, W_out)
    """
    N, Cin, H, W = input.shape
    Cout, _, kH, kW = kernel.shape
    sH, sW = stride
    pH, pW = pad
    
    # Calculate output dimensions
    out_h = (H + 2 * pH - kH) // sH + 1
    out_w = (W + 2 * pW - kW) // sW + 1
    
    # Pad input
    input_padded = np.pad(input, ((0,0), (0,0), (pH, pH), (pW, pW)), mode='constant')
    
    output = np.zeros((N, Cout, out_h, out_w), dtype=np.float32)
    
    for n in range(N):
        for c_out in range(Cout):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * sH
                    w_start = w * sW
                    h_end = h_start + kH
                    w_end = w_start + kW
                    
                    # Sum over input channels
                    patch = input_padded[n, :, h_start:h_end, w_start:w_end]
                    # Direct convolution (correlation) sum
                    output[n, c_out, h, w] = np.sum(patch * kernel[c_out])
                    
    return output

# ---------------------------------------------------------
# 1. Setup Data: Small "Image"
# ---------------------------------------------------------
print("Setting up test data...")

# N=1, C=1, H=8, W=8 (Small enough to debug, big enough to visualize)
N, C, H, W = 1, 1, 16, 16
input_image = np.zeros((N, C, H, W), dtype=np.float32)

# Create a recognizable pattern (e.g., a cross)
input_image[0, 0, 4:12, 7:9] = 1.0  # Vertical line
input_image[0, 0, 7:9, 4:12] = 1.0  # Horizontal line

# Define a Kernel: Edge detection / Sobel-like
# Cout=1, Cin=1, kH=3, kW=3
kernel = np.array([[[[
    -1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
]]], dtype=np.float32)

stride = (1, 1)
pad = (1, 1)

print(f"Input Shape: {input_image.shape}")
print(f"Kernel Shape: {kernel.shape}")

# ---------------------------------------------------------
# 2. Run Implementations
# ---------------------------------------------------------

# Reference (NumPy)
print("Running NumPy reference...")
ref_output = numpy_conv2d(input_image, kernel, stride, pad)

# RVV (Scalar variant for correctness on standard machines, or M1/M8 if emulated)
variant_to_test = "im2col_M8" 
print(f"Running RVV Kernel (variant='{variant_to_test}')...")
rvv_output = conv2d(input_image, kernel, stride=stride, pad=pad, variant=variant_to_test)

# ---------------------------------------------------------
# 3. Validation
# ---------------------------------------------------------
is_correct = np.allclose(ref_output, rvv_output, atol=1e-5)
max_diff = np.max(np.abs(ref_output - rvv_output))
print(f"\nVerification: {'PASSED' if is_correct else 'FAILED'}")
print(f"Max Absolute Difference: {max_diff}")

# ---------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------
print("Generating plot...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot Input
im0 = axes[0].imshow(input_image[0, 0], cmap='gray')
axes[0].set_title("Input Image")
axes[0].axis('off')
plt.colorbar(im0, ax=axes[0], fraction=0.046)

# Plot Reference Output
im1 = axes[1].imshow(ref_output[0, 0], cmap='viridis')
axes[1].set_title("NumPy Reference Output")
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046)

# Plot RVV Output
im2 = axes[2].imshow(rvv_output[0, 0], cmap='viridis')
axes[2].set_title(f"RVV ({variant_to_test}) Output")
axes[2].axis('off')
plt.colorbar(im2, ax=axes[2], fraction=0.046)

plt.suptitle(f"Convolution Test (Diff: {max_diff:.6f})")
plt.tight_layout()

# Save locally or show
output_plot_path = "conv_test_visual.png"
plt.savefig(output_plot_path)
print(f"Plot saved to {output_plot_path}")