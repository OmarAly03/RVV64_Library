#include <iostream>
#include <vector>
#include <cmath>  // For flo2or, but not used here

// Function for 2D normal convolution (scalar version)
// Input: flattened 1D array [in_h * in_w * in_c]
// Kernel: flattened 1D array [out_c * in_c * k_h * k_w]
// Output: flattened 1D array [out_h * out_w * out_c]
// Assumes channel-last layout (HWC)
void conv2d_scalar(const float* input, int in_h, int in_w, int in_c,
                   const float* kernel, int k_h, int k_w, int out_c,
                   float* output, int out_h, int out_w,
                   int stride_h, int stride_w, int pad_h, int pad_w) {
    // Clear output
    std::fill(output, output + out_h * out_w * out_c, 0.0f);

    // Nested loops: output channels, height, width
    for (int oc = 0; oc < out_c; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float sum = 0.0f;
                // Kernel loops: input channels, kernel height, width
                for (int ic = 0; ic < in_c; ++ic) {
                    for (int kh = 0; kh < k_h; ++kh) {
                        for (int kw = 0; kw < k_w; ++kw) {
                            // Compute input positions with stride and padding
                            int ih = oh * stride_h + kh - pad_h;
                            int iw = ow * stride_w + kw - pad_w;
                            // Check bounds
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                // Input index: ih * (in_w * in_c) + iw * in_c + ic
                                // Kernel index: oc * (in_c * k_h * k_w) + ic * (k_h * k_w) + kh * k_w + kw
                                sum += input[ih * in_w * in_c + iw * in_c + ic] *
                                       kernel[oc * in_c * k_h * k_w + ic * k_h * k_w + kh * k_w + kw];
                            }
                        }
                    }
                }
                // Output index: oh * (out_w * out_c) + ow * out_c + oc
                output[oh * out_w * out_c + ow * out_c + oc] = sum;
            }
        }
    }
}

int main() {
    // Example: 4x4 input, 1 channel, 3x3 kernel, 1 output channel, stride=1, pad=1
    // Expected output size: 4x4
    int in_h = 4, in_w = 4, in_c = 1;
    int k_h = 3, k_w = 3, out_c = 1;
    int stride_h = 1, stride_w = 1, pad_h = 1, pad_w = 1;
    int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;  // 4
    int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;  // 4

    std::vector<float> input(in_h * in_w * in_c, 1.0f);  // All 1s
    std::vector<float> kernel(out_c * in_c * k_h * k_w, 1.0f / 9.0f);  // Average kernel
    std::vector<float> output(out_h * out_w * out_c, 0.0f);

    conv2d_scalar(input.data(), in_h, in_w, in_c,
                  kernel.data(), k_h, k_w, out_c,
                  output.data(), out_h, out_w,
                  stride_h, stride_w, pad_h, pad_w);

    // Print output (should be all ~1.0 due to averaging 1s with padding)
    std::cout << "Output:" << std::endl;
    for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
            std::cout << output[h * out_w * out_c + w * out_c] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}