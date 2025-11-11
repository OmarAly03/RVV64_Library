#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cassert>


/**
 * @brief 2D Convolution - Generalized scalar implementation with batch support
 * 
 * Performs 2D convolution with zero-padding and configurable stride.
 * Supports batched inputs and follows standard deep learning conventions.
 * 
 * Memory Layout:
 * - Input:  [batch_size, in_channels, input_h, input_w] (NCHW format)
 * - Kernel: [out_channels, in_channels, kernel_h, kernel_w] (OIHW format)
 * - Output: [batch_size, out_channels, out_h, out_w] (NCHW format)
 * 
 * @param input         Input tensor (flattened 1D array)
 * @param kernel        Convolution kernel/weights (flattened 1D array)
 * @param output        Output tensor (flattened 1D array, pre-allocated)
 * @param batch_size    Number of samples in the batch
 * @param in_channels   Number of input channels
 * @param out_channels  Number of output channels (number of filters)
 * @param input_h       Input height
 * @param input_w       Input width
 * @param kernel_h      Kernel height
 * @param kernel_w      Kernel width
 * @param stride_h      Vertical stride
 * @param stride_w      Horizontal stride
 * @param pad_h         Vertical padding (applied to top and bottom)
 * @param pad_w         Horizontal padding (applied to left and right)
 */

void conv2d_scalar(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
    
    
    // Calculate output spatial dimensions
    int out_height = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    
    // Initialize output tensor to zero
    std::memset(output, 0, batch_size * out_channels * out_height * out_width * sizeof(float));
    
    // Main convolution loops
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0f;
                    
                    // Accumulate over all input channels and kernel spatial dimensions
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                // Compute input coordinates with stride and padding
                                int in_h = oh * stride_h - pad_h + kh;
                                int in_w = ow * stride_w - pad_w + kw;
                                
                                // Check if coordinates are within input bounds (zero-padding)
                                if (in_h >= 0 && in_h < input_h && in_w >= 0 && in_w < input_w) {
                                    // Calculate flat indices for NCHW layout
                                    // Input index: [b][ic][in_h][in_w]
                                    int input_idx = b * (in_channels * input_h * input_w) +
                                                   ic * (input_h * input_w) + 
                                                   in_h * input_w + in_w;
                                    
                                    // Kernel index: [oc][ic][kh][kw]
                                    int kernel_idx = oc * (in_channels * kernel_h * kernel_w) +
                                                    ic * (kernel_h * kernel_w) + 
                                                    kh * kernel_w + kw;
                                    
                                    // Multiply-accumulate operation
                                    sum += input[input_idx] * kernel[kernel_idx];
                                }
                            }
                        }
                    }
                    
                    // Store result in output tensor
                    // Output index: [b][oc][oh][ow]
                    int output_idx = b * (out_channels * out_height * out_width) +
                                    oc * (out_height * out_width) + 
                                    oh * out_width + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

#ifdef BUILD_SCALAR_CONV_MAIN
int main() {
    // Example: 4x4 input, 1 channel, 3x3 kernel, 1 output channel, stride=1, pad=1
    // Expected output size: 4x4
    int batch_size = 1;
    int in_channels = 1, out_channels = 1;
    int input_h = 4, input_w = 4;
    int kernel_h = 3, kernel_w = 3;
    int stride_h = 1, stride_w = 1;
    int pad_h = 1, pad_w = 1;
    
    // Calculate output dimensions
    int out_height = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;  // 4
    int out_width = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;   // 4

    // Create test data (NCHW layout)
    std::vector<float> input(batch_size * in_channels * input_h * input_w, 1.0f);  // All 1s
    std::vector<float> kernel(out_channels * in_channels * kernel_h * kernel_w, 1.0f / 9.0f);  // Average kernel
    std::vector<float> output(batch_size * out_channels * out_height * out_width, 0.0f);

    // Call the updated conv2d_scalar function
    conv2d_scalar(input.data(), kernel.data(), output.data(),
                  batch_size, in_channels, out_channels,
                  input_h, input_w, kernel_h, kernel_w,
                  stride_h, stride_w, pad_h, pad_w);

    // Print output (should be all ~1.0 due to averaging 1s with padding)
    std::cout << "Convolution Test Results:" << std::endl;
    std::cout << "Input: " << input_h << "x" << input_w << " with " << in_channels << " channels" << std::endl;
    std::cout << "Kernel: " << kernel_h << "x" << kernel_w << " with " << out_channels << " filters" << std::endl;
    std::cout << "Output: " << out_height << "x" << out_width << " with " << out_channels << " channels" << std::endl;
    std::cout << "Stride: (" << stride_h << ", " << stride_w << "), Padding: (" << pad_h << ", " << pad_w << ")" << std::endl;
    std::cout << std::endl << "Output values:" << std::endl;
    
    for (int h = 0; h < out_height; ++h) {
        for (int w = 0; w < out_width; ++w) {
            // NCHW layout: [batch=0][channel=0][h][w]
            int idx = 0 * (out_channels * out_height * out_width) + 
                     0 * (out_height * out_width) + 
                     h * out_width + w;
            std::cout << output[idx] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
#endif // BUILD_SCALAR_CONV_MAIN