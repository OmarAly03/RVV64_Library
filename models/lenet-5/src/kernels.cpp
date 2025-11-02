#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>   // For std::iota
#include <algorithm> // For std::max_element, std::distance
#include <cstddef>   // For size_t
#include <stdexcept> // For std::runtime_error
#include <cmath>     // For exp

#include <cassert>   // For assert()
#include <cstring>   // For std::memset
#include <cfloat>    // For FLT_MAX

#include "../include/defs.hpp"

std::vector<float> load_weights(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Error: Cannot open file: " + filename);
    }
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size % sizeof(float) != 0) {
        throw std::runtime_error("Error: File size is not a multiple of float: " + filename);
    }
    std::vector<float> buffer(size / sizeof(float));
    if (!in.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Error reading data from file: " + filename);
    }
    return buffer;
}

void load_preprocessed_image(std::vector<float>& img_buffer, const std::string& filename) {
    std::cout << "Loading image: " << filename << std::endl;
    std::vector<float> loaded_data = load_weights(filename);
    if (loaded_data.size() != img_buffer.size()) {
        throw std::runtime_error("Image file size mismatch. Expected " + 
            std::to_string(img_buffer.size()) + " floats, but file has " +
            std::to_string(loaded_data.size()));
    }
    std::copy(loaded_data.begin(), loaded_data.end(), img_buffer.begin());
}

// =======================================================
// KERNEL HEADERS & PLACEHOLDERS
// =======================================================

// --- Your Provided Kernels ---
inline int conv_output_size(int input_size, int kernel_size, int stride, int pad) {
    return (input_size + 2 * pad - kernel_size) / stride + 1;
}


void conv2d_scalar(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
    
    // Parameter validation
    assert(input != nullptr && kernel != nullptr && output != nullptr);
    assert(batch_size > 0 && in_channels > 0 && out_channels > 0);
    assert(input_h > 0 && input_w > 0);
    assert(kernel_h > 0 && kernel_w > 0);
    assert(stride_h > 0 && stride_w > 0);
    assert(pad_h >= 0 && pad_w >= 0);
    
    // Calculate output spatial dimensions using helper function
    int out_height = conv_output_size(input_h, kernel_h, stride_h, pad_h);
    int out_width = conv_output_size(input_w, kernel_w, stride_w, pad_w);
    
    // Ensure output dimensions are positive
    assert(out_height > 0 && out_width > 0);
    
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
                                // Note: Outside bounds contributes 0 (implicit zero-padding)
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

void maxpool_scalar_tile(
    const float* X, float* Y, int64_t* I,
    size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode,
    size_t OH, size_t OW,
    size_t tile_oh_start, size_t tile_ow_start,
    size_t tile_oh_end, size_t tile_ow_end)
{
    // (Scalar code remains the same)
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t oh = tile_oh_start; oh < tile_oh_end; ++oh) {
                for (size_t ow = tile_ow_start; ow < tile_ow_end; ++ow) {
                    float max_val = -FLT_MAX;
                    int64_t max_idx = -1;
                    for (size_t kh = 0; kh < K; ++kh) {
                        for (size_t kw = 0; kw < K; ++kw) {
                            size_t ih = oh * S + kh;
                            size_t iw = ow * S + kw;
                            if (ih < H && iw < W) {
                                float val = X[n*C*H*W + c*H*W + ih*W + iw];
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = (ih * W + iw);
                                }
                            }
                        }
                    }
                    Y[n*C*OH*OW + c*OH*OW + oh*OW + ow] = max_val;
                    I[n*C*OH*OW + c*OH*OW + oh*OW + ow] = (max_idx != -1) ? (n * C * H * W) + (c * H * W) + max_idx : -1;
                }
            }
        }
    }
}

void relu_scalar(float* input, float* output, size_t size) {
    for (size_t i = 0; i < size; i++) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

void dense_scalar(const float* input, const float* weights, const float* bias,
                        float* output, size_t in_features, size_t out_features) {
    // Implements Y = A*B^T + C, where A=input, B=weights, C=bias
    // A shape: [in_features]
    // B shape: [out_features, in_features]
    // C shape: [out_features]
    // Y shape: [out_features]
    for (size_t out_f = 0; out_f < out_features; ++out_f) {
        float sum = 0.0f;
        for (size_t in_f = 0; in_f < in_features; ++in_f) {
            sum += input[in_f] * weights[out_f * in_features + in_f];
        }
        output[out_f] = sum + bias[out_f];
    }
}


void bias_add_scalar(const float* input, const float* bias, float* output,
                       size_t batch_size, size_t channels,
                       size_t height, size_t width) {
    
    // Size of one 2D feature map
    size_t channel_size = height * width; 
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            // Get the scalar bias value for this channel
            float b_val = bias[c]; 
            // Calculate the starting offset for this channel
            size_t offset = (b * channels + c) * channel_size;
            
            const float* in_ptr = input + offset;
            float* out_ptr = output + offset;
            
            // This is the loop we will vectorize
            for (size_t i = 0; i < channel_size; ++i) {
                out_ptr[i] = in_ptr[i] + b_val;
            }
        }
    }
}

/**
 * ### NEW PLACEHOLDER KERNEL ###
 * Element-wise addition of two tensors.
 */
void tensor_add_scalar(const float* input_a, const float* input_b, float* output,
                           size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = input_a[i] + input_b[i];
    }
}

/**
 * ### NEW PLACEHOLDER KERNEL ###
 * LogSoftmax implementation.
 */
void softmax_scalar(float* input, float* output, size_t size) {
    // Pass 1: Find Max
    float max_val = -__builtin_inff();
    for (size_t i = 0; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // Pass 2: Calculate Exponentials and Sum
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    // Pass 3: Divide by Sum
    for (size_t i = 0; i < size; i++) {
        output[i] /= sum;
    }
}