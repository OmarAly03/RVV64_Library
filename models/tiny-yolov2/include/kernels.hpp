// kernels.hpp
#pragma once
#include "model.hpp"

void preprocess_image(
    float* data, // In-place operation
    const float* scale, // Shape [1]
    const float* bias,  // Shape [3]
    int channels, int height, int width
);

void conv2d(
    const float* input,
    float* output,
    const float* weights, // [Out_C, In_C, K, K]
    int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int pad_top, int pad_left
);

void batch_normalization(
    float* data, // In-place operation
    const float* scale,
    const float* bias,
    const float* mean,
    const float* variance,
    int channels, int height, int width,
    float epsilon = 1e-5f
);

void leaky_relu(
    float* data, // In-place operation
    size_t num_elements,
    float alpha = 0.1f
);

void max_pool_2d(
    const float* input,
    float* output,
    int in_channels, int in_height, int in_width,
    int out_height, int out_width,
    int kernel_size, int stride, int pad_top, int pad_left
);

// Add bias (for the final layer)
void add_bias(
    float* data, // In-place
    const float* biases, // Shape [Channels]
    int channels, int height, int width
);