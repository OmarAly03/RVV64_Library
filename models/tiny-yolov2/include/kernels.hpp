// kernels.hpp
#pragma once
#include "model.hpp"

void preprocess_image(
    float* data, // In-place operation
    const std::vector<float>& scale, // Shape [1]
    const std::vector<float>& bias,  // Shape [3]
    int channels, int height, int width
);

void conv2d(
    const float* input,
    float* output,
    const std::vector<float>& weights, // [Out_C, In_C, K, K]
    int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int pad_top, int pad_left
);

void batch_normalization(
    float* data, // In-place operation
    const std::vector<float>& scale,
    const std::vector<float>& bias,
    const std::vector<float>& mean,
    const std::vector<float>& variance,
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
    const std::vector<float>& biases, // Shape [Channels]
    int channels, int height, int width
);