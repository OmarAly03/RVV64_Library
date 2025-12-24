#ifndef DEFS_H
#define DEFS_H

#include <cstddef>

// Scalar implementation
void conv_transpose_2d_scalar(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w
);

// ============================================================================
// 3x3 Specialized RVV implementations - All LMUL variants
// ============================================================================
void transposed_conv2d_3x3_rvv_m1(
    const float* input, const float* kernel, float* output,
    int in_channels, int in_h, int in_w, int out_channels, int stride_h, int stride_w
);

void transposed_conv2d_3x3_rvv_m2(
    const float* input, const float* kernel, float* output,
    int in_channels, int in_h, int in_w, int out_channels, int stride_h, int stride_w
);

void transposed_conv2d_3x3_rvv_m4(
    const float* input, const float* kernel, float* output,
    int in_channels, int in_h, int in_w, int out_channels, int stride_h, int stride_w
);

void transposed_conv2d_3x3_rvv_m8(
    const float* input, const float* kernel, float* output,
    int in_channels, int in_h, int in_w, int out_channels, int stride_h, int stride_w
);

// ============================================================================
// General RVV implementations - All LMUL variants
// ============================================================================
void conv_transpose_2d_e32m1(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w
);

void conv_transpose_2d_e32m2(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w
);

void conv_transpose_2d_e32m4(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w
);

void conv_transpose_2d_e32m8(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w
);

// Utility functions
void write_matrix_binary(const char* filename, const float* data, size_t count);

#endif
