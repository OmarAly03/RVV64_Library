#ifndef DEFS_H
#define DEFS_H

#include <cstddef>

void conv_transpose_2d_scalar(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w
);

void transposed_conv2d_3x3_s2_direct_m8(
    const float* input,
    const float* kernel,
    float* output,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int stride_h,
    int stride_w
);

void write_matrix_binary(const char* filename, const float* data, size_t count);

#endif
