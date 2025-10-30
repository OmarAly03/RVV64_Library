#ifndef DEFS_H
#define DEFS_H

#include <cstddef>

void conv_transpose_2d_scalar(
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

void write_matrix_binary(const char* filename, const float* data, size_t count);

#endif
