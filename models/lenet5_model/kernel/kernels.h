#ifndef KERNELS_H
#define KERNELS_H

#include <stddef.h>
#include <stdint.h>

extern "C" {
    // 1. Direct Convolution (C1)
    void conv2d_e32m8_direct(const float* input, const float* kernel, float* output,
        int batch_size, int in_channels, int out_channels,
        int input_h, int input_w, int kernel_h, int kernel_w,
        int stride_h, int stride_w, int pad_h, int pad_w,
        int do_relu);

    // 2. Im2Col + GEMM Convolution (C2, C3)
    void conv2d_e32m8_im2col(const float* input, const float* kernel, const float* bias,
        float* output, float* col_buf, float* gemm_buf,
        int in_channels, int input_h, int input_w, 
        int out_channels, int kernel_h, int kernel_w,
        int pad_h, int pad_w, int stride_h, int stride_w,
        int has_bias, int do_relu);

    // 3. Vector MaxPool (P1, P2)
    void maxpool_e32m8(const float* input, float* output,
        int batch, int channels, int in_h, int in_w,
        int k_h, int k_w, int stride_h, int stride_w);

    // 4. Vector Dense (F4, F5)
    void dense_e32m8(const float* input, const float* weights, const float* bias, 
        float* output, int in_dim, int out_dim, int do_relu);

    // 5. Bias Add (Helper)
    void bias_add_e32m8(const float* input, const float* bias, float* output,
        size_t batch_size, size_t channels, size_t height, size_t width,
        int do_relu);

    // 6. Tensor Add (Merge)
    void tensor_add_e32m8(const float* a, const float* b, float* out, size_t size);

    // 7. Scalar Softmax (Safe Version)
    void softmax_scalar(float* input, float* output, size_t size);
}

#endif