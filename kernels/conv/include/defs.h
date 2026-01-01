#ifndef DEFS_H
#define DEFS_H

#include <cstddef>

// RVV optimized 2D convolution functions
void conv2d_e32m1(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w);

void conv2d_e32m2(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w);

void conv2d_e32m4(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w);

void conv2d_e32m8(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w);

// Scalar reference implementation
void conv2d_scalar(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w);

void conv2d_scalar(const float* input, int in_h, int in_w, int in_c,
                   const float* kernel, int k_h, int k_w, int out_c,
                   float* output, int out_h, int out_w,
                   int stride_h, int stride_w, int pad_h, int pad_w);

// Im2Col and GEMM helper functions
void im2col_scalar(const float* input,
                   int C, int H, int W,
                   int kernel_h, int kernel_w,
                   int pad_h, int pad_w,
                   int stride_h, int stride_w,
                   float* col, 
                   int out_h, int out_w);

void gemm_blocked_scalar(const float* A, const float* B, float* C,
                         int M, int N, int K,
                         int BM, int BN, int BK);

void gemm_blocked_e32m8(const float* A, const float* B, float* C,
                        int M, int N, int K,
                        int BM, int BN, int BK);

void im2col_e32m8(const float* data_im, float* data_col,
                  int channels, int height, int width,
                  int kernel_h, int kernel_w,
                  int pad_h, int pad_w,
                  int stride_h, int stride_w);

// Im2Col-GEMM convolution functions
void conv2d_im2col_gemm_scalar(
    const float* input, const float* weights, const float* bias,
    float* output,
    float* col_buf, float* gemm_buf,
    int C, int H, int W, int M, int KH, int KW,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int has_bias);

void conv2d_im2col_gemm_vector(
    const float* input, const float* weights, const float* bias,
    float* output,
    float* col_buf, float* gemm_buf,
    int C, int H, int W, int M, int KH, int KW,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int has_bias);

void conv2d_im2col_gemm_m8(
    const float* input, const float* kernel, const float* bias,
    float* output,
    float* col_buf, float* gemm_buf,
    int in_channels, int input_h, int input_w, 
    int out_channels, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int has_bias);

// 3x3 specialized RVV functions
void conv2d_3x3_m1(
    const float* input, const float* kernel, float* output,
    int H, int W, bool use_padding);

void conv2d_3x3_m2(
    const float* input, const float* kernel, float* output,
    int H, int W, bool use_padding);

void conv2d_3x3_m4(
    const float* input, const float* kernel, float* output,
    int H, int W, bool use_padding);

void conv2d_3x3_m8(
    const float* input, const float* kernel, float* output,
    int H, int W, bool use_padding);

// Batched versions
void conv2d_3x3_m2_batched(
    const float* input, const float* kernel, float* output,
    int H, int W, bool use_padding, int batch_rows);

void conv2d_3x3_m4_batched(
    const float* input, const float* kernel, float* output,
    int H, int W, bool use_padding, int batch_rows);

void conv2d_3x3_m8_batched(
    const float* input, const float* kernel, float* output,
    int H, int W, bool use_padding, int batch_rows);

// RGB versions
void conv2d_3x3_m2_rgb(
    const float* input, const float* kernel, float* output,
    int H, int W, bool use_padding);

void conv2d_3x3_m4_rgb(
    const float* input, const float* kernel, float* output,
    int H, int W, bool use_padding);

void conv2d_3x3_m8_rgb(
    const float* input, const float* kernel, float* output,
    int H, int W, bool use_padding);

// Utility functions
void write_matrix_to_file(const char* filename, float* matrix, std::size_t rows, std::size_t cols);
void write_matrix_binary(const char* filename, float* matrix, std::size_t count);

#endif