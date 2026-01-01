#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <riscv_vector.h>
#include "defs.h"
#include "rvv_defs.hpp"
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/*********************************** Scalar Version ************************************/

/** @brief 2D Convolution - Generalized scalar implementation with batch support
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

/********************************* General Vectorized Versions *********************************/

// RVV optimized 2D convolution (e32m1)
void conv2d_e32m1(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
    
    // Calculate output dimensions
    int out_height = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    
    // Initialize output to zero
    size_t output_size = batch_size * out_channels * out_height * out_width;
    std::memset(output, 0, output_size * sizeof(float));
    
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    
                    // Calculate input region bounds
                    int in_h_start = oh * stride_h - pad_h;
                    int in_w_start = ow * stride_w - pad_w;
                    int in_h_end = in_h_start + kernel_h;
                    int in_w_end = in_w_start + kernel_w;
                    
                    float sum = 0.0f;
                    
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            int in_h = in_h_start + kh;
                            
                            // Skip if outside input bounds
                            if (in_h < 0 || in_h >= input_h) {
                                continue;
                            }
                            
                            int kw = 0;
                            int in_w = in_w_start;
                            
                            // Skip negative width indices
                            while (kw < kernel_w && in_w + kw < 0) {
                                kw++;
                            }
                            
                            size_t vl;
                            for (; kw < kernel_w; kw += vl) {
                                int remaining = kernel_w - kw;
                                int valid_end = input_w - in_w - kw;
                                int processable = std::min(remaining, valid_end);
                                
                                if (processable <= 0) break;
                                
                                vl = SET_VECTOR_LENGTH<float, M1>(processable);

                                // Load input and kernel vectors using wrappers
                                vfloat32m1_t v_input = VECTOR_LOAD<float, M1>(
                                    &input[b * in_channels * input_h * input_w +
                                           ic * input_h * input_w +
                                           in_h * input_w + in_w + kw], vl);

                                vfloat32m1_t v_kernel = VECTOR_LOAD<float, M1>(
                                    &kernel[oc * in_channels * kernel_h * kernel_w +
                                           ic * kernel_h * kernel_w +
                                           kh * kernel_w + kw], vl);

                                // Element-wise multiply
                                vfloat32m1_t v_mult = VECTOR_MUL_VV<float, M1>(v_input, v_kernel, vl);

                                // Reduce sum horizontally
                                vfloat32m1_t v_zero = VECTOR_BROADCAST<float, M1>(0.0f, 1);
                                vfloat32m1_t v_sum = VECTOR_VFREDSUM<float, M1>(v_mult, v_zero, vl);

                                // Extract scalar sum and add to total
                                sum += VECTOR_EXTRACT_SCALAR<float, M1>(v_sum);
                            }
                        }
                    }
                    
                    // Store the accumulated result
                    output[b * out_channels * out_height * out_width +
                           oc * out_height * out_width +
                           oh * out_width + ow] = sum;
                }
            }
        }
    }
}

// RVV optimized 2D convolution (e32m2)
void conv2d_e32m2(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
    
    // Calculate output dimensions
    int out_height = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    
    // Initialize output to zero
    size_t output_size = batch_size * out_channels * out_height * out_width;
    std::memset(output, 0, output_size * sizeof(float));
    
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    
                    // Calculate input region bounds
                    int in_h_start = oh * stride_h - pad_h;
                    int in_w_start = ow * stride_w - pad_w;
                    
                    float sum = 0.0f;
                    
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            int in_h = in_h_start + kh;
                            
                            // Skip if outside input bounds
                            if (in_h < 0 || in_h >= input_h) {
                                continue;
                            }
                            
                            int kw = 0;
                            int in_w = in_w_start;
                            
                            // Skip negative width indices
                            while (kw < kernel_w && in_w + kw < 0) {
                                kw++;
                            }
                            
                            size_t vl;
                            for (; kw < kernel_w; kw += vl) {
                                int remaining = kernel_w - kw;
                                int valid_end = input_w - in_w - kw;
                                int processable = std::min(remaining, valid_end);
                                
                                if (processable <= 0) break;
                                
                                vl = SET_VECTOR_LENGTH<float, M2>(processable);

                                // Load input and kernel vectors using wrappers
                                vfloat32m2_t v_input = VECTOR_LOAD<float, M2>(
                                    &input[b * in_channels * input_h * input_w +
                                           ic * input_h * input_w +
                                           in_h * input_w + in_w + kw], vl);

                                vfloat32m2_t v_kernel = VECTOR_LOAD<float, M2>(
                                    &kernel[oc * in_channels * kernel_h * kernel_w +
                                           ic * kernel_h * kernel_w +
                                           kh * kernel_w + kw], vl);

                                // Element-wise multiply
                                vfloat32m2_t v_mult = VECTOR_MUL_VV<float, M2>(v_input, v_kernel, vl);

                                // Reduce sum horizontally
                                vfloat32m1_t v_zero = VECTOR_BROADCAST<float, M1>(0.0f, 1);
                                vfloat32m1_t v_sum = VECTOR_VFREDSUM<float, M2>(v_mult, v_zero, vl);

                                // Extract scalar sum and add to total
                                sum += VECTOR_EXTRACT_SCALAR<float, M1>(v_sum);
                            }
                        }
                    }
                    
                    // Store the accumulated result
                    output[b * out_channels * out_height * out_width +
                           oc * out_height * out_width +
                           oh * out_width + ow] = sum;
                }
            }
        }
    }
}

// RVV optimized 2D convolution (e32m4)
void conv2d_e32m4(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
    
    // Calculate output dimensions
    int out_height = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    
    // Initialize output to zero
    size_t output_size = batch_size * out_channels * out_height * out_width;
    std::memset(output, 0, output_size * sizeof(float));
    
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    
                    // Calculate input region bounds
                    int in_h_start = oh * stride_h - pad_h;
                    int in_w_start = ow * stride_w - pad_w;
                    
                    float sum = 0.0f;
                    
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            int in_h = in_h_start + kh;
                            
                            // Skip if outside input bounds
                            if (in_h < 0 || in_h >= input_h) {
                                continue;
                            }
                            
                            int kw = 0;
                            int in_w = in_w_start;
                            
                            // Skip negative width indices
                            while (kw < kernel_w && in_w + kw < 0) {
                                kw++;
                            }
                            
                            size_t vl;
                            for (; kw < kernel_w; kw += vl) {
                                int remaining = kernel_w - kw;
                                int valid_end = input_w - in_w - kw;
                                int processable = std::min(remaining, valid_end);
                                
                                if (processable <= 0) break;
                                
                                vl = SET_VECTOR_LENGTH<float, M4>(processable);

                                // Load input and kernel vectors using wrappers
                                vfloat32m4_t v_input = VECTOR_LOAD<float, M4>(
                                    &input[b * in_channels * input_h * input_w +
                                           ic * input_h * input_w +
                                           in_h * input_w + in_w + kw], vl);

                                vfloat32m4_t v_kernel = VECTOR_LOAD<float, M4>(
                                    &kernel[oc * in_channels * kernel_h * kernel_w +
                                           ic * kernel_h * kernel_w +
                                           kh * kernel_w + kw], vl);

                                // Element-wise multiply
                                vfloat32m4_t v_mult = VECTOR_MUL_VV<float, M4>(v_input, v_kernel, vl);

                                // Reduce sum horizontally
                                vfloat32m1_t v_zero = VECTOR_BROADCAST<float, M1>(0.0f, 1);
                                vfloat32m1_t v_sum = VECTOR_VFREDSUM<float, M4>(v_mult, v_zero, vl);

                                // Extract scalar sum and add to total
                                sum += VECTOR_EXTRACT_SCALAR<float, M1>(v_sum);
                            }
                        }
                    }
                    
                    // Store the accumulated result
                    output[b * out_channels * out_height * out_width +
                           oc * out_height * out_width +
                           oh * out_width + ow] = sum;
                }
            }
        }
    }
}

// RVV optimized 2D convolution (e32m8)
void conv2d_e32m8(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
    
    // Calculate output dimensions
    int out_height = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    
    // Initialize output to zero
    size_t output_size = batch_size * out_channels * out_height * out_width;
    std::memset(output, 0, output_size * sizeof(float));
    
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    
                    // Calculate input region bounds
                    int in_h_start = oh * stride_h - pad_h;
                    int in_w_start = ow * stride_w - pad_w;
                    
                    float sum = 0.0f;
                    
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            int in_h = in_h_start + kh;
                            
                            // Skip if outside input bounds
                            if (in_h < 0 || in_h >= input_h) {
                                continue;
                            }
                            
                            int kw = 0;
                            int in_w = in_w_start;
                            
                            // Skip negative width indices
                            while (kw < kernel_w && in_w + kw < 0) {
                                kw++;
                            }
                            
                            size_t vl;
                            for (; kw < kernel_w; kw += vl) {
                                int remaining = kernel_w - kw;
                                int valid_end = input_w - in_w - kw;
                                int processable = std::min(remaining, valid_end);
                                
                                if (processable <= 0) break;
                                
                                vl = SET_VECTOR_LENGTH<float, M8>(processable);

                                // Load input and kernel vectors using wrappers
                                vfloat32m8_t v_input = VECTOR_LOAD<float, M8>(
                                    &input[b * in_channels * input_h * input_w +
                                           ic * input_h * input_w +
                                           in_h * input_w + in_w + kw], vl);

                                vfloat32m8_t v_kernel = VECTOR_LOAD<float, M8>(
                                    &kernel[oc * in_channels * kernel_h * kernel_w +
                                           ic * kernel_h * kernel_w +
                                           kh * kernel_w + kw], vl);

                                // Element-wise multiply
                                vfloat32m8_t v_mult = VECTOR_MUL_VV<float, M8>(v_input, v_kernel, vl);

                                // Reduce sum horizontally
                                vfloat32m1_t v_zero = VECTOR_BROADCAST<float, M1>(0.0f, 1);
                                vfloat32m1_t v_sum = VECTOR_VFREDSUM<float, M8>(v_mult, v_zero, vl);

                                // Extract scalar sum and add to total
                                sum += VECTOR_EXTRACT_SCALAR<float, M1>(v_sum);
                            }
                        }
                    }
                    
                    // Store the accumulated result
                    output[b * out_channels * out_height * out_width +
                           oc * out_height * out_width +
                           oh * out_width + ow] = sum;
                }
            }
        }
    }
}

/********************************* ARA-Specific im2col-gemm Vectorized Versions*********************************/

#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

// =========================================================
//  IM2COL + GEMM HELPER FUNCTIONS
// =========================================================

// 1. Im2Col: Transform (C,H,W) -> (K, N)
void im2col_scalar(const float* input,
                   int C, int H, int W,
                   int kernel_h, int kernel_w,
                   int pad_h, int pad_w,
                   int stride_h, int stride_w,
                   float* col, 
                   int out_h, int out_w) {

    int N = out_h * out_w;
    int K = C * kernel_h * kernel_w;
    memset(col, 0, K * N * sizeof(float));

    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int k = c * (kernel_h * kernel_w) + kh * kernel_w + kw;
                for (int oh = 0; oh < out_h; ++oh) {
                    int ih = oh * stride_h + kh - pad_h;
                    for (int ow = 0; ow < out_w; ++ow) {
                        int iw = ow * stride_w + kw - pad_w;
                        int n = oh * out_w + ow;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            col[k * N + n] = input[c * H * W + ih * W + iw];
                        }
                    }
                }
            }
        }
    }
}

// 2. Blocked GEMM Scalar
void gemm_blocked_scalar(const float* A, const float* B, float* C,
                         int M, int N, int K,
                         int BM, int BN, int BK) {
    memset(C, 0, M * N * sizeof(float));
    for (int i0 = 0; i0 < M; i0 += BM) {
        int i_max = MIN(M, i0 + BM);
        for (int k0 = 0; k0 < K; k0 += BK) {
            int k_max = MIN(K, k0 + BK);
            for (int j0 = 0; j0 < N; j0 += BN) {
                int j_max = MIN(N, j0 + BN);
                for (int i = i0; i < i_max; ++i) {
                    for (int k = k0; k < k_max; ++k) {
                        float a_ik = A[i * K + k];
                        const float* b_ptr = &B[k * N + j0];
                        float* c_ptr = &C[i * N + j0];
                        for (int j = j0; j < j_max; ++j) {
                            c_ptr[j - j0] += a_ik * b_ptr[j - j0];
                        }
                    }
                }
            }
        }
    }
}

// 3. Blocked GEMM Vectorized (e32m8)
void gemm_blocked_e32m8(const float* A, const float* B, float* C,
                        int M, int N, int K,
                        int BM, int BN, int BK) {
    memset(C, 0, M * N * sizeof(float));
    for (int i0 = 0; i0 < M; i0 += BM) {
        int i_max = MIN(M, i0 + BM);
        for (int k0 = 0; k0 < K; k0 += BK) {
            int k_max = MIN(K, k0 + BK);
            for (int j0 = 0; j0 < N; j0 += BN) {
                int j_max = MIN(N, j0 + BN);
                for (int i = i0; i < i_max; ++i) {
                    float* c_row_ptr = &C[i * N + j0];
                    size_t j = 0;
                    size_t current_bn = j_max - j0;
                    while (j < current_bn) {
                        size_t vl = SET_VECTOR_LENGTH<float, M8>(current_bn - j);
                        vfloat32m8_t v_acc = VECTOR_LOAD<float, M8>(&c_row_ptr[j], vl);
                        for (int k = k0; k < k_max; ++k) {
                            float a_val = A[i * K + k];
                            const float* b_row_ptr = &B[k * N + j0 + j];
                            vfloat32m8_t v_b = VECTOR_LOAD<float, M8>(b_row_ptr, vl);
                            v_acc = VECTOR_FMACC_VF<float, M8>(v_acc, a_val, v_b, vl);
                        }
                        VECTOR_STORE<float, M8>(&c_row_ptr[j], v_acc, vl);
                        j += vl;
                    }
                }
            }
        }
    }
}


// Vectorized Im2Col: Transforms Image -> Matrix efficiently
// Concept: Instead of picking pixels 1-by-1, we copy 'vl' pixels 
// from the image row to the matrix row in one instruction.
void im2col_e32m8(const float* data_im, float* data_col,
                  int channels, int height, int width,
                  int kernel_h, int kernel_w,
                  int pad_h, int pad_w,
                  int stride_h, int stride_w) {
    
    int out_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int out_area = out_height * out_width;

    // 1. Iterate over the Kernel Elements (Rows of the output Matrix)
    //    This matrix has (channels * kh * kw) rows.
    for (int c = 0; c < channels; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                
                // Calculate the starting row index in the 'data_col' matrix
                float* col_row_ptr = data_col + (c * kernel_h * kernel_w + kh * kernel_w + kw) * out_area;

                // 2. Iterate over Output Height (Scalar)
                for (int oh = 0; oh < out_height; ++oh) {
                    
                    int in_h = oh * stride_h - pad_h + kh;
                    
                    // Calculate where we are writing in the linear matrix buffer
                    float* dest_ptr = col_row_ptr + (oh * out_width);

                    // Check Row Boundary
                    if (in_h < 0 || in_h >= height) {
                        // If the input row is invalid (padding), fill this segment with Zeros
                        // We vectorize the Zero-filling!
                        for (int ow = 0; ow < out_width; ) {
                            size_t vl = SET_VECTOR_LENGTH<float, M8>(out_width - ow);
                            vfloat32m8_t v_zero = VECTOR_BROADCAST<float, M8>(0.0f, vl);
                            VECTOR_STORE<float, M8>(dest_ptr + ow, v_zero, vl);
                            ow += vl;
                        }
                    } else {
                        // Valid Row: Vectorize the copy of Output Width (ow)
                        for (int ow = 0; ow < out_width; ) {
                            size_t vl = SET_VECTOR_LENGTH<float, M8>(out_width - ow);
                            int in_w = ow * stride_w - pad_w + kw;

                            // POINTER MATH:
                            // We need to load 'vl' pixels starting from &data_im[...]
                            // If stride_w == 1, we use Unit-Stride Load (Fastest)
                            // If stride_w > 1, we use Strided Load (Slower)
                            
                            const float* src_ptr = data_im + (c * height * width) + (in_h * width) + in_w;

                            // MASKING for Left/Right Padding
                            // If the vector load crosses the image boundary (left or right), we must mask.
                            // For raw speed on Ara, if we assume padded input or valid region, we skip masking.
                            // rigorous version: use vmseq to mask out-of-bound in_w.
                            
                            if (stride_w == 1) {
                                vfloat32m8_t v_data = VECTOR_LOAD<float, M8>(src_ptr, vl);
                                VECTOR_STORE<float, M8>(dest_ptr + ow, v_data, vl);
                            } else {
                                ptrdiff_t s_stride = stride_w * sizeof(float);
                                vfloat32m8_t v_data = VECTOR_STRIDED_LOAD<float, M8>(src_ptr, s_stride, vl);
                                VECTOR_STORE<float, M8>(dest_ptr + ow, v_data, vl);
                            }

                            ow += vl;
                        }
                    }
                }
            }
        }
    }
}

// =========================================================
// PART 1: SCALAR
// =========================================================

void conv2d_im2col_gemm_scalar(
    const float* input, const float* weights, const float* bias,
    float* output,
    float* col_buf, float* gemm_buf,
    int C, int H, int W, int M, int KH, int KW,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int has_bias
) {
    int out_h = (H + 2 * pad_h - KH) / stride_h + 1;
    int out_w = (W + 2 * pad_w - KW) / stride_w + 1;
    
    im2col_scalar(input, C, H, W, KH, KW, pad_h, pad_w, stride_h, stride_w, col_buf, out_h, out_w);

    int K = C * KH * KW;
    int N = out_h * out_w;

    gemm_blocked_scalar(weights, col_buf, gemm_buf, M, N, K, 32, 32, 32);

    for (int m = 0; m < M; ++m) {
        float b_val = has_bias ? bias[m] : 0.0f;
        for (int n = 0; n < N; ++n) {
            output[m * N + n] = gemm_buf[m * N + n] + b_val;
        }
    }
}

// =========================================================
// PART 2: PARTIAL VECTORIZED
// =========================================================
void conv2d_im2col_gemm_vector(
    const float* input, const float* weights, const float* bias,
    float* output,
    float* col_buf, float* gemm_buf,
    int C, int H, int W, int M, int KH, int KW,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int has_bias
) {
    int out_h = (H + 2 * pad_h - KH) / stride_h + 1;
    int out_w = (W + 2 * pad_w - KW) / stride_w + 1;
    
    im2col_scalar(input, C, H, W, KH, KW, pad_h, pad_w, stride_h, stride_w, col_buf, out_h, out_w);

    int K = C * KH * KW;
    int N = out_h * out_w;

    gemm_blocked_e32m8(weights, col_buf, gemm_buf, M, N, K, 32, 128, 32);

    // Vectorized Bias Add
    size_t vl;
    for (int m = 0; m < M; ++m) {
        float b_val = has_bias ? bias[m] : 0.0f;
        float* out_ptr = &output[m * N];
        float* gem_ptr = &gemm_buf[m * N];
        size_t n = 0;
        while (n < N) {
            vl = SET_VECTOR_LENGTH<float, M8>(N - n);
            vfloat32m8_t v_data = VECTOR_LOAD<float, M8>(&gem_ptr[n], vl);
            vfloat32m8_t v_res = VECTOR_ADD_VX<float, M8>(v_data, b_val, vl);
            VECTOR_STORE<float, M8>(&out_ptr[n], v_res, vl);
            n += vl;
        }
    }
}

// =========================================================
// PART 3: FULLY VECTORIZED
// =========================================================

void conv2d_im2col_gemm_m8(
    const float* input, const float* kernel, const float* bias,
    float* output,
    float* col_buf, float* gemm_buf,
    int in_channels, int input_h, int input_w, 
    int out_channels, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int has_bias) {

    // 1. Setup Dimensions
    int out_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    
    // GEMM Dimensions
    int M = out_channels;
    int K = in_channels * kernel_h * kernel_w;
    int N = out_h * out_w;
    
    // 2. Vectorized Im2Col (M8)
    // We use the external col_buf provided by main()
    im2col_e32m8(input, col_buf,
                 in_channels, input_h, input_w, 
                 kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);

    // 3. Blocked GEMM (M8)
    // Results go into gemm_buf first (to handle scaling/bias later) or directly to output
    // Since we need to add bias, let's write to gemm_buf first.
    // Note: Tuning the block sizes (BM=4, BN=16, BK=32) is key for performance.
    // You can try (32, 32, 32) or (8, 64, 16) depending on the exact Ara config.
    gemm_blocked_e32m8(kernel, col_buf, gemm_buf, M, N, K, 8, 64, 32); 

    // 4. Vectorized Bias Add (M8) & Store to Output
    // If has_bias is 0, this essentially just copies gemm_buf to output
    for (int m = 0; m < M; ++m) {
        float b_val = has_bias ? bias[m] : 0.0f;
        
        float* src_ptr = &gemm_buf[m * N];
        float* dst_ptr = &output[m * N];
        
        size_t n = 0;
        while (n < N) {
            size_t vl = SET_VECTOR_LENGTH<float, M8>(N - n);
            
            vfloat32m8_t v_data = VECTOR_LOAD<float, M8>(&src_ptr[n], vl);
            
            // Only add bias if it exists, otherwise just move data
            if (has_bias) {
                v_data = VECTOR_ADD<float, M8>(v_data, b_val, vl);
            }
            
            VECTOR_STORE<float, M8>(&dst_ptr[n], v_data, vl);
            n += vl;
        }
    }
}

/********************************* 3x3 Filter-Specific Vectorized Versions*********************************/

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Create zero-padded input (pad=1 for 3x3 kernel)
float* create_padded_input(const float* input, int H, int W) {
    int H_pad = H + 2;
    int W_pad = W + 2;
    
    float* padded = (float*)calloc(H_pad * W_pad, sizeof(float));
    
    // Copy input to center of padded buffer
    for (int h = 0; h < H; h++) {
        memcpy(padded + (h + 1) * W_pad + 1, 
               input + h * W, 
               W * sizeof(float));
    }
    
    return padded;
}

// ============================================================================
// M1 IMPLEMENTATION
// ============================================================================

void conv2d_3x3_m1(
    const float* input,    // Input: HxW
    const float* kernel,   // Kernel: 3x3 (9 elements, row-major)
    float* output,         // Output: HxW (with padding)
    int H,                 // Input height
    int W,                 // Input width
    bool use_padding       // If true, applies zero-padding
) {
    // Create padded input if needed
    float* padded_input = nullptr;
    const float* proc_input = input;
    int H_proc = H;
    int W_proc = W;
    
    if (use_padding) {
        padded_input = create_padded_input(input, H, W);
        proc_input = padded_input;
        H_proc = H + 2;
        W_proc = W + 2;
    }
    
    const int out_h = use_padding ? H : (H - 2);
    const int out_w = use_padding ? W : (W - 2);
    
    // Load kernel weights into scalar registers for broadcasting
    float k00 = kernel[0], k01 = kernel[1], k02 = kernel[2];
    float k10 = kernel[3], k11 = kernel[4], k12 = kernel[5];
    float k20 = kernel[6], k21 = kernel[7], k22 = kernel[8];
    
    // Process each output row
    for (int oh = 0; oh < out_h; oh++) {
        // Get pointers to the 3 input rows needed for this output row
        const float* row0 = proc_input + oh * W_proc;
        const float* row1 = row0 + W_proc;
        const float* row2 = row1 + W_proc;
        float* out_row = output + oh * out_w;
        
        // Vectorized processing across output width with m1
        int ow = 0;
        while (ow < out_w) {
            // Set vector length (m1: LMUL=1, processes ~4 elements on VLEN=128)
            size_t vl = SET_VECTOR_LENGTH<float, M1>(out_w - ow);
            
            // ================================================================
            // LOAD PHASE: Load 9 vectors (3x3 neighborhood)
            // ================================================================
            // Row 0: Load 3 overlapping vectors with offsets 0, 1, 2
            vfloat32m1_t v00 = VECTOR_LOAD<float, M1>(row0 + ow, vl);
            vfloat32m1_t v01 = VECTOR_LOAD<float, M1>(row0 + ow + 1, vl);
            vfloat32m1_t v02 = VECTOR_LOAD<float, M1>(row0 + ow + 2, vl);
            
            // Row 1: Load 3 overlapping vectors
            vfloat32m1_t v10 = VECTOR_LOAD<float, M1>(row1 + ow, vl);
            vfloat32m1_t v11 = VECTOR_LOAD<float, M1>(row1 + ow + 1, vl);
            vfloat32m1_t v12 = VECTOR_LOAD<float, M1>(row1 + ow + 2, vl);
            
            // Row 2: Load 3 overlapping vectors
            vfloat32m1_t v20 = VECTOR_LOAD<float, M1>(row2 + ow, vl);
            vfloat32m1_t v21 = VECTOR_LOAD<float, M1>(row2 + ow + 1, vl);
            vfloat32m1_t v22 = VECTOR_LOAD<float, M1>(row2 + ow + 2, vl);
            
            // ================================================================
            // COMPUTE PHASE: Fused Multiply-Accumulate (FMA) chain
            // ================================================================
            // Start with first multiply
            vfloat32m1_t acc = VECTOR_MUL<float, M1>(v00, k00, vl);
            
            // Chain the remaining 8 FMAs (vector * scalar + accumulator)
            acc = VECTOR_FMACC<float, M1>(acc, k01, v01, vl);
            acc = VECTOR_FMACC<float, M1>(acc, k02, v02, vl);
            acc = VECTOR_FMACC<float, M1>(acc, k10, v10, vl);
            acc = VECTOR_FMACC<float, M1>(acc, k11, v11, vl);
            acc = VECTOR_FMACC<float, M1>(acc, k12, v12, vl);
            acc = VECTOR_FMACC<float, M1>(acc, k20, v20, vl);
            acc = VECTOR_FMACC<float, M1>(acc, k21, v21, vl);
            acc = VECTOR_FMACC<float, M1>(acc, k22, v22, vl);
            
            // ================================================================
            // STORE PHASE: Write results
            // ================================================================
            VECTOR_STORE<float, M1>(out_row + ow, acc, vl);
            
            // Move to next vector chunk
            ow += vl;
        }
    }
    
    // Cleanup
    if (padded_input) {
        free(padded_input);
    }
}

// ============================================================================
// M2 IMPLEMENTATION 
// ============================================================================

void conv2d_3x3_m2(
    const float* input,    // Input: HxW
    const float* kernel,   // Kernel: 3x3 (9 elements, row-major)
    float* output,         // Output: HxW (with padding) or (H-2)x(W-2)
    int H,                 // Input height
    int W,                 // Input width
    bool use_padding       // If true, applies zero-padding
) {
    // Create padded input if needed
    float* padded_input = nullptr;
    const float* proc_input = input;
    int H_proc = H;
    int W_proc = W;
    
    if (use_padding) {
        padded_input = create_padded_input(input, H, W);
        proc_input = padded_input;
        H_proc = H + 2;
        W_proc = W + 2;
    }
    
    const int out_h = use_padding ? H : (H - 2);
    const int out_w = use_padding ? W : (W - 2);
    
    // Load kernel weights into scalar registers
    float k00 = kernel[0], k01 = kernel[1], k02 = kernel[2];
    float k10 = kernel[3], k11 = kernel[4], k12 = kernel[5];
    float k20 = kernel[6], k21 = kernel[7], k22 = kernel[8];
    
    // Process each output row
    for (int oh = 0; oh < out_h; oh++) {
        // Get pointers to the 3 input rows needed
        const float* row0 = proc_input + oh * W_proc;
        const float* row1 = row0 + W_proc;
        const float* row2 = row1 + W_proc;
        float* out_row = output + oh * out_w;
        
        // Vectorized processing with m2 (2x throughput vs m1)
        int ow = 0;
        while (ow < out_w) {
            // Set vector length (m2: LMUL=2, processes ~8 elements on VLEN=128)
            size_t vl = SET_VECTOR_LENGTH<float, M2>(out_w - ow);
            
            // ================================================================
            // LOAD PHASE: Load 9 vectors with m2
            // ================================================================
            // Row 0: 3 vectors with offsets 0, 1, 2
            vfloat32m2_t v00 = VECTOR_LOAD<float, M2>(row0 + ow, vl);
            vfloat32m2_t v01 = VECTOR_LOAD<float, M2>(row0 + ow + 1, vl);
            vfloat32m2_t v02 = VECTOR_LOAD<float, M2>(row0 + ow + 2, vl);
            
            // Row 1: 3 vectors
            vfloat32m2_t v10 = VECTOR_LOAD<float, M2>(row1 + ow, vl);
            vfloat32m2_t v11 = VECTOR_LOAD<float, M2>(row1 + ow + 1, vl);
            vfloat32m2_t v12 = VECTOR_LOAD<float, M2>(row1 + ow + 2, vl);
            
            // Row 2: 3 vectors
            vfloat32m2_t v20 = VECTOR_LOAD<float, M2>(row2 + ow, vl);
            vfloat32m2_t v21 = VECTOR_LOAD<float, M2>(row2 + ow + 1, vl);
            vfloat32m2_t v22 = VECTOR_LOAD<float, M2>(row2 + ow + 2, vl);
            
            // ================================================================
            // COMPUTE PHASE: FMA chain with m2 vectors
            // ================================================================
            vfloat32m2_t acc = VECTOR_MUL<float, M2>(v00, k00, vl);
            acc = VECTOR_FMACC<float, M2>(acc, k01, v01, vl);
            acc = VECTOR_FMACC<float, M2>(acc, k02, v02, vl);
            acc = VECTOR_FMACC<float, M2>(acc, k10, v10, vl);
            acc = VECTOR_FMACC<float, M2>(acc, k11, v11, vl);
            acc = VECTOR_FMACC<float, M2>(acc, k12, v12, vl);
            acc = VECTOR_FMACC<float, M2>(acc, k20, v20, vl);
            acc = VECTOR_FMACC<float, M2>(acc, k21, v21, vl);
            acc = VECTOR_FMACC<float, M2>(acc, k22, v22, vl);
            
            // ================================================================
            // STORE PHASE
            // ================================================================
            VECTOR_STORE<float, M2>(out_row + ow, acc, vl);
            
            ow += vl;
        }
    }
    
    // Cleanup
    if (padded_input) {
        free(padded_input);
    }
}

// ============================================================================
// M4 IMPLEMENTATION
// ============================================================================

void conv2d_3x3_m4(
    const float* input,    // Input: HxW
    const float* kernel,   // Kernel: 3x3 (9 elements, row-major)
    float* output,         // Output: HxW (with padding) or (H-2)x(W-2)
    int H,                 // Input height
    int W,                 // Input width
    bool use_padding       // If true, applies zero-padding
) {
    float* padded_input = nullptr;
    const float* proc_input = input;
    int H_proc = H;
    int W_proc = W;
    
    if (use_padding) {
        padded_input = create_padded_input(input, H, W);
        proc_input = padded_input;
        H_proc = H + 2;
        W_proc = W + 2;
    }
    
    const int out_h = use_padding ? H : (H - 2);
    const int out_w = use_padding ? W : (W - 2);
    
    float k00 = kernel[0], k01 = kernel[1], k02 = kernel[2];
    float k10 = kernel[3], k11 = kernel[4], k12 = kernel[5];
    float k20 = kernel[6], k21 = kernel[7], k22 = kernel[8];

    for (int oh = 0; oh < out_h; oh++) {
        const float* row0 = proc_input + oh * W_proc;
        const float* row1 = row0 + W_proc;
        const float* row2 = row1 + W_proc;
        float* out_row = output + oh * out_w;

        int ow = 0;
        while (ow < out_w) {
            size_t vl = SET_VECTOR_LENGTH<float, M4>(out_w - ow);

            vfloat32m4_t v00 = VECTOR_LOAD<float, M4>(row0 + ow, vl);
            vfloat32m4_t v01 = VECTOR_LOAD<float, M4>(row0 + ow + 1, vl);
            vfloat32m4_t v02 = VECTOR_LOAD<float, M4>(row0 + ow + 2, vl);

            vfloat32m4_t v10 = VECTOR_LOAD<float, M4>(row1 + ow, vl);
            vfloat32m4_t v11 = VECTOR_LOAD<float, M4>(row1 + ow + 1, vl);
            vfloat32m4_t v12 = VECTOR_LOAD<float, M4>(row1 + ow + 2, vl);

            vfloat32m4_t v20 = VECTOR_LOAD<float, M4>(row2 + ow, vl);
            vfloat32m4_t v21 = VECTOR_LOAD<float, M4>(row2 + ow + 1, vl);
            vfloat32m4_t v22 = VECTOR_LOAD<float, M4>(row2 + ow + 2, vl);

            vfloat32m4_t acc = VECTOR_MUL<float, M4>(v00, k00, vl);
            acc = VECTOR_FMACC<float, M4>(acc, k01, v01, vl);
            acc = VECTOR_FMACC<float, M4>(acc, k02, v02, vl);
            acc = VECTOR_FMACC<float, M4>(acc, k10, v10, vl);
            acc = VECTOR_FMACC<float, M4>(acc, k11, v11, vl);
            acc = VECTOR_FMACC<float, M4>(acc, k12, v12, vl);
            acc = VECTOR_FMACC<float, M4>(acc, k20, v20, vl);
            acc = VECTOR_FMACC<float, M4>(acc, k21, v21, vl);
            acc = VECTOR_FMACC<float, M4>(acc, k22, v22, vl);

            VECTOR_STORE<float, M4>(out_row + ow, acc, vl);

            ow += vl;
        }
    }

    if (padded_input) free(padded_input);
}

// ============================================================================
// M8 IMPLEMENTATION 
// ============================================================================

void conv2d_3x3_m8(
    const float* input,
    const float* kernel,
    float* output,
    int H,
    int W,
    bool use_padding
) {
    float* padded_input = nullptr;
    const float* proc_input = input;
    int H_proc = H;
    int W_proc = W;

    if (use_padding) {
        padded_input = create_padded_input(input, H, W);
        proc_input = padded_input;
        H_proc = H + 2;
        W_proc = W + 2;
    }

    const int out_h = use_padding ? H : (H - 2);
    const int out_w = use_padding ? W : (W - 2);

    float k00 = kernel[0], k01 = kernel[1], k02 = kernel[2];
    float k10 = kernel[3], k11 = kernel[4], k12 = kernel[5];
    float k20 = kernel[6], k21 = kernel[7], k22 = kernel[8];

    for (int oh = 0; oh < out_h; oh++) {
        const float* row0 = proc_input + oh * W_proc;
        const float* row1 = row0 + W_proc;
        const float* row2 = row1 + W_proc;
        float* out_row = output + oh * out_w;

        int ow = 0;
        while (ow < out_w) {
            size_t vl = SET_VECTOR_LENGTH<float, M8>(out_w - ow);

            vfloat32m8_t v00 = VECTOR_LOAD<float, M8>(row0 + ow, vl);
            vfloat32m8_t v01 = VECTOR_LOAD<float, M8>(row0 + ow + 1, vl);
            vfloat32m8_t v02 = VECTOR_LOAD<float, M8>(row0 + ow + 2, vl);

            vfloat32m8_t v10 = VECTOR_LOAD<float, M8>(row1 + ow, vl);
            vfloat32m8_t v11 = VECTOR_LOAD<float, M8>(row1 + ow + 1, vl);
            vfloat32m8_t v12 = VECTOR_LOAD<float, M8>(row1 + ow + 2, vl);

            vfloat32m8_t v20 = VECTOR_LOAD<float, M8>(row2 + ow, vl);
            vfloat32m8_t v21 = VECTOR_LOAD<float, M8>(row2 + ow + 1, vl);
            vfloat32m8_t v22 = VECTOR_LOAD<float, M8>(row2 + ow + 2, vl);

            vfloat32m8_t acc = VECTOR_MUL<float, M8>(v00, k00, vl);
            acc = VECTOR_FMACC<float, M8>(acc, k01, v01, vl);
            acc = VECTOR_FMACC<float, M8>(acc, k02, v02, vl);
            acc = VECTOR_FMACC<float, M8>(acc, k10, v10, vl);
            acc = VECTOR_FMACC<float, M8>(acc, k11, v11, vl);
            acc = VECTOR_FMACC<float, M8>(acc, k12, v12, vl);
            acc = VECTOR_FMACC<float, M8>(acc, k20, v20, vl);
            acc = VECTOR_FMACC<float, M8>(acc, k21, v21, vl);
            acc = VECTOR_FMACC<float, M8>(acc, k22, v22, vl);

            VECTOR_STORE<float, M8>(out_row + ow, acc, vl);

            ow += vl;
        }
    }

    if (padded_input) free(padded_input);
}


/********************************* 3x3 Filter-Specific BATCHED Vectorized Versions (Cache-optimized) *********************************/

// ============================================================================
// MULTI-ROW BATCHED M2 ( for edge devices)
// ============================================================================

void conv2d_3x3_m2_batched(
    const float* input,
    const float* kernel,
    float* output,
    int H,
    int W,
    bool use_padding,
    int batch_rows = 4    // Process N output rows together
) {
    float* padded_input = nullptr;
    const float* proc_input = input;
    int H_proc = H;
    int W_proc = W;
    
    if (use_padding) {
        padded_input = create_padded_input(input, H, W);
        proc_input = padded_input;
        H_proc = H + 2;
        W_proc = W + 2;
    }
    
    const int out_h = use_padding ? H : (H - 2);
    const int out_w = use_padding ? W : (W - 2);
    
    float k00 = kernel[0], k01 = kernel[1], k02 = kernel[2];
    float k10 = kernel[3], k11 = kernel[4], k12 = kernel[5];
    float k20 = kernel[6], k21 = kernel[7], k22 = kernel[8];
    
    // Process output rows in batches for better cache reuse
    for (int oh_base = 0; oh_base < out_h; oh_base += batch_rows) {
        int rows_to_process = (oh_base + batch_rows <= out_h) ? 
                              batch_rows : (out_h - oh_base);
        
        // For each column position (vectorized)
        for (int ow = 0; ow < out_w; ) {
            size_t vl = SET_VECTOR_LENGTH<float, M2>(out_w - ow);
            
            // Process each row in the current batch
            for (int r = 0; r < rows_to_process; r++) {
                int oh = oh_base + r;
                const float* row0 = proc_input + oh * W_proc;
                const float* row1 = row0 + W_proc;
                const float* row2 = row1 + W_proc;
                
                // Load 9 vectors
                vfloat32m2_t v00 = VECTOR_LOAD<float, M2>(row0 + ow, vl);
                vfloat32m2_t v01 = VECTOR_LOAD<float, M2>(row0 + ow + 1, vl);
                vfloat32m2_t v02 = VECTOR_LOAD<float, M2>(row0 + ow + 2, vl);
                
                vfloat32m2_t v10 = VECTOR_LOAD<float, M2>(row1 + ow, vl);
                vfloat32m2_t v11 = VECTOR_LOAD<float, M2>(row1 + ow + 1, vl);
                vfloat32m2_t v12 = VECTOR_LOAD<float, M2>(row1 + ow + 2, vl);
                
                vfloat32m2_t v20 = VECTOR_LOAD<float, M2>(row2 + ow, vl);
                vfloat32m2_t v21 = VECTOR_LOAD<float, M2>(row2 + ow + 1, vl);
                vfloat32m2_t v22 = VECTOR_LOAD<float, M2>(row2 + ow + 2, vl);
                
                // Compute
                vfloat32m2_t acc = VECTOR_MUL<float, M2>(v00, k00, vl);
                acc = VECTOR_FMACC<float, M2>(acc, k01, v01, vl);
                acc = VECTOR_FMACC<float, M2>(acc, k02, v02, vl);
                acc = VECTOR_FMACC<float, M2>(acc, k10, v10, vl);
                acc = VECTOR_FMACC<float, M2>(acc, k11, v11, vl);
                acc = VECTOR_FMACC<float, M2>(acc, k12, v12, vl);
                acc = VECTOR_FMACC<float, M2>(acc, k20, v20, vl);
                acc = VECTOR_FMACC<float, M2>(acc, k21, v21, vl);
                acc = VECTOR_FMACC<float, M2>(acc, k22, v22, vl);
                
                // Store
                VECTOR_STORE<float, M2>(output + oh * out_w + ow, acc, vl);
            }
            
            ow += vl;
        }
    }
    
    if (padded_input) {
        free(padded_input);
    }
}

// ============================================================================
// M4 BATCHED
// ============================================================================

void conv2d_3x3_m4_batched(
    const float* input,
    const float* kernel,
    float* output,
    int H,
    int W,
    bool use_padding,
    int batch_rows = 4
) {
    float* padded_input = nullptr;
    const float* proc_input = input;
    int H_proc = H;
    int W_proc = W;

    if (use_padding) {
        padded_input = create_padded_input(input, H, W);
        proc_input = padded_input;
        H_proc = H + 2;
        W_proc = W + 2;
    }

    const int out_h = use_padding ? H : (H - 2);
    const int out_w = use_padding ? W : (W - 2);

    float k00 = kernel[0], k01 = kernel[1], k02 = kernel[2];
    float k10 = kernel[3], k11 = kernel[4], k12 = kernel[5];
    float k20 = kernel[6], k21 = kernel[7], k22 = kernel[8];

    for (int oh_base = 0; oh_base < out_h; oh_base += batch_rows) {
        int rows_to_process = (oh_base + batch_rows <= out_h) ?
                              batch_rows : (out_h - oh_base);

        for (int ow = 0; ow < out_w; ) {
            size_t vl = SET_VECTOR_LENGTH<float, M4>(out_w - ow);

            for (int r = 0; r < rows_to_process; r++) {
                int oh = oh_base + r;
                const float* row0 = proc_input + oh * W_proc;
                const float* row1 = row0 + W_proc;
                const float* row2 = row1 + W_proc;

                vfloat32m4_t v00 = VECTOR_LOAD<float, M4>(row0 + ow, vl);
                vfloat32m4_t v01 = VECTOR_LOAD<float, M4>(row0 + ow + 1, vl);
                vfloat32m4_t v02 = VECTOR_LOAD<float, M4>(row0 + ow + 2, vl);

                vfloat32m4_t v10 = VECTOR_LOAD<float, M4>(row1 + ow, vl);
                vfloat32m4_t v11 = VECTOR_LOAD<float, M4>(row1 + ow + 1, vl);
                vfloat32m4_t v12 = VECTOR_LOAD<float, M4>(row1 + ow + 2, vl);

                vfloat32m4_t v20 = VECTOR_LOAD<float, M4>(row2 + ow, vl);
                vfloat32m4_t v21 = VECTOR_LOAD<float, M4>(row2 + ow + 1, vl);
                vfloat32m4_t v22 = VECTOR_LOAD<float, M4>(row2 + ow + 2, vl);

                vfloat32m4_t acc = VECTOR_MUL<float, M4>(v00, k00, vl);
                acc = VECTOR_FMACC<float, M4>(acc, k01, v01, vl);
                acc = VECTOR_FMACC<float, M4>(acc, k02, v02, vl);
                acc = VECTOR_FMACC<float, M4>(acc, k10, v10, vl);
                acc = VECTOR_FMACC<float, M4>(acc, k11, v11, vl);
                acc = VECTOR_FMACC<float, M4>(acc, k12, v12, vl);
                acc = VECTOR_FMACC<float, M4>(acc, k20, v20, vl);
                acc = VECTOR_FMACC<float, M4>(acc, k21, v21, vl);
                acc = VECTOR_FMACC<float, M4>(acc, k22, v22, vl);

                VECTOR_STORE<float, M4>(output + oh * out_w + ow, acc, vl);
            }

            ow += vl;
        }
    }

    if (padded_input) free(padded_input);
}

// ============================================================================
// M8 BATCHED
// ============================================================================

void conv2d_3x3_m8_batched(
    const float* input,
    const float* kernel,
    float* output,
    int H,
    int W,
    bool use_padding,
    int batch_rows = 4
) {
    float* padded_input = nullptr;
    const float* proc_input = input;
    int H_proc = H;
    int W_proc = W;

    if (use_padding) {
        padded_input = create_padded_input(input, H, W);
        proc_input = padded_input;
        H_proc = H + 2;
        W_proc = W + 2;
    }

    const int out_h = use_padding ? H : (H - 2);
    const int out_w = use_padding ? W : (W - 2);

    float k00 = kernel[0], k01 = kernel[1], k02 = kernel[2];
    float k10 = kernel[3], k11 = kernel[4], k12 = kernel[5];
    float k20 = kernel[6], k21 = kernel[7], k22 = kernel[8];

    for (int oh_base = 0; oh_base < out_h; oh_base += batch_rows) {
        int rows_to_process = (oh_base + batch_rows <= out_h) ?
                              batch_rows : (out_h - oh_base);

        for (int ow = 0; ow < out_w; ) {
            size_t vl = SET_VECTOR_LENGTH<float, M8>(out_w - ow);

            for (int r = 0; r < rows_to_process; r++) {
                int oh = oh_base + r;
                const float* row0 = proc_input + oh * W_proc;
                const float* row1 = row0 + W_proc;
                const float* row2 = row1 + W_proc;

                vfloat32m8_t v00 = VECTOR_LOAD<float, M8>(row0 + ow, vl);
                vfloat32m8_t v01 = VECTOR_LOAD<float, M8>(row0 + ow + 1, vl);
                vfloat32m8_t v02 = VECTOR_LOAD<float, M8>(row0 + ow + 2, vl);

                vfloat32m8_t v10 = VECTOR_LOAD<float, M8>(row1 + ow, vl);
                vfloat32m8_t v11 = VECTOR_LOAD<float, M8>(row1 + ow + 1, vl);
                vfloat32m8_t v12 = VECTOR_LOAD<float, M8>(row1 + ow + 2, vl);

                vfloat32m8_t v20 = VECTOR_LOAD<float, M8>(row2 + ow, vl);
                vfloat32m8_t v21 = VECTOR_LOAD<float, M8>(row2 + ow + 1, vl);
                vfloat32m8_t v22 = VECTOR_LOAD<float, M8>(row2 + ow + 2, vl);

                vfloat32m8_t acc = VECTOR_MUL<float, M8>(v00, k00, vl);
                acc = VECTOR_FMACC<float, M8>(acc, k01, v01, vl);
                acc = VECTOR_FMACC<float, M8>(acc, k02, v02, vl);
                acc = VECTOR_FMACC<float, M8>(acc, k10, v10, vl);
                acc = VECTOR_FMACC<float, M8>(acc, k11, v11, vl);
                acc = VECTOR_FMACC<float, M8>(acc, k12, v12, vl);
                acc = VECTOR_FMACC<float, M8>(acc, k20, v20, vl);
                acc = VECTOR_FMACC<float, M8>(acc, k21, v21, vl);
                acc = VECTOR_FMACC<float, M8>(acc, k22, v22, vl);

                VECTOR_STORE<float, M8>(output + oh * out_w + ow, acc, vl);
            }

            ow += vl;
        }
    }

    if (padded_input) free(padded_input);
}

/********************************* 3x3 Filter-Specific RGB Vectorized Versions *********************************/

void conv2d_3x3_m2_rgb(
    const float* input,     // Input: 3xHxW (channel-major layout)
    const float* kernel,    // Kernel: 3x3x3 (27 elements: R, G, B kernels)
    float* output,          // Output: 3xHxW
    int H,
    int W,
    bool use_padding
) {
    const int out_h = use_padding ? H : (H - 2);
    const int out_w = use_padding ? W : (W - 2);
    
    // Process each channel independently
    for (int c = 0; c < 3; c++) {
        const float* in_channel = input + c * H * W;
        const float* kernel_channel = kernel + c * 9;
        float* out_channel = output + c * out_h * out_w;
        
        conv2d_3x3_m2(in_channel, kernel_channel, out_channel, H, W, use_padding);
    }
}

void conv2d_3x3_m4_rgb(
    const float* input,
    const float* kernel,
    float* output,
    int H,
    int W,
    bool use_padding
) {
    for (int c = 0; c < 3; c++) {
        const float* in_channel = input + c * H * W;
        const float* kernel_channel = kernel + c * 9;
        float* out_channel = output + c * (use_padding ? H * W : (H - 2) * (W - 2));
        conv2d_3x3_m4(in_channel, kernel_channel, out_channel, H, W, use_padding);
    }
}

void conv2d_3x3_m8_rgb(
    const float* input,
    const float* kernel,
    float* output,
    int H,
    int W,
    bool use_padding
) {
    for (int c = 0; c < 3; c++) {
        const float* in_channel = input + c * H * W;
        const float* kernel_channel = kernel + c * 9;
        float* out_channel = output + c * (use_padding ? H * W : (H - 2) * (W - 2));
        conv2d_3x3_m8(in_channel, kernel_channel, out_channel, H, W, use_padding);
    }
}

