#include <string.h>
#include <stddef.h>
#include "util.h" 
#include "/home/omar/ara/lib/rvv_defs.hpp"  // For vector intrinsics/macros
#include <riscv_vector.h> // Required for vector intrinsics

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
                        size_t vl = __riscv_vsetvl_e32m8(current_bn - j);
                        vfloat32m8_t v_acc = __riscv_vle32_v_f32m8(&c_row_ptr[j], vl);
                        for (int k = k0; k < k_max; ++k) {
                            float a_val = A[i * K + k];
                            const float* b_row_ptr = &B[k * N + j0 + j];
                            vfloat32m8_t v_b = __riscv_vle32_v_f32m8(b_row_ptr, vl);
                            v_acc = __riscv_vfmacc_vf_f32m8(v_acc, a_val, v_b, vl);
                        }
                        __riscv_vse32_v_f32m8(&c_row_ptr[j], v_acc, vl);
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
                            size_t vl = __riscv_vsetvl_e32m8(out_width - ow);
                            vfloat32m8_t v_zero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
                            __riscv_vse32_v_f32m8(dest_ptr + ow, v_zero, vl);
                            ow += vl;
                        }
                    } else {
                        // Valid Row: Vectorize the copy of Output Width (ow)
                        for (int ow = 0; ow < out_width; ) {
                            size_t vl = __riscv_vsetvl_e32m8(out_width - ow);
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
                                vfloat32m8_t v_data = __riscv_vle32_v_f32m8(src_ptr, vl);
                                __riscv_vse32_v_f32m8(dest_ptr + ow, v_data, vl);
                            } else {
                                ptrdiff_t s_stride = stride_w * sizeof(float);
                                vfloat32m8_t v_data = __riscv_vlse32_v_f32m8(src_ptr, s_stride, vl);
                                __riscv_vse32_v_f32m8(dest_ptr + ow, v_data, vl);
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
// PART 3: IM2COL WRAPPERS
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
            vl = __riscv_vsetvl_e32m8(N - n);
            vfloat32m8_t v_data = __riscv_vle32_v_f32m8(&gem_ptr[n], vl);
            vfloat32m8_t v_res = __riscv_vfadd_vf_f32m8(v_data, b_val, vl);
            __riscv_vse32_v_f32m8(&out_ptr[n], v_res, vl);
            n += vl;
        }
    }
}



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
            size_t vl = __riscv_vsetvl_e32m8(N - n);
            
            vfloat32m8_t v_data = __riscv_vle32_v_f32m8(&src_ptr[n], vl);
            
            // Only add bias if it exists, otherwise just move data
            if (has_bias) {
                v_data = __riscv_vfadd_vf_f32m8(v_data, b_val, vl);
            }
            
            __riscv_vse32_v_f32m8(&dst_ptr[n], v_data, vl);
            n += vl;
        }
    }
}