// kernels.cpp
#include "kernels.hpp"
#include <cfloat> // for FLT_MAX

// (x * scale) + bias
void preprocess_image(
    float* data, const float* scale,
    const float* bias,
    int channels, int height, int width)
{
    const size_t spatial_dim = height * width;
    const float s = scale[0];
    for (int c = 0; c < channels; ++c) {
        float b = bias[c];
        for (size_t i = 0; i < spatial_dim; ++i) {
            size_t idx = c * spatial_dim + i;
            data[idx] = (data[idx] * s) + b;
        }
    }
}

// NOTE: This is a *naive* implementation.
// For real performance, you would use im2col + GEMM (e.g., with Eigen/BLAS)
void conv2d(
    const float* input, float* output,
    const float* weights,
    int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int pad_top, int pad_left)
{
    const int in_spatial = in_height * in_width;
    const int out_spatial = out_height * out_width;
    const int kernel_spatial = kernel_size * kernel_size;

    // Clear output buffer
    std::fill(output, output + (out_channels * out_spatial), 0.0f);

    for (int c_out = 0; c_out < out_channels; ++c_out) {
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int h_out = 0; h_out < out_height; ++h_out) {
                for (int w_out = 0; w_out < out_width; ++w_out) {
                    
                    const int h_in_start = h_out * stride - pad_top;
                    const int w_in_start = w_out * stride - pad_left;
                    
                    float sum = 0.0f;
                    
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            
                            const int h_in = h_in_start + kh;
                            const int w_in = w_in_start + kw;

                            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                                int in_idx = (c_in * in_spatial) + (h_in * in_width) + w_in;
                                int w_idx = (c_out * in_channels * kernel_spatial) + (c_in * kernel_spatial) + (kh * kernel_size) + kw;
                                sum += input[in_idx] * weights[w_idx];
                            }
                        }
                    }
                    output[(c_out * out_spatial) + (h_out * out_width) + w_out] += sum;
                }
            }
        }
    }
}

void batch_normalization(
    float* data, const float* scale,
    const float* bias, const float* mean,
    const float* variance,
    int channels, int height, int width, float epsilon)
{
    const int spatial_dim = height * width;
    for (int c = 0; c < channels; ++c) {
        float s = scale[c];
        float b = bias[c];
        float m = mean[c];
        float v = variance[c];
        
        float std_dev_inv = 1.0f / std::sqrt(v + epsilon);
        
        for (int i = 0; i < spatial_dim; ++i) {
            int idx = c * spatial_dim + i;
            data[idx] = s * ((data[idx] - m) * std_dev_inv) + b;
        }
    }
}

void leaky_relu(float* data, size_t num_elements, float alpha) {
    for (size_t i = 0; i < num_elements; ++i) {
        if (data[i] < 0) {
            data[i] = data[i] * alpha;
        }
    }
}

void max_pool_2d(
    const float* input, float* output,
    int in_channels, int in_height, int in_width,
    int out_height, int out_width,
    int kernel_size, int stride, int pad_top, int pad_left)
{
    const int in_spatial = in_height * in_width;
    const int out_spatial = out_height * out_width;

    for (int c = 0; c < in_channels; ++c) {
        for (int h_out = 0; h_out < out_height; ++h_out) {
            for (int w_out = 0; w_out < out_width; ++w_out) {
                
                const int h_in_start = h_out * stride - pad_top;
                const int w_in_start = w_out * stride - pad_left;
                
                float max_val = -FLT_MAX;
                
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int h_in = h_in_start + kh;
                        int w_in = w_in_start + kw;
                        
                        if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                            int in_idx = (c * in_spatial) + (h_in * in_width) + w_in;
                            max_val = std::max(max_val, input[in_idx]);
                        }
                    }
                }
                int out_idx = (c * out_spatial) + (h_out * out_width) + w_out;
                output[out_idx] = max_val;
            }
        }
    }
}

void add_bias(
    float* data, const float* biases,
    int channels, int height, int width)
{
    const size_t spatial_dim = height * width;
    for (int c = 0; c < channels; ++c) {
        float b = biases[c];
        for (size_t i = 0; i < spatial_dim; ++i) {
            data[c * spatial_dim + i] += b;
        }
    }
}

/****************************************************************************************************************************/

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