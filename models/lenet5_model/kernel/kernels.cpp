#include "kernels.h"
#include <riscv_vector.h>
#include <string.h>
// No math.h or float.h included to prevent linking issues on baremetal

#define MIN(a,b) (((a)<(b))?(a):(b))

extern "C" {

// --- SCALAR EXP APPROXIMATION (For Softmax) ---
// Polynomial approximation to avoid libm dependency
static inline float exp_approx(float x) {
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f; // Clamp to avoid Inf

    // Constants for Cephes approx
    float log2ef = 1.44269504088896341f;
    float C1 = 0.693359375f;
    float C2 = -2.12194440e-4f;
    float p0 = 1.9875691500E-4f;
    float p1 = 1.3981999507E-3f;
    float p2 = 8.3334519073E-3f;
    float p3 = 4.1665795894E-2f;
    float p4 = 1.6666665459E-1f;
    float p5 = 5.0000001201E-1f;

    float fx = x * log2ef + 0.5f;
    int n = (int)fx;
    float fn = (float)n;
    float t = fn * C1;
    float z = fn * C2;
    float r = x - t - z;

    float y = p0 * r + p1;
    y = y * r + p2;
    y = y * r + p3;
    y = y * r + p4;
    y = y * r + p5;
    y = y * r * r + r + 1.0f;

    // Build 2^n
    int exponent = n + 127;
    union { int i; float f; } packer;
    packer.i = (exponent << 23);
    
    return y * packer.f;
}

// ============================================================
// 1. DIRECT CONVOLUTION
// ============================================================
void conv2d_e32m8_direct(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int do_relu) {

    int out_height = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    
    size_t output_size = batch_size * out_channels * out_height * out_width;
    memset(output, 0, output_size * sizeof(float));

    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                int in_h_origin = oh * stride_h - pad_h;
                
                for (int ow = 0; ow < out_width; ) {
                    size_t vl = __riscv_vsetvl_e32m8(out_width - ow);
                    vfloat32m8_t v_acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
                    int in_w_origin = ow * stride_w - pad_w;

                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                int in_h = in_h_origin + kh;
                                if (in_h < 0 || in_h >= input_h) continue;

                                int in_w_base = in_w_origin + kw;
                                float k_val = kernel[oc * in_channels * kernel_h * kernel_w +
                                                     ic * kernel_h * kernel_w +
                                                     kh * kernel_w + kw];

                                ptrdiff_t in_stride = stride_w * sizeof(float);
                                const float* in_ptr = &input[b * in_channels * input_h * input_w +
                                                             ic * input_h * input_w +
                                                             in_h * input_w + in_w_base];
                                
                                vfloat32m8_t v_input = __riscv_vlse32_v_f32m8(in_ptr, in_stride, vl);
                                v_acc = __riscv_vfmacc_vf_f32m8(v_acc, k_val, v_input, vl);
                            }
                        }
                    }
                    
                    if (do_relu) {
                         vfloat32m8_t v_zero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
                         v_acc = __riscv_vfmax_vv_f32m8(v_acc, v_zero, vl);
                    }

                    float* out_ptr = &output[b * out_channels * out_height * out_width +
                                             oc * out_height * out_width +
                                             oh * out_width + ow];
                    __riscv_vse32_v_f32m8(out_ptr, v_acc, vl);
                    ow += vl;
                }
            }
        }
    }
}

// ============================================================
// 2. IM2COL + GEMM HELPERS
// ============================================================
void im2col_e32m8(const float* data_im, float* data_col,
                  int channels, int height, int width,
                  int kernel_h, int kernel_w,
                  int pad_h, int pad_w,
                  int stride_h, int stride_w) {
    
    int out_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int out_area = out_height * out_width;

    for (int c = 0; c < channels; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                float* col_row_ptr = data_col + (c * kernel_h * kernel_w + kh * kernel_w + kw) * out_area;
                for (int oh = 0; oh < out_height; ++oh) {
                    int in_h = oh * stride_h - pad_h + kh;
                    float* dest_ptr = col_row_ptr + (oh * out_width);

                    if (in_h < 0 || in_h >= height) {
                        for (int ow = 0; ow < out_width; ) {
                            size_t vl = __riscv_vsetvl_e32m8(out_width - ow);
                            __riscv_vse32_v_f32m8(dest_ptr + ow, __riscv_vfmv_v_f_f32m8(0.0f, vl), vl);
                            ow += vl;
                        }
                    } else {
                        for (int ow = 0; ow < out_width; ) {
                            size_t vl = __riscv_vsetvl_e32m8(out_width - ow);
                            int in_w = ow * stride_w - pad_w + kw;
                            const float* src_ptr = data_im + (c * height * width) + (in_h * width) + in_w;
                            ptrdiff_t s_stride = stride_w * sizeof(float);
                            __riscv_vse32_v_f32m8(dest_ptr + ow, __riscv_vlse32_v_f32m8(src_ptr, s_stride, vl), vl);
                            ow += vl;
                        }
                    }
                }
            }
        }
    }
}

void gemm_blocked_e32m8(const float* A, const float* B, float* C,
                        int M, int N, int K, int BM, int BN, int BK) {
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

void conv2d_e32m8_im2col(const float* input, const float* kernel, const float* bias,
    float* output, float* col_buf, float* gemm_buf,
    int in_channels, int input_h, int input_w, 
    int out_channels, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int has_bias, int do_relu) {

    int out_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    int M = out_channels;
    int K = in_channels * kernel_h * kernel_w;
    int N = out_h * out_w;
    
    im2col_e32m8(input, col_buf, in_channels, input_h, input_w, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);
    gemm_blocked_e32m8(kernel, col_buf, gemm_buf, M, N, K, 8, 64, 32);
    
    for (int m = 0; m < M; ++m) {
        float b_val = has_bias ? bias[m] : 0.0f;
        float* src_ptr = &gemm_buf[m * N];
        float* dst_ptr = &output[m * N];
        size_t n = 0;
        while (n < N) {
            size_t vl = __riscv_vsetvl_e32m8(N - n);
            vfloat32m8_t v_data = __riscv_vle32_v_f32m8(&src_ptr[n], vl);
            if (has_bias) v_data = __riscv_vfadd_vf_f32m8(v_data, b_val, vl);
            
            if (do_relu) {
                vfloat32m8_t v_zero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
                v_data = __riscv_vfmax_vv_f32m8(v_data, v_zero, vl);
            }
            
            __riscv_vse32_v_f32m8(&dst_ptr[n], v_data, vl);
            n += vl;
        }
    }
}

// ============================================================
// 3. MAXPOOL
// ============================================================
void maxpool_e32m8(const float* input, float* output,
                   int batch, int channels, int in_h, int in_w,
                   int k_h, int k_w, int stride_h, int stride_w) {

    int out_h = (in_h - k_h) / stride_h + 1;
    int out_w = (in_w - k_w) / stride_w + 1;

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            const float* in_ptr_base = input + (b * channels + c) * in_h * in_w;
            float* out_ptr_base = output + (b * channels + c) * out_h * out_w;
            for (int oh = 0; oh < out_h; ++oh) {
                int h_start = oh * stride_h;
                float* out_row = out_ptr_base + oh * out_w;
                for (int ow = 0; ow < out_w; ) {
                    size_t vl = __riscv_vsetvl_e32m8(out_w - ow);
                    vfloat32m8_t v_max = __riscv_vfmv_v_f_f32m8(-3.402823466e+38F, vl);
                    int w_start_base = ow * stride_w;
                    for (int kh = 0; kh < k_h; ++kh) {
                        int cur_h = h_start + kh;
                        if (cur_h >= in_h) continue;
                        const float* in_row_ptr = in_ptr_base + cur_h * in_w;
                        for (int kw = 0; kw < k_w; ++kw) {
                            ptrdiff_t in_stride = stride_w * sizeof(float);
                            const float* load_addr = in_row_ptr + w_start_base + kw;
                            vfloat32m8_t v_in = __riscv_vlse32_v_f32m8(load_addr, in_stride, vl);
                            v_max = __riscv_vfmax_vv_f32m8(v_max, v_in, vl);
                        }
                    }
                    __riscv_vse32_v_f32m8(out_row + ow, v_max, vl);
                    ow += vl;
                }
            }
        }
    }
}

// ============================================================
// 4. BIAS ADD
// ============================================================
void bias_add_e32m8(const float* input, const float* bias, float* output,
                      size_t batch_size, size_t channels,
                      size_t height, size_t width, int do_relu) {
    size_t channel_size = height * width;
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            float b_val = bias[c];
            size_t offset = (b * channels + c) * channel_size;
            const float* in_ptr = input + offset;
            float* out_ptr = output + offset;
            size_t cnt = channel_size;
            while (cnt > 0) {
                size_t vl = __riscv_vsetvl_e32m8(cnt);
                vfloat32m8_t v_in = __riscv_vle32_v_f32m8(in_ptr, vl);
                vfloat32m8_t v_out = __riscv_vfadd_vf_f32m8(v_in, b_val, vl);
                if(do_relu) {
                    vfloat32m8_t v_zero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
                    v_out = __riscv_vfmax_vv_f32m8(v_out, v_zero, vl);
                }
                __riscv_vse32_v_f32m8(out_ptr, v_out, vl);
                in_ptr += vl; out_ptr += vl; cnt -= vl;
            }
        }
    }
}

// ============================================================
// 5. TENSOR ADD
// ============================================================
void tensor_add_e32m8(const float* a, const float* b, float* out, size_t size) {
    size_t cnt = size;
    while (cnt > 0) {
        size_t vl = __riscv_vsetvl_e32m8(cnt);
        vfloat32m8_t v_a = __riscv_vle32_v_f32m8(a, vl);
        vfloat32m8_t v_b = __riscv_vle32_v_f32m8(b, vl);
        vfloat32m8_t v_o = __riscv_vfadd_vv_f32m8(v_a, v_b, vl);
        __riscv_vse32_v_f32m8(out, v_o, vl);
        a += vl; b += vl; out += vl; cnt -= vl;
    }
}

// ============================================================
// 6. DENSE
// ============================================================
void dense_e32m8(const float* input, const float* weights, const float* bias, float* output, int in_dim, int out_dim, int do_relu) {
    for (int m = 0; m < out_dim; ++m) {
        float sum = bias ? bias[m] : 0.0f;
        const float* w_row = weights + m * in_dim;
        int k = 0;
        vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        vfloat32m8_t v_sum_vec = __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvlmax_e32m8());
        for (; k < in_dim; ) {
            size_t vl = __riscv_vsetvl_e32m8(in_dim - k);
            vfloat32m8_t v_in = __riscv_vle32_v_f32m8(input + k, vl);
            vfloat32m8_t v_w = __riscv_vle32_v_f32m8(w_row + k, vl);
            v_sum_vec = __riscv_vfmacc_vv_f32m8(v_sum_vec, v_in, v_w, vl);
            k += vl;
        }
        vfloat32m1_t v_res = __riscv_vfredusum_vs_f32m8_f32m1(v_sum_vec, v_zero, __riscv_vsetvlmax_e32m8());
        sum += __riscv_vfmv_f_s_f32m1_f32(v_res);
        
        if (do_relu && sum < 0.0f) sum = 0.0f;
        
        output[m] = sum;
    }
}

// ============================================================
// 7. SCALAR SOFTMAX (Fixed for Spike)
// ============================================================
void softmax_scalar(float* input, float* output, size_t size) {
    if (size == 0) return;
    float max_val = input[0];
    for (size_t i = 1; i < size; i++) if (input[i] > max_val) max_val = input[i];
    
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        output[i] = exp_approx(input[i] - max_val); // Use local approx
        sum += output[i];
    }
    for (size_t i = 0; i < size; i++) output[i] /= sum;
}

}