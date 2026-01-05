#include "kernels.hpp"
#include <cfloat> 
#include "../../../lib/rvv_defs.hpp"

/****************** Specific Image Pre-processing Kernel ******************/
void preprocess_image(
    float* data, const float* scale, const float* bias,
    int channels, int height, int width){
		
    const size_t spatial_dim = height * width;
    const float s = scale[0];
    
    for (int c = 0; c < channels; ++c) {
        float b = bias[c];
        float* channel_data = data + c * spatial_dim;
        
        size_t i = 0;
        while (i < spatial_dim) {
            size_t vl = SET_VECTOR_LENGTH<float, M8>(spatial_dim - i);
            
            vfloat32m8_t v_data = VECTOR_LOAD<float, M8>(&channel_data[i], vl);
            v_data = VECTOR_FMACC<float, M8>(VECTOR_BROADCAST<float, M8>(b, vl), s, v_data, vl);
            VECTOR_STORE<float, M8>(&channel_data[i], v_data, vl);
            
            i += vl;
        }
    }
}

/************************************ CONV ************************************/
void conv2d(
    const float* input, float* output, const float* weights,
    int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int pad_top, int pad_left){

    // Use vector convolution with temporary buffers
    int out_h = (in_height + 2 * pad_top - kernel_size) / stride + 1;
    int out_w = (in_width + 2 * pad_left - kernel_size) / stride + 1;
    int K = in_channels * kernel_size * kernel_size;
    int N = out_h * out_w;
    
    float* col_buf = new float[K * N];
    float* gemm_buf = new float[out_channels * N];
    
    conv2d_im2col_gemm_m8(input, weights, nullptr, output, 
                          col_buf, gemm_buf,
                          in_channels, in_height, in_width,
                          out_channels, kernel_size, kernel_size,
                          pad_top, pad_left, stride, stride, 0);
    
    delete[] col_buf;
    delete[] gemm_buf;
}

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
                            v_acc = VECTOR_FMACC<float, M8>(v_acc, a_val, v_b, vl);
                        }
                        VECTOR_STORE<float, M8>(&c_row_ptr[j], v_acc, vl);
                        j += vl;
                    }
                }
            }
        }
    }
}

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
                            size_t vl = SET_VECTOR_LENGTH<float, M8>(out_width - ow);
                            vfloat32m8_t v_zero = VECTOR_BROADCAST<float, M8>(0.0f, vl);
                            VECTOR_STORE<float, M8>(dest_ptr + ow, v_zero, vl);
                            ow += vl;
                        }
                    } else {
                        for (int ow = 0; ow < out_width; ) {
                            size_t vl = SET_VECTOR_LENGTH<float, M8>(out_width - ow);
                            int in_w = ow * stride_w - pad_w + kw;
                            
                            const float* src_ptr = data_im + (c * height * width) + (in_h * width) + in_w;
                            
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

void conv2d_im2col_gemm_m8(
    const float* input, const float* kernel, const float* bias,
    float* output,
    float* col_buf, float* gemm_buf,
    int in_channels, int input_h, int input_w, 
    int out_channels, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int has_bias) {

    int out_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    
    int M = out_channels;
    int K = in_channels * kernel_h * kernel_w;
    int N = out_h * out_w;
    
    im2col_e32m8(input, col_buf,
                 in_channels, input_h, input_w, 
                 kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);

    gemm_blocked_e32m8(kernel, col_buf, gemm_buf, M, N, K, 
                       GEMM_BLOCK_M, GEMM_BLOCK_N, GEMM_BLOCK_K); 

    for (int m = 0; m < M; ++m) {
        float b_val = has_bias ? bias[m] : 0.0f;
        
        float* src_ptr = &gemm_buf[m * N];
        float* dst_ptr = &output[m * N];
        
        size_t n = 0;
        while (n < N) {
            size_t vl = SET_VECTOR_LENGTH<float, M8>(N - n);
            
            vfloat32m8_t v_data = VECTOR_LOAD<float, M8>(&src_ptr[n], vl);
            
            if (has_bias) {
                v_data = VECTOR_ADD<float, M8>(v_data, b_val, vl);
            }
            
            VECTOR_STORE<float, M8>(&dst_ptr[n], v_data, vl);
            n += vl;
        }
    }
}

/************************************ Bias Add ************************************/
void bias_add_e32m8(const float* input, const float* bias, float* output,
                       size_t channels, size_t channel_size) {
    
    const float* in_ptr  = input;
    float* out_ptr       = output;

    for (size_t c = 0; c < channels; ++c) {
        float b_val = bias[c]; 
        size_t cnt = channel_size;
        
        while (cnt > 0) {
            size_t vl = SET_VECTOR_LENGTH<float, M8>(cnt);
            
            auto v_input = VECTOR_LOAD<float, M8>(in_ptr, vl);
            auto v_output = VECTOR_ADD<float, M8>(v_input, b_val, vl);
            VECTOR_STORE<float, M8>(out_ptr, v_output, vl);
            
            in_ptr  += vl;
            out_ptr += vl;
            cnt     -= vl;
        }
    }
}

/************************************ Batch Norm ************************************/
void batch_norm_e32m8(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon) {
    int spatial_dim = height * width;
    for (int c = 0; c < channels; ++c) {
        float alpha = scale[c] / std::sqrt(variance[c] + epsilon);
        float beta = bias[c] - mean[c] * alpha;
        const float* in_ptr = input + c * spatial_dim;
        float* out_ptr = output + c * spatial_dim;
        for (size_t i = 0; i < (size_t)spatial_dim; ) {
            size_t vl = SET_VECTOR_LENGTH<float, M8>(spatial_dim - i);
            auto v = VECTOR_LOAD<float, M8>(in_ptr + i, vl);
            v = VECTOR_ADD<float, M8>(VECTOR_MUL<float, M8>(v, alpha, vl), beta, vl);
            VECTOR_STORE<float, M8>(out_ptr + i, v, vl);
            i += vl;
        }
    }
}

/************************************ Maxpool ************************************/
void maxpool_e32m8(const float* input, float* output,
                           int batch, int channels,
                           int in_h, int in_w,
                           int k_h, int k_w,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w) {

    int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            const float* in_ptr_base = input + (b * channels + c) * in_h * in_w;
            float* out_ptr_base = output + (b * channels + c) * out_h * out_w;

            for (int oh = 0; oh < out_h; ++oh) {
                int ih_start = oh * stride_h - pad_h;

                for (int ow = 0; ow < out_w; ) {
                    size_t vl = SET_VECTOR_LENGTH<float, M8>(out_w - ow);
                    vfloat32m8_t v_max = VECTOR_BROADCAST<float, M8>(-FLT_MAX, vl);

                    for (int kh = 0; kh < k_h; ++kh) {
                        int ih = ih_start + kh;
                        if (ih < 0 || ih >= in_h) continue;

                        for (int kw = 0; kw < k_w; ++kw) {
                            // Calculate current input width for this vector segment
                            int iw_base = ow * stride_w - pad_w + kw;
                            
                            // Note: For a production library, you could use a mask here 
                            // to handle pixels that fall into 'pad_w' areas.
                            // For simplicity/speed, we assume standard padding.
                            const float* load_addr = in_ptr_base + ih * in_w + iw_base;

                            vfloat32m8_t v_in;
                            if (stride_w == 1) {
                                v_in = VECTOR_LOAD<float, M8>(load_addr, vl);
                            } else {
                                v_in = VECTOR_STRIDED_LOAD<float, M8>(load_addr, stride_w * sizeof(float), vl);
                            }
                            v_max = VECTOR_MAX<float, M8>(v_max, v_in, vl);
                        }
                    }
                    VECTOR_STORE<float, M8>(out_ptr_base + oh * out_w + ow, v_max, vl);
                    ow += vl;
                }
            }
        }
    }
}


void maxpool_e32m8_fixed(const float* input, float* output,
                         int batch, int channels,
                         int in_h, int in_w,
                         int out_h, int out_w,
                         int k_h, int k_w,
                         int stride_h, int stride_w,
                         int pad_h, int pad_w) {

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            const float* in_ptr_base = input + (b * channels + c) * in_h * in_w;
            float* out_ptr_base = output + (b * channels + c) * out_h * out_w;

            for (int oh = 0; oh < out_h; ++oh) {
                int ih_start = oh * stride_h - pad_h;
                float* out_row = out_ptr_base + oh * out_w;

                int ow = 0;
                // Fast scalar path (stride=1 fallback logic)
                for (; ow <= out_w - 4; ow += 4) {
                    float m0 = -FLT_MAX, m1 = -FLT_MAX, m2 = -FLT_MAX, m3 = -FLT_MAX;
                    for (int kh = 0; kh < k_h; ++kh) {
                        int ih = ih_start + kh;
                        if (ih >= 0 && ih < in_h) {
                            const float* in_line = in_ptr_base + ih * in_w;
                            for (int kw = 0; kw < k_w; ++kw) {
                                int iw0 = ow * stride_w - pad_w + kw;
                                // We check each iw independently to mirror your working kernel
                                if (iw0 >= 0 && iw0 < in_w) m0 = std::max(m0, in_line[iw0]);
                                if (iw0 + 1 >= 0 && iw0 + 1 < in_w) m1 = std::max(m1, in_line[iw0 + 1]);
                                if (iw0 + 2 >= 0 && iw0 + 2 < in_w) m2 = std::max(m2, in_line[iw0 + 2]);
                                if (iw0 + 3 >= 0 && iw0 + 3 < in_w) m3 = std::max(m3, in_line[iw0 + 3]);
                            }
                        }
                    }
                    out_row[ow] = m0; out_row[ow+1] = m1; out_row[ow+2] = m2; out_row[ow+3] = m3;
                }
                // Tail
                for (; ow < out_w; ++ow) {
                    float m = -FLT_MAX;
                    for (int kh = 0; kh < k_h; ++kh) {
                        int ih = ih_start + kh;
                        if (ih >= 0 && ih < in_h) {
                            const float* in_line = in_ptr_base + ih * in_w;
                            for (int kw = 0; kw < k_w; ++kw) {
                                int iw = ow * stride_w - pad_w + kw;
                                if (iw >= 0 && iw < in_w) m = std::max(m, in_line[iw]);
                            }
                        }
                    }
                    out_row[ow] = m;
                }
            }
        }
    }
}


/************************************ LeakyRelu ************************************/
void leaky_relu_e32m8(const float* src, float* dest, size_t n, float alpha) {
	size_t vl = SET_VECTOR_LENGTH<float, M8>(n);
	while (n >= vl * 2) {
		auto v0 = VECTOR_LOAD<float, M8>(src + vl*0, vl);
		auto v1 = VECTOR_LOAD<float, M8>(src + vl*1, vl);

		VECTOR_STORE<float, M8>(dest + vl*0, VECTOR_MUL_MASKED<float, M8>(__riscv_vmflt_vf_f32m8_b4(v0, 0.0f, vl), v0, alpha, vl), vl);
		VECTOR_STORE<float, M8>(dest + vl*1, VECTOR_MUL_MASKED<float, M8>(__riscv_vmflt_vf_f32m8_b4(v1, 0.0f, vl), v1, alpha, vl), vl);

		src += vl * 2; dest += vl * 2; n -= vl * 2;
	}
	while (n > 0) {
		vl = SET_VECTOR_LENGTH<float, M8>(n);
		auto v = VECTOR_LOAD<float, M8>(src, vl);
		VECTOR_STORE<float, M8>(dest, VECTOR_MUL_MASKED<float, M8>(__riscv_vmflt_vf_f32m8_b4(v, 0.0f, vl), v, alpha, vl), vl);
		src += vl; dest += vl; n -= vl;
	}
}


