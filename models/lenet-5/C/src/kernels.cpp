#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>   // For std::iota
#include <algorithm> // For std::max_element, std::distance
#include <cstddef>   // For size_t
#include <stdexcept> // For std::runtime_error
#include <cmath>     // For exp
#include <algorithm> // For std::min

#include <cassert>   // For assert()
#include <cstring>   // For std::memset
#include <cfloat>    // For FLT_MAX
#include "rvv_defs.hpp"

#include "../include/defs.hpp"

std::vector<float> load_weights(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Error: Cannot open file: " + filename);
    }
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size % sizeof(float) != 0) {
        throw std::runtime_error("Error: File size is not a multiple of float: " + filename);
    }
    std::vector<float> buffer(size / sizeof(float));
    if (!in.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Error reading data from file: " + filename);
    }
    return buffer;
}

void load_preprocessed_image(std::vector<float>& img_buffer, const std::string& filename) {
    std::cout << "Loading image: " << filename << std::endl;
    std::vector<float> loaded_data = load_weights(filename);
    if (loaded_data.size() != img_buffer.size()) {
        throw std::runtime_error("Image file size mismatch. Expected " + 
            std::to_string(img_buffer.size()) + " floats, but file has " +
            std::to_string(loaded_data.size()));
    }
    std::copy(loaded_data.begin(), loaded_data.end(), img_buffer.begin());
}

// =======================================================
// KERNELS
// =======================================================

#define MIN(a,b) (((a)<(b))?(a):(b))

void im2col_e32m8(
    const float* data_im,
    float* data_col,
    int channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w)
{
    const int out_height =
        (height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int out_width =
        (width + 2 * pad_w - kernel_w) / stride_w + 1;
    const int out_area = out_height * out_width;

    for (int c = 0; c < channels; ++c) {
        const float* im_c = data_im + c * height * width;

        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {

                const int col_row =
                    (c * kernel_h * kernel_w) + (kh * kernel_w + kw);
                float* col_ptr = data_col + col_row * out_area;

                for (int oh = 0; oh < out_height; ++oh) {
                    const int ih = oh * stride_h - pad_h + kh;
                    float* dst = col_ptr + oh * out_width;

                    // If height is out of bounds → whole row is zero
                    if (ih < 0 || ih >= height) {
                        int ow = 0;
                        while (ow < out_width) {
                            size_t vl =
                                SET_VECTOR_LENGTH<float, M8>(out_width - ow);
                            vfloat32m8_t vz =
                                VECTOR_BROADCAST<float, M8>(0.0f, vl);
                            VECTOR_STORE<float, M8>(dst + ow, vz, vl);
                            ow += vl;
                        }
                        continue;
                    }

                    // Height valid → check width per element
                    const float* im_row = im_c + ih * width;

                    int ow = 0;
                    while (ow < out_width) {
                        size_t vl =
                            SET_VECTOR_LENGTH<float, M8>(out_width - ow);

                        int iw0 = ow * stride_w - pad_w + kw;

                        // ---------- FAST PATH ----------
                        if (stride_w == 1 &&
                            iw0 >= 0 &&
                            iw0 + (int)vl <= width) {

                            const float* src = im_row + iw0;
                            vfloat32m8_t v =
                                VECTOR_LOAD<float, M8>(src, vl);
                            VECTOR_STORE<float, M8>(dst + ow, v, vl);
                        }
                        // ---------- SAFE PATH ----------
                        else {
                            float tmp[256]; // enough for any RVV VL

                            for (size_t i = 0; i < vl; ++i) {
                                int iw = iw0 + (int)i * stride_w;
                                if (iw >= 0 && iw < width) {
                                    tmp[i] = im_row[iw];
                                } else {
                                    tmp[i] = 0.0f;
                                }
                            }

                            vfloat32m8_t v =
                                VECTOR_LOAD<float, M8>(tmp, vl);
                            VECTOR_STORE<float, M8>(dst + ow, v, vl);
                        }

                        ow += vl;
                    }
                }
            }
        }
    }
}

void gemm_blocked_e32m8(const float* A, const float* B, float* C,
                        int M, int N, int K,
                        int BM, int BN, int BK) {
    std::memset(C, 0, M * N * sizeof(float));
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

void conv2d(
    const float* input, float* output, const float* weights,
    int batch,
    int in_channels, int in_height, int in_width,
    int out_channels,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    int out_h = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_width  + 2 * pad_w - kernel_w) / stride_w + 1;

    int K = in_channels * kernel_h * kernel_w;
    int N = out_h * out_w;

    float* col_buf  = new float[K * N];
    float* gemm_buf = new float[out_channels * N];

    for (int n = 0; n < batch; ++n) {
        const float* in_ptr  = input  + n * in_channels * in_height * in_width;
        float* out_ptr = output + n * out_channels * out_h * out_w;

        conv2d_im2col_gemm_m8(
            in_ptr, weights, nullptr, out_ptr,
            col_buf, gemm_buf,
            in_channels, in_height, in_width,
            out_channels, kernel_h, kernel_w,
            pad_h, pad_w, stride_h, stride_w,
            0);
    }

    delete[] col_buf;
    delete[] gemm_buf;
}

void tensor_add_e32m8(const float* input_a, const float* input_b, float* output,
                           size_t size) {
    const float* in_a_ptr = input_a;
    const float* in_b_ptr = input_b;
    float* out_ptr = output;
    
    size_t cnt = size;
    size_t vl;

    while (cnt > 0) {
        vl = SET_VECTOR_LENGTH<float, M8>(cnt);
        auto v_a = VECTOR_LOAD<float, M8>(in_a_ptr, vl);
        auto v_b = VECTOR_LOAD<float, M8>(in_b_ptr, vl);
        auto v_out = VECTOR_ADD<float, M8>(v_a, v_b, vl);
        VECTOR_STORE<float, M8>(out_ptr, v_out, vl);

        in_a_ptr += vl;
        in_b_ptr += vl;
        out_ptr += vl;
        cnt -= vl;
    }
}

void softmax(
    const float* input,
    float* output,
    size_t n
) {
    // Find max value (numerical stability)
    float max_val = -FLT_MAX;
    for (size_t i = 0; i < n; ++i) {
        if (input[i] > max_val)
            max_val = input[i];
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float e = expf(input[i] - max_val);
        output[i] = e;
        sum += e;
    }

    // Normalize
    for (size_t i = 0; i < n; ++i) {
        output[i] /= sum;
    }
}

void bias_add_e32m8(const float* input, const float* bias, float* output,
                       size_t channels, size_t channel_size) {
    
    // We use temporary pointers to traverse the memory linearly
    const float* in_ptr  = input;
    float* out_ptr       = output;

    for (size_t c = 0; c < channels; ++c) {
        float b_val = bias[c]; // Scalar bias for this channel
        size_t cnt = channel_size;
        
        // Ara processes the channel's spatial data here
        while (cnt > 0) {
            size_t vl = SET_VECTOR_LENGTH<float, M8>(cnt);
            
            auto v_input = VECTOR_LOAD<float, M8>(in_ptr, vl);
            auto v_output = VECTOR_ADD<float, M8>(v_input, b_val, vl);
            VECTOR_STORE<float, M8>(out_ptr, v_output, vl);
            
            in_ptr  += vl;
            out_ptr += vl;
            cnt     -= vl;
        }
        // When this while loop ends, in_ptr and out_ptr are already
        // perfectly positioned for the start of the next channel.
    }
}

void relu_e32m8(float* input, float* output, size_t size) {
    float* in_ptr = input;
    float* out_ptr = output;
    
	size_t vlmax = SET_VECTOR_LENGTH_MAX<float, M8>();    
	auto v_zero = VECTOR_MOVE<float, M8>(0.0f, vlmax);
    
    for (size_t cnt = size; cnt > 0; ) {
        // Set VL for the current chunk
        size_t vl = SET_VECTOR_LENGTH<float, M8>(cnt);
        
        // Load chunk from memory
        auto v_input = VECTOR_LOAD<float, M8>(in_ptr, vl);
        
        // Vector MAX: result = max(input, 0.0)
        auto v_result = VECTOR_MAX<float, M8>(v_input, v_zero, vl);
        
        // Store results back to memory
        VECTOR_STORE<float, M8>(out_ptr, v_result, vl);
        
        // Update pointers and remaining count
        cnt -= vl;
        in_ptr += vl;
        out_ptr += vl;
    }
}

void dense_e32m8(const float* input, const float* weights, const float* bias,
	float* output, size_t in_features, size_t out_features) {
	 
	 size_t K = in_features;
	 size_t N = out_features;
 
	 // We iterate through outputs (j)
	 for (size_t j_idx = 0; j_idx < N; ) {
		 size_t vl = SET_VECTOR_LENGTH<float, M8>(N - j_idx);
 
		 // 1. Initialize with bias
		 auto v_acc = VECTOR_LOAD<float, M8>(&bias[j_idx], vl);
 
		 for (size_t k = 0; k < K; k++) {
			 
			 // This requires a STRIDED LOAD
			 auto v_w = VECTOR_STRIDED_LOAD<float, M8>(&weights[j_idx * K + k], K * sizeof(float), vl);
			 auto v_in = VECTOR_MOVE<float, M8>(input[k], vl);
			 v_acc = VECTOR_FMACC<float, M8>(v_acc, v_in, v_w, vl);
		 }
 
		 VECTOR_STORE<float, M8>(&output[j_idx], v_acc, vl);
		 j_idx += vl;
	 }
 }

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
