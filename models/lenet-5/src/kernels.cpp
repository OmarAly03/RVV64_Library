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
#include "../include/exp.h"
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
// KERNEL HEADERS & PLACEHOLDERS
// =======================================================

// --- Your Provided Kernels ---
inline int conv_output_size(int input_size, int kernel_size, int stride, int pad) {
    return (input_size + 2 * pad - kernel_size) / stride + 1;
}


void conv2d_scalar(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
    
    // Parameter validation
    assert(input != nullptr && kernel != nullptr && output != nullptr);
    assert(batch_size > 0 && in_channels > 0 && out_channels > 0);
    assert(input_h > 0 && input_w > 0);
    assert(kernel_h > 0 && kernel_w > 0);
    assert(stride_h > 0 && stride_w > 0);
    assert(pad_h >= 0 && pad_w >= 0);
    
    // Calculate output spatial dimensions using helper function
    int out_height = conv_output_size(input_h, kernel_h, stride_h, pad_h);
    int out_width = conv_output_size(input_w, kernel_w, stride_w, pad_w);
    
    // Ensure output dimensions are positive
    assert(out_height > 0 && out_width > 0);
    
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
                                // Note: Outside bounds contributes 0 (implicit zero-padding)
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

void maxpool_scalar_tile(
    const float* X, float* Y, int64_t* I,
    size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode,
    size_t OH, size_t OW,
    size_t tile_oh_start, size_t tile_ow_start,
    size_t tile_oh_end, size_t tile_ow_end)
{
    // (Scalar code remains the same)
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t oh = tile_oh_start; oh < tile_oh_end; ++oh) {
                for (size_t ow = tile_ow_start; ow < tile_ow_end; ++ow) {
                    float max_val = -FLT_MAX;
                    int64_t max_idx = -1;
                    for (size_t kh = 0; kh < K; ++kh) {
                        for (size_t kw = 0; kw < K; ++kw) {
                            size_t ih = oh * S + kh;
                            size_t iw = ow * S + kw;
                            if (ih < H && iw < W) {
                                float val = X[n*C*H*W + c*H*W + ih*W + iw];
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = (ih * W + iw);
                                }
                            }
                        }
                    }
                    Y[n*C*OH*OW + c*OH*OW + oh*OW + ow] = max_val;
                    I[n*C*OH*OW + c*OH*OW + oh*OW + ow] = (max_idx != -1) ? (n * C * H * W) + (c * H * W) + max_idx : -1;
                }
            }
        }
    }
}

void relu_scalar(float* input, float* output, size_t size) {
    for (size_t i = 0; i < size; i++) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

void dense_scalar(const float* input, const float* weights, const float* bias,
                        float* output, size_t in_features, size_t out_features) {
    // Implements Y = A*B^T + C, where A=input, B=weights, C=bias
    // A shape: [in_features]
    // B shape: [out_features, in_features]
    // C shape: [out_features]
    // Y shape: [out_features]
    for (size_t out_f = 0; out_f < out_features; ++out_f) {
        float sum = 0.0f;
        for (size_t in_f = 0; in_f < in_features; ++in_f) {
            sum += input[in_f] * weights[out_f * in_features + in_f];
        }
        output[out_f] = sum + bias[out_f];
    }
}


void bias_add_scalar(const float* input, const float* bias, float* output,
                       size_t batch_size, size_t channels,
                       size_t height, size_t width) {
    
    // Size of one 2D feature map
    size_t channel_size = height * width; 
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            // Get the scalar bias value for this channel
            float b_val = bias[c]; 
            // Calculate the starting offset for this channel
            size_t offset = (b * channels + c) * channel_size;
            
            const float* in_ptr = input + offset;
            float* out_ptr = output + offset;
            
            // This is the loop we will vectorize
            for (size_t i = 0; i < channel_size; ++i) {
                out_ptr[i] = in_ptr[i] + b_val;
            }
        }
    }
}

void tensor_add_scalar(const float* input_a, const float* input_b, float* output,
                           size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = input_a[i] + input_b[i];
    }
}

void softmax_scalar(float* input, float* output, size_t size) {
    // Pass 1: Find Max
    float max_val = -__builtin_inff();
    for (size_t i = 0; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // Pass 2: Calculate Exponentials and Sum
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    // Pass 3: Divide by Sum
    for (size_t i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

/********************************************************************************************/

void dense_e32m8(const float* input, const float* weights, const float* bias,
                   float* output, size_t in_features, size_t out_features) {
    for (size_t out_f = 0; out_f < out_features; ++out_f) {
        
        const float* a_ptr = input;
        const float* b_ptr = &weights[out_f * in_features];
        size_t cnt = in_features;
        size_t vl;

        vfloat32m8_t v_sum = __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvl_e32m8(in_features));

        for (; cnt > 0; cnt -= vl) {
            vl = __riscv_vsetvl_e32m8(cnt);
            vfloat32m8_t v_a = __riscv_vle32_v_f32m8(a_ptr, vl);
            vfloat32m8_t v_b = __riscv_vle32_v_f32m8(b_ptr, vl);
            v_sum = __riscv_vfmacc_vv_f32m8(v_sum, v_a, v_b, vl);
            a_ptr += vl;
            b_ptr += vl;
        }

        vfloat32m1_t v_scalar_sum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
        v_scalar_sum = __riscv_vfredusum_vs_f32m8_f32m1(v_sum, v_scalar_sum, __riscv_vsetvl_e32m8(in_features));
        float sum = __riscv_vfmv_f_s_f32m1_f32(v_scalar_sum);

        output[out_f] = sum + bias[out_f];
    }
}

void tensor_add_e32m8(const float* input_a, const float* input_b, float* output,
                           size_t size) {
    const float* in_a_ptr = input_a;
    const float* in_b_ptr = input_b;
    float* out_ptr = output;
    
    size_t cnt = size;
    size_t vl;

    while (cnt > 0) {
        vl = __riscv_vsetvl_e32m8(cnt);
        vfloat32m8_t v_a = __riscv_vle32_v_f32m8(in_a_ptr, vl);
        vfloat32m8_t v_b = __riscv_vle32_v_f32m8(in_b_ptr, vl);
        vfloat32m8_t v_out = __riscv_vfadd_vv_f32m8(v_a, v_b, vl);
        __riscv_vse32_v_f32m8(out_ptr, v_out, vl);
        in_a_ptr += vl;
        in_b_ptr += vl;
        out_ptr += vl;
        cnt -= vl;
    }
}

void softmax_vec(const float *i, float *o, uint64_t channels,
                 uint64_t innerSize) {

  size_t avl = innerSize;
  size_t vl;

  // Stripmining pointers
  float *_i = (float *)i;
  float *_o = (float *)o;
  // Channel pointers
  float *__i = (float *)i;
  float *__o = (float *)o;

  // Vector registers
  vfloat32m1_t max_chunk_v;
  vfloat32m1_t buf_chunk_v;
  vfloat32m1_t num_chunk_v;
  vfloat32m1_t den_chunk_v;
  vfloat32m1_t res_chunk_v;

  // Stripmine on innerSize
  for (vl = __riscv_vsetvl_e32m1(avl); avl > 0; avl -= vl) {

    vl = __riscv_vsetvl_e32m1(avl);

    /*
      Calculate the maximum along the channel dimension
    */

    // Initialize the max vector
    max_chunk_v = __riscv_vle32_v_f32m1(__i, vl);
    // Bump the pointer
    __i += innerSize;
    for (uint64_t ch = 1; ch < channels; ++ch) {
      // Load a chunk of the input vector
      buf_chunk_v = __riscv_vle32_v_f32m1(__i, vl);
      // Bump the channel pointer
      __i += innerSize;
      // Calculate the elm-wise maximum between the two chunks
      max_chunk_v = __riscv_vfmax_vv_f32m1(max_chunk_v, buf_chunk_v, vl);
    }
    // Restore the channel pointer
    __i = _i;

    /*
      Fetch, subtract, exponentiate along the channel dimension
    */

    // Initialize accumulator
    den_chunk_v = __riscv_vfmv_v_f_f32m1(0, vl);
    for (uint64_t ch = 0; ch < channels; ++ch) {
      // Fetch one chunk from channel ch
      buf_chunk_v = __riscv_vle32_v_f32m1(__i, vl);
      // Subtract the maximum
      buf_chunk_v = __riscv_vfsub_vv_f32m1(buf_chunk_v, max_chunk_v, vl);
      // Exponentiate
      buf_chunk_v = __exp_2xf32(buf_chunk_v, vl);
      // Store the numerator to memory
      __riscv_vse32_v_f32m1(__o, buf_chunk_v, vl);
      // Accumulate
      den_chunk_v = __riscv_vfadd_vv_f32m1(den_chunk_v, buf_chunk_v, vl);
      // Bump channel pointers
      __i += innerSize;
      __o += innerSize;
    }
    // Restore the pointers
    __i = _i;
    __o = _o;

    /*
      Divide by the computed sum
    */

    for (uint64_t ch = 0; ch < channels; ++ch) {
      // Load numerator from memory
      num_chunk_v = __riscv_vle32_v_f32m1(__o, vl);
      // Divide
      res_chunk_v = __riscv_vfdiv_vv_f32m1(num_chunk_v, den_chunk_v, vl);
      // Store the result to memory
      __riscv_vse32_v_f32m1(__o, res_chunk_v, vl);
      // Bump channel pointers
      __o += innerSize;
    }
    // Bump stripmining pointers
    _i += vl;
    _o += vl;
    // Reset channel pointers
    __i = _i;
    __o = _o;
  }
}

void relu_e32m8(float* input, float* output, size_t size) {
    float* in_ptr = input;
    float* out_ptr = output;
    
    for (size_t cnt = size; cnt > 0; ) {
        size_t vl = SET_VECTOR_LENGTH<float, M8>(cnt);
        auto v_input = VECTOR_LOAD<float, M8>(in_ptr, vl);
        auto v_zero = VECTOR_MOVE<float, M8>(0.0f, vl);
        auto v_result = VECTOR_MAX<float, M8>(v_input, v_zero, vl);
        VECTOR_STORE<float, M8>(out_ptr, v_result, vl);
        
        cnt -= vl;
        in_ptr += vl;
        out_ptr += vl;
    }
}

void bias_add_e32m8(const float* input, const float* bias, float* output,
                      size_t batch_size, size_t channels,
                      size_t height, size_t width) {
    
    size_t channel_size = height * width;
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            float b_val = bias[c];
            size_t offset = (b * channels + c) * channel_size;
            
            const float* in_ptr = input + offset;
            float* out_ptr = output + offset;
            
            size_t cnt = channel_size;
            size_t vl;
            
            while (cnt > 0) {
                vl = __riscv_vsetvl_e32m8(cnt);
                vfloat32m8_t v_input = __riscv_vle32_v_f32m8(in_ptr, vl);
                vfloat32m8_t v_output = __riscv_vfadd_vf_f32m8(v_input, b_val, vl);
                __riscv_vse32_v_f32m8(out_ptr, v_output, vl);
                
                in_ptr += vl;
                out_ptr += vl;
                cnt -= vl;
            }
        }
    }
}

// ================== NEW MAXPOOL FUNCTIONS ==================

// Helper macro for calculating output dimensions for pooling
#define CALC_OUT_DIM(in_dim, k, s, ceil_mode) \
    (ceil_mode ? ((in_dim - k + s - 1) / s + 1) : ((in_dim - k) / s + 1))

// Tiling parameters (adjust as needed)
#define TILE_H 14
#define TILE_W 256 // Example: Process 256 output elements at a time horizontally

/**
 * @brief Processes a single tile of the maxpool operation (Vectorized)
 */
void maxpool_e32m8_tile(
    const float* X, float* Y, int64_t* I,
    size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode,
    size_t OH, size_t OW,
    size_t tile_oh_start, size_t tile_ow_start,
    size_t tile_oh_end, size_t tile_ow_end)
{
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const float* x_channel = X + (n * C + c) * H * W;
            float* y_channel = Y + (n * C + c) * OH * OW;
            int64_t* i_channel = I + (n * C + c) * OH * OW;
            for (size_t oh = tile_oh_start; oh < tile_oh_end; ++oh) {
                for (size_t ow = tile_ow_start; ow < tile_ow_end; ) {
                    size_t current_tile_width = tile_ow_end - ow;
                    size_t vl = __riscv_vsetvl_e32m8(current_tile_width);

                    vfloat32m8_t max_vec = __riscv_vfmv_v_f_f32m8(-FLT_MAX, vl);
                    vint32m8_t max_idx_vec32 = __riscv_vmv_v_x_i32m8(-1, vl);
                    for (size_t kh = 0; kh < K; ++kh) {
                        for (size_t kw = 0; kw < K; ++kw) {
                            size_t ih = oh * S + kh;
                            if (ih < H) {
                                // Start of the horizontal stripe for this kernel pixel
                                const float* x_ptr = x_channel + ih * W + ow * S + kw;
                                
                                // Create a vector of input width indices (iw)
                                // iw = ow*S + kw + [0, 1, ..., vl-1] * S
                                vuint32m8_t element_indices = __riscv_vid_v_u32m8(vl);
                                vuint32m8_t iw_vec = __riscv_vadd_vx_u32m8(__riscv_vmul_vx_u32m8(element_indices, S, vl), ow * S + kw, vl);
                                
                                // Create mask for elements where iw < W
                                vbool4_t load_mask = __riscv_vmsltu_vx_u32m8_b4(iw_vec, W, vl);

                                // Load elements using strided load
                                vfloat32m8_t x_vec = __riscv_vlse32_v_f32m8_m(load_mask, x_ptr, S * sizeof(float), vl);
                                
                                // Find where new x_vec is greater than current max
                                vbool4_t is_greater_mask = __riscv_vmfgt_vv_f32m8_b4_m(load_mask, x_vec, max_vec, vl);
                                
                                // Update max_vec
                                max_vec = __riscv_vfmax_vv_f32m8_m(is_greater_mask, max_vec, x_vec, vl);
                                
                                // Update indices
                                // current_indices = (ih * W + kw) + (ow * S) + [0, 1, ..., vl-1] * S
                                // No, current_indices = ih * W + iw_vec
                                // current_indices = ih * W + (ow * S + kw) + element_indices * S
                                int32_t current_idx_base = ih * W;
                                vuint32m8_t offsets = __riscv_vmul_vx_u32m8(element_indices, S, vl);
                                // We use vint32m8_t because vadd requires signed
                                vint32m8_t current_indices = __riscv_vadd_vx_i32m8(__riscv_vreinterpret_v_u32m8_i32m8(offsets), current_idx_base + ow * S + kw, vl);
                                
                                max_idx_vec32 = __riscv_vmerge_vvm_i32m8(max_idx_vec32, current_indices, is_greater_mask, vl);
                            }
                        }
                    }
                    // Store the max values
                    __riscv_vse32_v_f32m8(y_channel + oh * OW + ow, max_vec, vl);
                    
                    // --- Widen and Store Indices ---
                    int64_t channel_offset = (n * C + c) * H * W;

                    // Widen the 32-bit indices to 64-bit in two halves
                    size_t current_vl_m8 = vl; // Save original vl
                    size_t half_vl_m4 = __riscv_vsetvl_e32m4(current_vl_m8); // Use m4's vl for splitting m8

                    vint32m4_t lo_idx32 = __riscv_vget_v_i32m8_i32m4(max_idx_vec32, 0);
                    vint32m4_t hi_idx32 = __riscv_vget_v_i32m8_i32m4(max_idx_vec32, 1);

                    // Create masks for valid indices (-1 means no valid element was found)
                    vbool8_t lo_valid_mask = __riscv_vmsne_vx_i32m4_b8(lo_idx32, -1, half_vl_m4);
                    vbool8_t hi_valid_mask = __riscv_vmsne_vx_i32m4_b8(hi_idx32, -1, current_vl_m8 - half_vl_m4);

                    // Widen, add offset, and merge
                    vint64m8_t widened_lo = __riscv_vsext_vf2_i64m8(lo_idx32, half_vl_m4);
                    vint64m8_t lo_idx64_added = __riscv_vadd_vx_i64m8(widened_lo, channel_offset, half_vl_m4);
                    vint64m8_t lo_idx64 = __riscv_vmerge_vvm_i64m8(__riscv_vmv_v_x_i64m8(-1, half_vl_m4), lo_idx64_added, lo_valid_mask, half_vl_m4);


                    vint64m8_t widened_hi = __riscv_vsext_vf2_i64m8(hi_idx32, current_vl_m8 - half_vl_m4);
                    vint64m8_t hi_idx64_added = __riscv_vadd_vx_i64m8(widened_hi, channel_offset, current_vl_m8 - half_vl_m4);
                    vint64m8_t hi_idx64 = __riscv_vmerge_vvm_i64m8(__riscv_vmv_v_x_i64m8(-1, current_vl_m8 - half_vl_m4), hi_idx64_added, hi_valid_mask, current_vl_m8 - half_vl_m4);


                    vl = __riscv_vsetvl_e32m8(current_tile_width); // Restore original vl for storage
                    
                    // Store the 64-bit indices
                    __riscv_vse64_v_i64m8(i_channel + oh * OW + ow, lo_idx64, half_vl_m4);
                    __riscv_vse64_v_i64m8(i_channel + oh * OW + ow + half_vl_m4, hi_idx64, vl - half_vl_m4);

                    ow += vl;
                }
            }
        }
    }
}

/**
 * @brief Tiled wrapper for the vectorized maxpool operation
 */
void maxpool_e32m8_tiled(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
    size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
    size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
    for (size_t oh_base = 0; oh_base < OH; oh_base += TILE_H) {
        for (size_t ow_base = 0; ow_base < OW; ow_base += TILE_W) {
            size_t oh_end = std::min(oh_base + TILE_H, OH);
            size_t ow_end = std::min(ow_base + TILE_W, OW);
            maxpool_e32m8_tile(X, Y, I, N, C, H, W, K, S, ceil_mode, OH, OW, oh_base, ow_base, oh_end, ow_end);
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
                                vfloat32m1_t v_sum = __riscv_vfredusum_vs_f32m8_f32m1(v_mult, v_zero, vl);

                                // Extract scalar sum and add to total
                                sum += __riscv_vfmv_f_s_f32m1_f32(v_sum);
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