#include <riscv_vector.h>
#include <cstring>
#include "defs.h"
#include "rvv_defs.hpp"

// RVV optimized version (e32m1)
void conv_transpose_2d_e32m1(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
    
    int out_height = (input_h - 1) * stride_h - 2 * pad_h + kernel_h;
    int out_width = (input_w - 1) * stride_w - 2 * pad_w + kernel_w;
    
    // Initialize output to zero
    size_t output_size = batch_size * out_channels * out_height * out_width;
    std::memset(output, 0, output_size * sizeof(float));
    
    for (int b = 0; b < batch_size; ++b) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int h = 0; h < input_h; ++h) {
                    for (int w = 0; w < input_w; ++w) {
                        float input_val = input[b * in_channels * input_h * input_w + 
                                              ic * input_h * input_w + h * input_w + w];
                        
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            int out_h_idx = h * stride_h - pad_h + kh;
                            
                            if (out_h_idx < 0 || out_h_idx >= out_height) {
                                continue;
                            }
                            
                            int kw = 0;
                            int out_w_idx = w * stride_w - pad_w;
                            
                            // Skip negative indices
                            while (kw < kernel_w && out_w_idx + kw < 0) {
                                kw++;
                            }
                            
                            size_t vl;
                            for (; kw < kernel_w; kw += vl) {
                                int remaining = kernel_w - kw;
                                int valid_end = out_width - out_w_idx - kw;
                                int processable = (remaining < valid_end) ? remaining : valid_end;
                                
                                if (processable <= 0) break;
                                
                                vl = SET_VECTOR_LENGTH<float, M1>(processable);
                                
                                // Load kernel values
                                auto v_kernel = VECTOR_LOAD<float, M1>(
                                    &kernel[ic * out_channels * kernel_h * kernel_w +
                                           oc * kernel_h * kernel_w +
                                           kh * kernel_w + kw], vl);
                                
                                // Load current output values
                                auto v_output = VECTOR_LOAD<float, M1>(
                                    &output[b * out_channels * out_height * out_width +
                                           oc * out_height * out_width +
                                           out_h_idx * out_width + out_w_idx + kw], vl);
                                
                                // Fused multiply-add: output = output + (input_val * kernel)
                                v_output = VECTOR_FMACC<float, M1>(v_output, input_val, v_kernel, vl);
                                
                                // Store back
                                VECTOR_STORE<float, M1>(
                                    &output[b * out_channels * out_height * out_width +
                                           oc * out_height * out_width +
                                           out_h_idx * out_width + out_w_idx + kw], v_output, vl);
                            }
                        }
                    }
                }
            }
        }
    }
}
