#include <riscv_vector.h>
#include <cstring>
#include <algorithm>
#include <iostream>
#include "defs.h"

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
                                
                                vl = __riscv_vsetvl_e32m1(processable);
                                
                                // Load input values
                                vfloat32m1_t v_input = __riscv_vle32_v_f32m1(
                                    &input[b * in_channels * input_h * input_w +
                                           ic * input_h * input_w +
                                           in_h * input_w + in_w + kw], vl);
                                
                                // Load kernel values (note: kernel layout is OIHW)
                                vfloat32m1_t v_kernel = __riscv_vle32_v_f32m1(
                                    &kernel[oc * in_channels * kernel_h * kernel_w +
                                           ic * kernel_h * kernel_w +
                                           kh * kernel_w + kw], vl);
                                
                                // Multiply and accumulate
                                vfloat32m1_t v_mult = __riscv_vfmul_vv_f32m1(v_input, v_kernel, vl);
                                
                                // Reduce sum horizontally
                                vfloat32m1_t v_sum = __riscv_vfredusum_vs_f32m1_f32m1(
                                    __riscv_vfmv_s_f_f32m1(0.0f, 1), v_mult, vl);
                                
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
                                
                                vl = __riscv_vsetvl_e32m2(processable);
                                
                                // Load input values
                                vfloat32m2_t v_input = __riscv_vle32_v_f32m2(
                                    &input[b * in_channels * input_h * input_w +
                                           ic * input_h * input_w +
                                           in_h * input_w + in_w + kw], vl);
                                
                                // Load kernel values
                                vfloat32m2_t v_kernel = __riscv_vle32_v_f32m2(
                                    &kernel[oc * in_channels * kernel_h * kernel_w +
                                           ic * kernel_h * kernel_w +
                                           kh * kernel_w + kw], vl);
                                
                                // Multiply and accumulate
                                vfloat32m2_t v_mult = __riscv_vfmul_vv_f32m2(v_input, v_kernel, vl);
                                
                                // Reduce sum horizontally
                                vfloat32m1_t v_sum = __riscv_vfredusum_vs_f32m2_f32m1(
                                    v_mult, __riscv_vfmv_s_f_f32m1(0.0f, 1), vl);
                                
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
                                
                                vl = __riscv_vsetvl_e32m4(processable);
                                
                                // Load input values
                                vfloat32m4_t v_input = __riscv_vle32_v_f32m4(
                                    &input[b * in_channels * input_h * input_w +
                                           ic * input_h * input_w +
                                           in_h * input_w + in_w + kw], vl);
                                
                                // Load kernel values
                                vfloat32m4_t v_kernel = __riscv_vle32_v_f32m4(
                                    &kernel[oc * in_channels * kernel_h * kernel_w +
                                           ic * kernel_h * kernel_w +
                                           kh * kernel_w + kw], vl);
                                
                                // Multiply and accumulate
                                vfloat32m4_t v_mult = __riscv_vfmul_vv_f32m4(v_input, v_kernel, vl);
                                
                                // Reduce sum horizontally
                                vfloat32m1_t v_sum = __riscv_vfredusum_vs_f32m4_f32m1(
                                    v_mult, __riscv_vfmv_s_f_f32m1(0.0f, 1), vl);
                                
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
                                
                                vl = __riscv_vsetvl_e32m8(processable);
                                
                                // Load input values
                                vfloat32m8_t v_input = __riscv_vle32_v_f32m8(
                                    &input[b * in_channels * input_h * input_w +
                                           ic * input_h * input_w +
                                           in_h * input_w + in_w + kw], vl);
                                
                                // Load kernel values
                                vfloat32m8_t v_kernel = __riscv_vle32_v_f32m8(
                                    &kernel[oc * in_channels * kernel_h * kernel_w +
                                           ic * kernel_h * kernel_w +
                                           kh * kernel_w + kw], vl);
                                
                                // Multiply and accumulate
                                vfloat32m8_t v_mult = __riscv_vfmul_vv_f32m8(v_input, v_kernel, vl);
                                
                                // Reduce sum horizontally
                                vfloat32m1_t v_sum = __riscv_vfredusum_vs_f32m8_f32m1(
                                    v_mult, __riscv_vfmv_s_f_f32m1(0.0f, 1), vl);
                                
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


// Simple scalar implementation for testing
void conv2d_scalar(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
    
    int out_height = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    
    std::memset(output, 0, batch_size * out_channels * out_height * out_width * sizeof(float));
    
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                int in_h = oh * stride_h - pad_h + kh;
                                int in_w = ow * stride_w - pad_w + kw;
                                
                                if (in_h >= 0 && in_h < input_h && in_w >= 0 && in_w < input_w) {
                                    float input_val = input[b * in_channels * input_h * input_w +
                                                           ic * input_h * input_w + in_h * input_w + in_w];
                                    float kernel_val = kernel[oc * in_channels * kernel_h * kernel_w +
                                                             ic * kernel_h * kernel_w + kh * kernel_w + kw];
                                    sum += input_val * kernel_val;
                                }
                            }
                        }
                    }
                    output[b * out_channels * out_height * out_width +
                           oc * out_height * out_width + oh * out_width + ow] = sum;
                }
            }
        }
    }
}

int main() {
    std::cout << "=== RVV Conv2D Test ===" << std::endl;
    
    // Simple test parameters
    const int batch_size = 1;
    const int in_channels = 2;
    const int out_channels = 4;
    const int input_h = 8;
    const int input_w = 8;
    const int kernel_h = 3;
    const int kernel_w = 3;
    const int stride_h = 1;
    const int stride_w = 1;
    const int pad_h = 1;
    const int pad_w = 1;
    
    // Calculate sizes
    const int input_size = batch_size * in_channels * input_h * input_w;
    const int kernel_size = out_channels * in_channels * kernel_h * kernel_w;
    const int out_height = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    const int out_width = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    const int output_size = batch_size * out_channels * out_height * out_width;
    
    std::cout << "Input: " << batch_size << "x" << in_channels << "x" << input_h << "x" << input_w << std::endl;
    std::cout << "Kernel: " << out_channels << "x" << in_channels << "x" << kernel_h << "x" << kernel_w << std::endl;
    std::cout << "Output: " << batch_size << "x" << out_channels << "x" << out_height << "x" << out_width << std::endl;
    
    // Allocate memory
    float* input = new float[input_size];
    float* kernel = new float[kernel_size];
    float* output_scalar = new float[output_size];
    float* output_rvv = new float[output_size];
    
    // Initialize with simple values
    for (int i = 0; i < input_size; ++i) {
        input[i] = static_cast<float>(i % 10) * 0.1f;
    }
    for (int i = 0; i < kernel_size; ++i) {
        kernel[i] = static_cast<float>((i % 5) + 1) * 0.2f;
    }
    
    // Test scalar implementation
    conv2d_scalar(input, kernel, output_scalar, batch_size, in_channels, out_channels,
                  input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
    
    // Test RVV implementation (e32m1)
    conv2d_e32m1(input, kernel, output_rvv, batch_size, in_channels, out_channels,
                 input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
    
    // Compare outputs
    bool correct = true;
    const float tolerance = 1e-5f;
    for (int i = 0; i < output_size; ++i) {
        if (std::abs(output_scalar[i] - output_rvv[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": scalar=" << output_scalar[i] 
                      << " rvv=" << output_rvv[i] << std::endl;
            correct = false;
            break;
        }
    }
    
    std::cout << "Correctness test: " << (correct ? "PASS" : "FAIL") << std::endl;
    
    // Print sample outputs
    std::cout << "First 5 scalar outputs: ";
    for (int i = 0; i < 5 && i < output_size; ++i) {
        std::cout << output_scalar[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "First 5 RVV outputs: ";
    for (int i = 0; i < 5 && i < output_size; ++i) {
        std::cout << output_rvv[i] << " ";
    }
    std::cout << std::endl;
    
    // Clean up
    delete[] input;
    delete[] kernel;
    delete[] output_scalar;
    delete[] output_rvv;
    
    std::cout << "Test completed!" << std::endl;
    return 0;
}
