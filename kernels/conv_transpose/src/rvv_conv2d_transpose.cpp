#include <cstring>
#include <cstdint>
#include "defs.h"
#include "rvv_defs.hpp"
#include <riscv_vector.h>


/*********************************** Scalar Version ************************************/
void conv2d_transpose_scalar(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {

    // Support both stride=1 and stride=2, no padding
    int out_height = (input_h - 1) * stride_h + kernel_h;
    int out_width = (input_w - 1) * stride_w + kernel_w;

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
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                int out_h_idx = h * stride_h - pad_h + kh;
                                int out_w_idx = w * stride_w - pad_w + kw;

                                if (out_h_idx >= 0 && out_h_idx < out_height &&
                                    out_w_idx >= 0 && out_w_idx < out_width) {

                                    int output_idx = b * out_channels * out_height * out_width +
                                                     oc * out_height * out_width +
                                                     out_h_idx * out_width + out_w_idx;

                                    int kernel_idx = ic * out_channels * kernel_h * kernel_w +
                                                     oc * kernel_h * kernel_w +
                                                     kh * kernel_w + kw;

                                    output[output_idx] += input_val * kernel[kernel_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


/********************************* Vectorized Versions *********************************/

// RVV optimized version (e32m1)
void conv2d_transpose_e32m1(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
    
    int out_height = (input_h - 1) * stride_h - 2 * pad_h + kernel_h;
    int out_width = (input_w - 1) * stride_w - 2 * pad_w + kernel_w;
    
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
                                
                                vfloat32m1_t v_kernel = VECTOR_LOAD<float, M1>(
                                    &kernel[ic * out_channels * kernel_h * kernel_w +
                                           oc * kernel_h * kernel_w +
                                           kh * kernel_w + kw], vl);
                                
                                vfloat32m1_t v_output = VECTOR_LOAD<float, M1>(
                                    &output[b * out_channels * out_height * out_width +
                                           oc * out_height * out_width +
                                           out_h_idx * out_width + out_w_idx + kw], vl);
                                
                                v_output = VECTOR_FMACC_VF<float, M1>(v_output, input_val, v_kernel, vl);
                                
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

// RVV optimized version (e32m2)
void conv2d_transpose_e32m2(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
    
    int out_height = (input_h - 1) * stride_h - 2 * pad_h + kernel_h;
    int out_width = (input_w - 1) * stride_w - 2 * pad_w + kernel_w;
    
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
                            
                            while (kw < kernel_w && out_w_idx + kw < 0) {
                                kw++;
                            }
                            
                            size_t vl;
                            for (; kw < kernel_w; kw += vl) {
                                int remaining = kernel_w - kw;
                                int valid_end = out_width - out_w_idx - kw;
                                int processable = (remaining < valid_end) ? remaining : valid_end;
                                
                                if (processable <= 0) break;
                                
                                vl = SET_VECTOR_LENGTH<float, M2>(processable);
                                
                                vfloat32m2_t v_kernel = VECTOR_LOAD<float, M2>(
                                    &kernel[ic * out_channels * kernel_h * kernel_w +
                                           oc * kernel_h * kernel_w +
                                           kh * kernel_w + kw], vl);
                                
                                vfloat32m2_t v_output = VECTOR_LOAD<float, M2>(
                                    &output[b * out_channels * out_height * out_width +
                                           oc * out_height * out_width +
                                           out_h_idx * out_width + out_w_idx + kw], vl);
                                
                                v_output = VECTOR_FMACC_VF<float, M2>(v_output, input_val, v_kernel, vl);
                                
                                VECTOR_STORE<float, M2>(
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

// RVV optimized version (e32m4)
void conv2d_transpose_e32m4(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
    
    int out_height = (input_h - 1) * stride_h - 2 * pad_h + kernel_h;
    int out_width = (input_w - 1) * stride_w - 2 * pad_w + kernel_w;
    
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
                            
                            while (kw < kernel_w && out_w_idx + kw < 0) {
                                kw++;
                            }
                            
                            size_t vl;
                            for (; kw < kernel_w; kw += vl) {
                                int remaining = kernel_w - kw;
                                int valid_end = out_width - out_w_idx - kw;
                                int processable = (remaining < valid_end) ? remaining : valid_end;
                                
                                if (processable <= 0) break;
                                
                                vl = SET_VECTOR_LENGTH<float, M4>(processable);
                                
                                vfloat32m4_t v_kernel = VECTOR_LOAD<float, M4>(
                                    &kernel[ic * out_channels * kernel_h * kernel_w +
                                           oc * kernel_h * kernel_w +
                                           kh * kernel_w + kw], vl);
                                
                                vfloat32m4_t v_output = VECTOR_LOAD<float, M4>(
                                    &output[b * out_channels * out_height * out_width +
                                           oc * out_height * out_width +
                                           out_h_idx * out_width + out_w_idx + kw], vl);
                                
                                v_output = VECTOR_FMACC_VF<float, M4>(v_output, input_val, v_kernel, vl);
                                
                                VECTOR_STORE<float, M4>(
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

// RVV optimized version (e32m8)
void conv2d_transpose_e32m8(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
    
    int out_height = (input_h - 1) * stride_h - 2 * pad_h + kernel_h;
    int out_width = (input_w - 1) * stride_w - 2 * pad_w + kernel_w;
    
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
                            
                            while (kw < kernel_w && out_w_idx + kw < 0) {
                                kw++;
                            }
                            
                            size_t vl;
                            for (; kw < kernel_w; kw += vl) {
                                int remaining = kernel_w - kw;
                                int valid_end = out_width - out_w_idx - kw;
                                int processable = (remaining < valid_end) ? remaining : valid_end;
                                
                                if (processable <= 0) break;
                                
                                vl = SET_VECTOR_LENGTH<float, M8>(processable);
                                
                                vfloat32m8_t v_kernel = VECTOR_LOAD<float, M8>(
                                    &kernel[ic * out_channels * kernel_h * kernel_w +
                                           oc * kernel_h * kernel_w +
                                           kh * kernel_w + kw], vl);
                                
                                vfloat32m8_t v_output = VECTOR_LOAD<float, M8>(
                                    &output[b * out_channels * out_height * out_width +
                                           oc * out_height * out_width +
                                           out_h_idx * out_width + out_w_idx + kw], vl);
                                
                                v_output = VECTOR_FMACC_VF<float, M8>(v_output, input_val, v_kernel, vl);
                                
                                VECTOR_STORE<float, M8>(
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


/********************************* 3x3 Filter-Specific Vectorized Versions*********************************/

// ========================== LMUL = M1 ==========================

static void conv2d_transpose_3x3_stride1_m1(
    const float* input, const float* kernel, float* output,
    int in_channels, int in_h, int in_w, int out_channels
) {
    const int out_h = in_h + 2;
    const int out_w = in_w + 2;
    
    memset(output, 0, (size_t)out_channels * out_h * out_w * sizeof(float));
    
    for (int oc = 0; oc < out_channels; oc++) {
        float* out_channel = output + (size_t)oc * out_h * out_w;
        
        for (int ic = 0; ic < in_channels; ic++) {
            const float* in_channel = input + (size_t)ic * in_h * in_w;
            const float* filter = kernel + (size_t)ic * out_channels * 9 + (size_t)oc * 9;
            
            float f00 = filter[0], f01 = filter[1], f02 = filter[2];
            float f10 = filter[3], f11 = filter[4], f12 = filter[5];
            float f20 = filter[6], f21 = filter[7], f22 = filter[8];
            
            for (int r = 0; r < in_h; r++) {
                const float* irow = in_channel + (size_t)r * in_w;
                
                for (int c = 0; c < in_w;) {
                    size_t vl = SET_VECTOR_LENGTH<float, M1>(in_w - c);
                    auto vin = VECTOR_LOAD<float, M1>(irow + c, vl);
                    
                    int64_t out_r0 = r, out_r1 = r + 1, out_r2 = r + 2;
                    
                    #define APPLY_FILTER_M1(row, col, fval) \
                        { auto vout = VECTOR_LOAD<float, M1>(out_channel + row * out_w + c + col, vl); \
                          vout = VECTOR_FMACC_VF<float, M1>(vout, fval, vin, vl); \
                          VECTOR_STORE<float, M1>(out_channel + row * out_w + c + col, vout, vl); }
                    
                    APPLY_FILTER_M1(out_r0, 0, f00); APPLY_FILTER_M1(out_r0, 1, f01); APPLY_FILTER_M1(out_r0, 2, f02);
                    APPLY_FILTER_M1(out_r1, 0, f10); APPLY_FILTER_M1(out_r1, 1, f11); APPLY_FILTER_M1(out_r1, 2, f12);
                    APPLY_FILTER_M1(out_r2, 0, f20); APPLY_FILTER_M1(out_r2, 1, f21); APPLY_FILTER_M1(out_r2, 2, f22);
                    
                    #undef APPLY_FILTER_M1
                    c += vl;
                }
            }
        }
    }
}

static void conv2d_transpose_3x3_stride2_m1(
    const float* input, const float* kernel, float* output,
    int in_channels, int in_h, int in_w, int out_channels
) {
    const int out_h = (in_h - 1) * 2 + 3;
    const int out_w = (in_w - 1) * 2 + 3;
    
    memset(output, 0, (size_t)out_channels * out_h * out_w * sizeof(float));
    
    for (int oc = 0; oc < out_channels; oc++) {
        float* out_channel = output + (size_t)oc * out_h * out_w;
        
        for (int ic = 0; ic < in_channels; ic++) {
            const float* in_channel = input + (size_t)ic * in_h * in_w;
            const float* filter = kernel + (size_t)ic * out_channels * 9 + (size_t)oc * 9;
            
            float f00 = filter[0], f01 = filter[1], f02 = filter[2];
            float f10 = filter[3], f11 = filter[4], f12 = filter[5];
            float f20 = filter[6], f21 = filter[7], f22 = filter[8];
            
            for (int r = 0; r < in_h; r++) {
                const float* irow = in_channel + (size_t)r * in_w;
                
                for (int c = 0; c < in_w;) {
                    size_t vl = SET_VECTOR_LENGTH<float, M1>(in_w - c);
                    auto vin = VECTOR_LOAD<float, M1>(irow + c, vl);
                    
                    const ptrdiff_t bstride = 2 * sizeof(float);
                    int64_t out_r0 = (int64_t)r * 2, out_r1 = out_r0 + 1, out_r2 = out_r0 + 2;
                    int64_t out_c_start = (int64_t)c * 2;
                    
                    #define APPLY_FILTER_STRIDED_M1(row, col, fval) \
                        { float* ptr = out_channel + row * out_w + out_c_start + col; \
                          auto vout = VECTOR_STRIDED_LOAD<float, M1>(ptr, bstride, vl); \
                          vout = VECTOR_FMACC_VF<float, M1>(vout, fval, vin, vl); \
                          VECTOR_STRIDED_STORE<float, M1>(ptr, bstride, vout, vl); }
                    
                    APPLY_FILTER_STRIDED_M1(out_r0, 0, f00); APPLY_FILTER_STRIDED_M1(out_r0, 1, f01); APPLY_FILTER_STRIDED_M1(out_r0, 2, f02);
                    APPLY_FILTER_STRIDED_M1(out_r1, 0, f10); APPLY_FILTER_STRIDED_M1(out_r1, 1, f11); APPLY_FILTER_STRIDED_M1(out_r1, 2, f12);
                    APPLY_FILTER_STRIDED_M1(out_r2, 0, f20); APPLY_FILTER_STRIDED_M1(out_r2, 1, f21); APPLY_FILTER_STRIDED_M1(out_r2, 2, f22);
                    
                    #undef APPLY_FILTER_STRIDED_M1
                    c += vl;
                }
            }
        }
    }
}

void conv2d_transpose_3x3_rvv_m1(
    const float* input, const float* kernel, float* output,
    int in_channels, int in_h, int in_w, int out_channels, int stride_h, int stride_w
) {
    if (stride_h == 1 && stride_w == 1) {
        conv2d_transpose_3x3_stride1_m1(input, kernel, output, in_channels, in_h, in_w, out_channels);
    } else {
        conv2d_transpose_3x3_stride2_m1(input, kernel, output, in_channels, in_h, in_w, out_channels);
    }
}

// ========================== LMUL = M2 ==========================

static void conv2d_transpose_3x3_stride1_m2(
    const float* input, const float* kernel, float* output,
    int in_channels, int in_h, int in_w, int out_channels
) {
    const int out_h = in_h + 2;
    const int out_w = in_w + 2;
    
    memset(output, 0, (size_t)out_channels * out_h * out_w * sizeof(float));
    
    for (int oc = 0; oc < out_channels; oc++) {
        float* out_channel = output + (size_t)oc * out_h * out_w;
        
        for (int ic = 0; ic < in_channels; ic++) {
            const float* in_channel = input + (size_t)ic * in_h * in_w;
            const float* filter = kernel + (size_t)ic * out_channels * 9 + (size_t)oc * 9;
            
            float f00 = filter[0], f01 = filter[1], f02 = filter[2];
            float f10 = filter[3], f11 = filter[4], f12 = filter[5];
            float f20 = filter[6], f21 = filter[7], f22 = filter[8];
            
            for (int r = 0; r < in_h; r++) {
                const float* irow = in_channel + (size_t)r * in_w;
                
                for (int c = 0; c < in_w;) {
                    size_t vl = SET_VECTOR_LENGTH<float, M2>(in_w - c);
                    auto vin = VECTOR_LOAD<float, M2>(irow + c, vl);
                    
                    int64_t out_r0 = r, out_r1 = r + 1, out_r2 = r + 2;
                    
                    #define APPLY_FILTER_M2(row, col, fval) \
                        { auto vout = VECTOR_LOAD<float, M2>(out_channel + row * out_w + c + col, vl); \
                          vout = VECTOR_FMACC_VF<float, M2>(vout, fval, vin, vl); \
                          VECTOR_STORE<float, M2>(out_channel + row * out_w + c + col, vout, vl); }
                    
                    APPLY_FILTER_M2(out_r0, 0, f00); APPLY_FILTER_M2(out_r0, 1, f01); APPLY_FILTER_M2(out_r0, 2, f02);
                    APPLY_FILTER_M2(out_r1, 0, f10); APPLY_FILTER_M2(out_r1, 1, f11); APPLY_FILTER_M2(out_r1, 2, f12);
                    APPLY_FILTER_M2(out_r2, 0, f20); APPLY_FILTER_M2(out_r2, 1, f21); APPLY_FILTER_M2(out_r2, 2, f22);
                    
                    #undef APPLY_FILTER_M2
                    c += vl;
                }
            }
        }
    }
}

static void conv2d_transpose_3x3_stride2_m2(
    const float* input, const float* kernel, float* output,
    int in_channels, int in_h, int in_w, int out_channels
) {
    const int out_h = (in_h - 1) * 2 + 3;
    const int out_w = (in_w - 1) * 2 + 3;
    
    memset(output, 0, (size_t)out_channels * out_h * out_w * sizeof(float));
    
    for (int oc = 0; oc < out_channels; oc++) {
        float* out_channel = output + (size_t)oc * out_h * out_w;
        
        for (int ic = 0; ic < in_channels; ic++) {
            const float* in_channel = input + (size_t)ic * in_h * in_w;
            const float* filter = kernel + (size_t)ic * out_channels * 9 + (size_t)oc * 9;
            
            float f00 = filter[0], f01 = filter[1], f02 = filter[2];
            float f10 = filter[3], f11 = filter[4], f12 = filter[5];
            float f20 = filter[6], f21 = filter[7], f22 = filter[8];
            
            for (int r = 0; r < in_h; r++) {
                const float* irow = in_channel + (size_t)r * in_w;
                
                for (int c = 0; c < in_w;) {
                    size_t vl = SET_VECTOR_LENGTH<float, M2>(in_w - c);
                    auto vin = VECTOR_LOAD<float, M2>(irow + c, vl);
                    
                    const ptrdiff_t bstride = 2 * sizeof(float);
                    int64_t out_r0 = (int64_t)r * 2, out_r1 = out_r0 + 1, out_r2 = out_r0 + 2;
                    int64_t out_c_start = (int64_t)c * 2;
                    
                    #define APPLY_FILTER_STRIDED_M2(row, col, fval) \
                        { float* ptr = out_channel + row * out_w + out_c_start + col; \
                          auto vout = VECTOR_STRIDED_LOAD<float, M2>(ptr, bstride, vl); \
                          vout = VECTOR_FMACC_VF<float, M2>(vout, fval, vin, vl); \
                          VECTOR_STRIDED_STORE<float, M2>(ptr, bstride, vout, vl); }
                    
                    APPLY_FILTER_STRIDED_M2(out_r0, 0, f00); APPLY_FILTER_STRIDED_M2(out_r0, 1, f01); APPLY_FILTER_STRIDED_M2(out_r0, 2, f02);
                    APPLY_FILTER_STRIDED_M2(out_r1, 0, f10); APPLY_FILTER_STRIDED_M2(out_r1, 1, f11); APPLY_FILTER_STRIDED_M2(out_r1, 2, f12);
                    APPLY_FILTER_STRIDED_M2(out_r2, 0, f20); APPLY_FILTER_STRIDED_M2(out_r2, 1, f21); APPLY_FILTER_STRIDED_M2(out_r2, 2, f22);
                    
                    #undef APPLY_FILTER_STRIDED_M2
                    c += vl;
                }
            }
        }
    }
}

void conv2d_transpose_3x3_rvv_m2(
    const float* input, const float* kernel, float* output,
    int in_channels, int in_h, int in_w, int out_channels, int stride_h, int stride_w
) {
    if (stride_h == 1 && stride_w == 1) {
        conv2d_transpose_3x3_stride1_m2(input, kernel, output, in_channels, in_h, in_w, out_channels);
    } else {
        conv2d_transpose_3x3_stride2_m2(input, kernel, output, in_channels, in_h, in_w, out_channels);
    }
}

// ========================== LMUL = M4 ==========================

static void conv2d_transpose_3x3_stride1_m4(
    const float* input, const float* kernel, float* output,
    int in_channels, int in_h, int in_w, int out_channels
) {
    const int out_h = in_h + 2;
    const int out_w = in_w + 2;
    
    memset(output, 0, (size_t)out_channels * out_h * out_w * sizeof(float));
    
    for (int oc = 0; oc < out_channels; oc++) {
        float* out_channel = output + (size_t)oc * out_h * out_w;
        
        for (int ic = 0; ic < in_channels; ic++) {
            const float* in_channel = input + (size_t)ic * in_h * in_w;
            const float* filter = kernel + (size_t)ic * out_channels * 9 + (size_t)oc * 9;
            
            float f00 = filter[0], f01 = filter[1], f02 = filter[2];
            float f10 = filter[3], f11 = filter[4], f12 = filter[5];
            float f20 = filter[6], f21 = filter[7], f22 = filter[8];
            
            for (int r = 0; r < in_h; r++) {
                const float* irow = in_channel + (size_t)r * in_w;
                
                for (int c = 0; c < in_w;) {
                    size_t vl = SET_VECTOR_LENGTH<float, M4>(in_w - c);
                    auto vin = VECTOR_LOAD<float, M4>(irow + c, vl);
                    
                    int64_t out_r0 = r, out_r1 = r + 1, out_r2 = r + 2;
                    
                    #define APPLY_FILTER_M4(row, col, fval) \
                        { auto vout = VECTOR_LOAD<float, M4>(out_channel + row * out_w + c + col, vl); \
                          vout = VECTOR_FMACC_VF<float, M4>(vout, fval, vin, vl); \
                          VECTOR_STORE<float, M4>(out_channel + row * out_w + c + col, vout, vl); }
                    
                    APPLY_FILTER_M4(out_r0, 0, f00); APPLY_FILTER_M4(out_r0, 1, f01); APPLY_FILTER_M4(out_r0, 2, f02);
                    APPLY_FILTER_M4(out_r1, 0, f10); APPLY_FILTER_M4(out_r1, 1, f11); APPLY_FILTER_M4(out_r1, 2, f12);
                    APPLY_FILTER_M4(out_r2, 0, f20); APPLY_FILTER_M4(out_r2, 1, f21); APPLY_FILTER_M4(out_r2, 2, f22);
                    
                    #undef APPLY_FILTER_M4
                    c += vl;
                }
            }
        }
    }
}

static void conv2d_transpose_3x3_stride2_m4(
    const float* input, const float* kernel, float* output,
    int in_channels, int in_h, int in_w, int out_channels
) {
    const int out_h = (in_h - 1) * 2 + 3;
    const int out_w = (in_w - 1) * 2 + 3;
    
    memset(output, 0, (size_t)out_channels * out_h * out_w * sizeof(float));
    
    for (int oc = 0; oc < out_channels; oc++) {
        float* out_channel = output + (size_t)oc * out_h * out_w;
        
        for (int ic = 0; ic < in_channels; ic++) {
            const float* in_channel = input + (size_t)ic * in_h * in_w;
            const float* filter = kernel + (size_t)ic * out_channels * 9 + (size_t)oc * 9;
            
            float f00 = filter[0], f01 = filter[1], f02 = filter[2];
            float f10 = filter[3], f11 = filter[4], f12 = filter[5];
            float f20 = filter[6], f21 = filter[7], f22 = filter[8];
            
            for (int r = 0; r < in_h; r++) {
                const float* irow = in_channel + (size_t)r * in_w;
                
                for (int c = 0; c < in_w;) {
                    size_t vl = SET_VECTOR_LENGTH<float, M4>(in_w - c);
                    auto vin = VECTOR_LOAD<float, M4>(irow + c, vl);
                    
                    const ptrdiff_t bstride = 2 * sizeof(float);
                    int64_t out_r0 = (int64_t)r * 2, out_r1 = out_r0 + 1, out_r2 = out_r0 + 2;
                    int64_t out_c_start = (int64_t)c * 2;
                    
                    #define APPLY_FILTER_STRIDED_M4(row, col, fval) \
                        { float* ptr = out_channel + row * out_w + out_c_start + col; \
                          auto vout = VECTOR_STRIDED_LOAD<float, M4>(ptr, bstride, vl); \
                          vout = VECTOR_FMACC_VF<float, M4>(vout, fval, vin, vl); \
                          VECTOR_STRIDED_STORE<float, M4>(ptr, bstride, vout, vl); }
                    
                    APPLY_FILTER_STRIDED_M4(out_r0, 0, f00); APPLY_FILTER_STRIDED_M4(out_r0, 1, f01); APPLY_FILTER_STRIDED_M4(out_r0, 2, f02);
                    APPLY_FILTER_STRIDED_M4(out_r1, 0, f10); APPLY_FILTER_STRIDED_M4(out_r1, 1, f11); APPLY_FILTER_STRIDED_M4(out_r1, 2, f12);
                    APPLY_FILTER_STRIDED_M4(out_r2, 0, f20); APPLY_FILTER_STRIDED_M4(out_r2, 1, f21); APPLY_FILTER_STRIDED_M4(out_r2, 2, f22);
                    
                    #undef APPLY_FILTER_STRIDED_M4
                    c += vl;
                }
            }
        }
    }
}

void conv2d_transpose_3x3_rvv_m4(
    const float* input, const float* kernel, float* output,
    int in_channels, int in_h, int in_w, int out_channels, int stride_h, int stride_w
) {
    if (stride_h == 1 && stride_w == 1) {
        conv2d_transpose_3x3_stride1_m4(input, kernel, output, in_channels, in_h, in_w, out_channels);
    } else {
        conv2d_transpose_3x3_stride2_m4(input, kernel, output, in_channels, in_h, in_w, out_channels);
    }
}

// ========================== LMUL = M8 ==========================

static void conv2d_transpose_3x3_stride1_m8(
    const float* input, const float* kernel, float* output,
    int in_channels, int in_h, int in_w, int out_channels
) {
    const int out_h = in_h + 2;
    const int out_w = in_w + 2;
    
    memset(output, 0, (size_t)out_channels * out_h * out_w * sizeof(float));
    
    for (int oc = 0; oc < out_channels; oc++) {
        float* out_channel = output + (size_t)oc * out_h * out_w;
        
        for (int ic = 0; ic < in_channels; ic++) {
            const float* in_channel = input + (size_t)ic * in_h * in_w;
            const float* filter = kernel + (size_t)ic * out_channels * 9 + (size_t)oc * 9;
            
            float f00 = filter[0], f01 = filter[1], f02 = filter[2];
            float f10 = filter[3], f11 = filter[4], f12 = filter[5];
            float f20 = filter[6], f21 = filter[7], f22 = filter[8];
            
            for (int r = 0; r < in_h; r++) {
                const float* irow = in_channel + (size_t)r * in_w;
                
                for (int c = 0; c < in_w;) {
                    size_t vl = SET_VECTOR_LENGTH<float, M8>(in_w - c);
                    auto vin = VECTOR_LOAD<float, M8>(irow + c, vl);
                    
                    int64_t out_r0 = r, out_r1 = r + 1, out_r2 = r + 2;
                    
                    #define APPLY_FILTER_M8(row, col, fval) \
                        { auto vout = VECTOR_LOAD<float, M8>(out_channel + row * out_w + c + col, vl); \
                          vout = VECTOR_FMACC_VF<float, M8>(vout, fval, vin, vl); \
                          VECTOR_STORE<float, M8>(out_channel + row * out_w + c + col, vout, vl); }
                    
                    APPLY_FILTER_M8(out_r0, 0, f00); APPLY_FILTER_M8(out_r0, 1, f01); APPLY_FILTER_M8(out_r0, 2, f02);
                    APPLY_FILTER_M8(out_r1, 0, f10); APPLY_FILTER_M8(out_r1, 1, f11); APPLY_FILTER_M8(out_r1, 2, f12);
                    APPLY_FILTER_M8(out_r2, 0, f20); APPLY_FILTER_M8(out_r2, 1, f21); APPLY_FILTER_M8(out_r2, 2, f22);
                    
                    #undef APPLY_FILTER_M8
                    c += vl;
                }
            }
        }
    }
}

static void conv2d_transpose_3x3_stride2_m8(
    const float* input, const float* kernel, float* output,
    int in_channels, int in_h, int in_w, int out_channels
) {
    const int out_h = (in_h - 1) * 2 + 3;
    const int out_w = (in_w - 1) * 2 + 3;
    
    memset(output, 0, (size_t)out_channels * out_h * out_w * sizeof(float));
    
    for (int oc = 0; oc < out_channels; oc++) {
        float* out_channel = output + (size_t)oc * out_h * out_w;
        
        for (int ic = 0; ic < in_channels; ic++) {
            const float* in_channel = input + (size_t)ic * in_h * in_w;
            const float* filter = kernel + (size_t)ic * out_channels * 9 + (size_t)oc * 9;
            
            float f00 = filter[0], f01 = filter[1], f02 = filter[2];
            float f10 = filter[3], f11 = filter[4], f12 = filter[5];
            float f20 = filter[6], f21 = filter[7], f22 = filter[8];
            
            for (int r = 0; r < in_h; r++) {
                const float* irow = in_channel + (size_t)r * in_w;
                
                for (int c = 0; c < in_w;) {
                    size_t vl = SET_VECTOR_LENGTH<float, M8>(in_w - c);
                    auto vin = VECTOR_LOAD<float, M8>(irow + c, vl);
                    
                    const ptrdiff_t bstride = 2 * sizeof(float);
                    int64_t out_r0 = (int64_t)r * 2, out_r1 = out_r0 + 1, out_r2 = out_r0 + 2;
                    int64_t out_c_start = (int64_t)c * 2;
                    
                    #define APPLY_FILTER_STRIDED_M8(row, col, fval) \
                        { float* ptr = out_channel + row * out_w + out_c_start + col; \
                          auto vout = VECTOR_STRIDED_LOAD<float, M8>(ptr, bstride, vl); \
                          vout = VECTOR_FMACC_VF<float, M8>(vout, fval, vin, vl); \
                          VECTOR_STRIDED_STORE<float, M8>(ptr, bstride, vout, vl); }
                    
                    APPLY_FILTER_STRIDED_M8(out_r0, 0, f00); APPLY_FILTER_STRIDED_M8(out_r0, 1, f01); APPLY_FILTER_STRIDED_M8(out_r0, 2, f02);
                    APPLY_FILTER_STRIDED_M8(out_r1, 0, f10); APPLY_FILTER_STRIDED_M8(out_r1, 1, f11); APPLY_FILTER_STRIDED_M8(out_r1, 2, f12);
                    APPLY_FILTER_STRIDED_M8(out_r2, 0, f20); APPLY_FILTER_STRIDED_M8(out_r2, 1, f21); APPLY_FILTER_STRIDED_M8(out_r2, 2, f22);
                    
                    #undef APPLY_FILTER_STRIDED_M8
                    c += vl;
                }
            }
        }
    }
}

void conv2d_transpose_3x3_rvv_m8(
    const float* input, const float* kernel, float* output,
    int in_channels, int in_h, int in_w, int out_channels, int stride_h, int stride_w
) {
    if (stride_h == 1 && stride_w == 1) {
        conv2d_transpose_3x3_stride1_m8(input, kernel, output, in_channels, in_h, in_w, out_channels);
    } else {
        conv2d_transpose_3x3_stride2_m8(input, kernel, output, in_channels, in_h, in_w, out_channels);
    }
}