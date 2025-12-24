#include <cstring>
#include <cstdint>
#include "defs.h"
#include "../../lib/rvv_defs.hpp"

// ============================================================================
// 3x3 Specialized Transposed Convolution - All LMUL variants
// Kernel layout: [in_channels, out_channels, 3, 3]
// ============================================================================

// ========================== LMUL = M1 ==========================

static void transposed_conv2d_3x3_stride1_m1(
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

static void transposed_conv2d_3x3_stride2_m1(
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

void transposed_conv2d_3x3_rvv_m1(
    const float* input, const float* kernel, float* output,
    int in_channels, int in_h, int in_w, int out_channels, int stride_h, int stride_w
) {
    if (stride_h == 1 && stride_w == 1) {
        transposed_conv2d_3x3_stride1_m1(input, kernel, output, in_channels, in_h, in_w, out_channels);
    } else {
        transposed_conv2d_3x3_stride2_m1(input, kernel, output, in_channels, in_h, in_w, out_channels);
    }
}

// ========================== LMUL = M2 ==========================

static void transposed_conv2d_3x3_stride1_m2(
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

static void transposed_conv2d_3x3_stride2_m2(
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

void transposed_conv2d_3x3_rvv_m2(
    const float* input, const float* kernel, float* output,
    int in_channels, int in_h, int in_w, int out_channels, int stride_h, int stride_w
) {
    if (stride_h == 1 && stride_w == 1) {
        transposed_conv2d_3x3_stride1_m2(input, kernel, output, in_channels, in_h, in_w, out_channels);
    } else {
        transposed_conv2d_3x3_stride2_m2(input, kernel, output, in_channels, in_h, in_w, out_channels);
    }
}

// ========================== LMUL = M4 ==========================

static void transposed_conv2d_3x3_stride1_m4(
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

static void transposed_conv2d_3x3_stride2_m4(
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

void transposed_conv2d_3x3_rvv_m4(
    const float* input, const float* kernel, float* output,
    int in_channels, int in_h, int in_w, int out_channels, int stride_h, int stride_w
) {
    if (stride_h == 1 && stride_w == 1) {
        transposed_conv2d_3x3_stride1_m4(input, kernel, output, in_channels, in_h, in_w, out_channels);
    } else {
        transposed_conv2d_3x3_stride2_m4(input, kernel, output, in_channels, in_h, in_w, out_channels);
    }
}

// ========================== LMUL = M8 ==========================

static void transposed_conv2d_3x3_stride1_m8(
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

static void transposed_conv2d_3x3_stride2_m8(
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

void transposed_conv2d_3x3_rvv_m8(
    const float* input, const float* kernel, float* output,
    int in_channels, int in_h, int in_w, int out_channels, int stride_h, int stride_w
) {
    if (stride_h == 1 && stride_w == 1) {
        transposed_conv2d_3x3_stride1_m8(input, kernel, output, in_channels, in_h, in_w, out_channels);
    } else {
        transposed_conv2d_3x3_stride2_m8(input, kernel, output, in_channels, in_h, in_w, out_channels);
    }
}
