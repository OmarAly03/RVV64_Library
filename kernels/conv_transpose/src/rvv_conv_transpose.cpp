#include <cstring>
#include <cstdint>
#include "defs.h"
#include "../../lib/rvv_defs.hpp"

// Transposed convolution 3x3, stride=1, multi-channel
// Kernel layout: [in_channels, out_channels, 3, 3]
static void transposed_conv2d_3x3_stride1(
    const float* input,
    const float* kernel,
    float* output,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels
) {
    // Output dimensions for stride=1, 3x3 kernel
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
                    
                    int64_t out_r0 = r;
                    int64_t out_r1 = r + 1;
                    int64_t out_r2 = r + 2;
                    
                    auto vout0 = VECTOR_LOAD<float, M2>(out_channel + out_r0 * out_w + c, vl);
                    vout0 = VECTOR_FMACC_VF<float, M2>(vout0, f00, vin, vl);
                    VECTOR_STORE<float, M2>(out_channel + out_r0 * out_w + c, vout0, vl);
                    
                    auto vout1 = VECTOR_LOAD<float, M2>(out_channel + out_r0 * out_w + c + 1, vl);
                    vout1 = VECTOR_FMACC_VF<float, M2>(vout1, f01, vin, vl);
                    VECTOR_STORE<float, M2>(out_channel + out_r0 * out_w + c + 1, vout1, vl);
                    
                    auto vout2 = VECTOR_LOAD<float, M2>(out_channel + out_r0 * out_w + c + 2, vl);
                    vout2 = VECTOR_FMACC_VF<float, M2>(vout2, f02, vin, vl);
                    VECTOR_STORE<float, M2>(out_channel + out_r0 * out_w + c + 2, vout2, vl);
                    
                    // Apply filter row 1
                    auto vout3 = VECTOR_LOAD<float, M2>(out_channel + out_r1 * out_w + c, vl);
                    vout3 = VECTOR_FMACC_VF<float, M2>(vout3, f10, vin, vl);
                    VECTOR_STORE<float, M2>(out_channel + out_r1 * out_w + c, vout3, vl);
                    
                    auto vout4 = VECTOR_LOAD<float, M2>(out_channel + out_r1 * out_w + c + 1, vl);
                    vout4 = VECTOR_FMACC_VF<float, M2>(vout4, f11, vin, vl);
                    VECTOR_STORE<float, M2>(out_channel + out_r1 * out_w + c + 1, vout4, vl);
                    
                    auto vout5 = VECTOR_LOAD<float, M2>(out_channel + out_r1 * out_w + c + 2, vl);
                    vout5 = VECTOR_FMACC_VF<float, M2>(vout5, f12, vin, vl);
                    VECTOR_STORE<float, M2>(out_channel + out_r1 * out_w + c + 2, vout5, vl);
                    
                    auto vout6 = VECTOR_LOAD<float, M2>(out_channel + out_r2 * out_w + c, vl);
                    vout6 = VECTOR_FMACC_VF<float, M2>(vout6, f20, vin, vl);
                    VECTOR_STORE<float, M2>(out_channel + out_r2 * out_w + c, vout6, vl);
                    
                    auto vout7 = VECTOR_LOAD<float, M2>(out_channel + out_r2 * out_w + c + 1, vl);
                    vout7 = VECTOR_FMACC_VF<float, M2>(vout7, f21, vin, vl);
                    VECTOR_STORE<float, M2>(out_channel + out_r2 * out_w + c + 1, vout7, vl);
                    
                    auto vout8 = VECTOR_LOAD<float, M2>(out_channel + out_r2 * out_w + c + 2, vl);
                    vout8 = VECTOR_FMACC_VF<float, M2>(vout8, f22, vin, vl);
                    VECTOR_STORE<float, M2>(out_channel + out_r2 * out_w + c + 2, vout8, vl);
                    
                    c += vl;
                }
            }
        }
    }
}

// Transposed convolution 3x3, stride=2, multi-channel
// Kernel layout: [in_channels, out_channels, 3, 3]
static void transposed_conv2d_3x3_stride2(
    const float* input,
    const float* kernel,
    float* output,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels
) {
    // Output dimensions for stride=2, 3x3 kernel
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
                    int64_t out_r0 = (int64_t)r * 2;
                    int64_t out_r1 = out_r0 + 1;
                    int64_t out_r2 = out_r0 + 2;
                    int64_t out_c_start = (int64_t)c * 2;
                    
                    float* out_ptr0 = out_channel + out_r0 * out_w + out_c_start;
                    auto vout0 = VECTOR_STRIDED_LOAD<float, M2>(out_ptr0, bstride, vl);
                    vout0 = VECTOR_FMACC_VF<float, M2>(vout0, f00, vin, vl);
                    VECTOR_STRIDED_STORE<float, M2>(out_ptr0, bstride, vout0, vl);
                    
                    float* out_ptr1 = out_channel + out_r0 * out_w + out_c_start + 1;
                    auto vout1 = VECTOR_STRIDED_LOAD<float, M2>(out_ptr1, bstride, vl);
                    vout1 = VECTOR_FMACC_VF<float, M2>(vout1, f01, vin, vl);
                    VECTOR_STRIDED_STORE<float, M2>(out_ptr1, bstride, vout1, vl);
                    
                    float* out_ptr2 = out_channel + out_r0 * out_w + out_c_start + 2;
                    auto vout2 = VECTOR_STRIDED_LOAD<float, M2>(out_ptr2, bstride, vl);
                    vout2 = VECTOR_FMACC_VF<float, M2>(vout2, f02, vin, vl);
                    VECTOR_STRIDED_STORE<float, M2>(out_ptr2, bstride, vout2, vl);
                    
                    // Apply filter row 1
                    float* out_ptr3 = out_channel + out_r1 * out_w + out_c_start;
                    auto vout3 = VECTOR_STRIDED_LOAD<float, M2>(out_ptr3, bstride, vl);
                    vout3 = VECTOR_FMACC_VF<float, M2>(vout3, f10, vin, vl);
                    VECTOR_STRIDED_STORE<float, M2>(out_ptr3, bstride, vout3, vl);
                    
                    float* out_ptr4 = out_channel + out_r1 * out_w + out_c_start + 1;
                    auto vout4 = VECTOR_STRIDED_LOAD<float, M2>(out_ptr4, bstride, vl);
                    vout4 = VECTOR_FMACC_VF<float, M2>(vout4, f11, vin, vl);
                    VECTOR_STRIDED_STORE<float, M2>(out_ptr4, bstride, vout4, vl);
                    
                    float* out_ptr5 = out_channel + out_r1 * out_w + out_c_start + 2;
                    auto vout5 = VECTOR_STRIDED_LOAD<float, M2>(out_ptr5, bstride, vl);
                    vout5 = VECTOR_FMACC_VF<float, M2>(vout5, f12, vin, vl);
                    VECTOR_STRIDED_STORE<float, M2>(out_ptr5, bstride, vout5, vl);
                    
                    float* out_ptr6 = out_channel + out_r2 * out_w + out_c_start;
                    auto vout6 = VECTOR_STRIDED_LOAD<float, M2>(out_ptr6, bstride, vl);
                    vout6 = VECTOR_FMACC_VF<float, M2>(vout6, f20, vin, vl);
                    VECTOR_STRIDED_STORE<float, M2>(out_ptr6, bstride, vout6, vl);
                    
                    float* out_ptr7 = out_channel + out_r2 * out_w + out_c_start + 1;
                    auto vout7 = VECTOR_STRIDED_LOAD<float, M2>(out_ptr7, bstride, vl);
                    vout7 = VECTOR_FMACC_VF<float, M2>(vout7, f21, vin, vl);
                    VECTOR_STRIDED_STORE<float, M2>(out_ptr7, bstride, vout7, vl);
                    
                    float* out_ptr8 = out_channel + out_r2 * out_w + out_c_start + 2;
                    auto vout8 = VECTOR_STRIDED_LOAD<float, M2>(out_ptr8, bstride, vl);
                    vout8 = VECTOR_FMACC_VF<float, M2>(vout8, f22, vin, vl);
                    VECTOR_STRIDED_STORE<float, M2>(out_ptr8, bstride, vout8, vl);
                    
                    c += vl;
                }
            }
        }
    }
}

// Dispatcher: selects stride=1 or stride=2 implementation
void transposed_conv2d_3x3_s2_direct_m8(
    const float* input,
    const float* kernel,
    float* output,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int stride_h,
    int stride_w
) {
    if (stride_h == 1 && stride_w == 1) {
        transposed_conv2d_3x3_stride1(input, kernel, output, in_channels, in_h, in_w, out_channels);
    } else {
        transposed_conv2d_3x3_stride2(input, kernel, output, in_channels, in_h, in_w, out_channels);
    }
}
