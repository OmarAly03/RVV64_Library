#include <riscv_vector.h>
#include "defs.h"
#include "../lib/rvv_defs.hpp"

/**
 * RVV-optimized 2D Transposed Convolution with LMUL=8
 * Uses gather-based vectorization for non-contiguous memory access
 * Kernel layout: [in_channels, out_channels, kernel_h, kernel_w]
 */
void conv_transpose_2d_e32m8(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
    
    const int output_h = (input_h - 1) * stride_h - 2 * pad_h + kernel_h;
    const int output_w = (input_w - 1) * stride_w - 2 * pad_w + kernel_w;
    const int input_spatial = input_h * input_w;
    const int output_spatial = output_h * output_w;
    
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            float* output_channel = output + b * out_channels * output_spatial + oc * output_spatial;
            
            for (int oh = 0; oh < output_h; ++oh) {
                // Vectorize along output width using strip-mining
                for (int ow = 0; ow < output_w; ) {
                    size_t vl = SET_VECTOR_LENGTH<float, M8>(output_w - ow);  // __riscv_vsetvl_e32m8
                    
                    // Create vector of output column indices [ow, ow+1, ..., ow+vl-1]
                    vuint32m8_t ow_vec = VECTOR_VID<uint32_t, M8>(vl);        // __riscv_vid_v_u32m8
                    ow_vec = VECTOR_ADD<uint32_t, M8>(ow_vec, ow, vl);        // __riscv_vadd_vx_u32m8
                    
                    auto main_acc = VECTOR_MOVE<float, M8>(0.0f, vl);         // __riscv_vfmv_v_f_f32m8
                    
                    for (int ic = 0; ic < in_channels; ++ic) {
                        const float* input_channel = input + b * in_channels * input_spatial + ic * input_spatial;
                        const float* kernel_ptr = kernel + ic * out_channels * kernel_h * kernel_w + oc * kernel_h * kernel_w;
                        
                        auto ic_acc = VECTOR_MOVE<float, M8>(0.0f, vl);       // __riscv_vfmv_v_f_f32m8
                        
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            // Compute input row: iw = (ow + pad - kw) / stride (if aligned)
                            int ih_base = oh + pad_h - kh;
                            if (ih_base < 0 || ih_base % stride_h != 0) continue;
                            int ih = ih_base / stride_h;
                            if (ih >= input_h) continue;
                            
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                // Compute input column for each output position: iw = (ow + pad_w - kw) / stride
                                vint32m8_t ow_vec_signed = VECTOR_REINTERPRET<uint32_t, int32_t, M8>(ow_vec);  // __riscv_vreinterpret_v_u32m8_i32m8
                                vint32m8_t iw_base = VECTOR_ADD<int32_t, M8>(ow_vec_signed, pad_w - kw, vl);   // __riscv_vadd_vx_i32m8
                                
                                vbool4_t can_compute = VECTOR_GE_SCALAR<int32_t, M8>(iw_base, 0, vl);          // __riscv_vmsge_vx_i32m8_b4
                                vuint32m8_t iw_candidate = VECTOR_REINTERPRET<int32_t, uint32_t, M8>(iw_base); // __riscv_vreinterpret_v_i32m8_u32m8
                                
                                // Check stride alignment (for stride=2, check if LSB is 0)
                                vuint32m8_t remainder = VECTOR_AND_VX<uint32_t, M8>(iw_candidate, stride_w - 1, vl);  // __riscv_vand_vx_u32m8
                                vbool4_t aligned = VECTOR_EQ_SCALAR<uint32_t, M8>(remainder, 0, vl);                  // __riscv_vmseq_vx_u32m8_b4
                                vuint32m8_t iw_vec = VECTOR_SRL_VX<uint32_t, M8>(iw_candidate, 1, vl);                // __riscv_vsrl_vx_u32m8
                                vbool4_t in_bounds = VECTOR_LTU_SCALAR<uint32_t, M8>(iw_vec, input_w, vl);            // __riscv_vmsltu_vx_u32m8_b4
                                
                                // Combine validity checks
                                vbool4_t valid = MASK_AND_E32<float, M8>(can_compute, aligned, vl);      // __riscv_vmand_mm_b4
                                valid = MASK_AND_E32<float, M8>(valid, in_bounds, vl);                   // __riscv_vmand_mm_b4
                                
                                // Compute gather indices and byte offsets
                                vuint32m8_t gather_idx = VECTOR_ADD<uint32_t, M8>(iw_vec, ih * input_w, vl);  // __riscv_vadd_vx_u32m8
                                vuint32m8_t byte_offset = VECTOR_SLL_VX<uint32_t, M8>(gather_idx, 2, vl);     // __riscv_vsll_vx_u32m8
                                
                                // Gather input values (masked)
                                auto gathered = VECTOR_INDEXED_LOAD_MU<float, M8>(valid, VECTOR_MOVE<float, M8>(0.0f, vl),  // __riscv_vluxei32_v_f32m8_mu
                                                                                   input_channel, byte_offset, vl);
                                
                                float k_val = kernel_ptr[kh * kernel_w + kw];
                                ic_acc = VECTOR_FMADD_VF<float, M8>(gathered, k_val, ic_acc, vl);  // __riscv_vfmadd_vf_f32m8
                            }
                        }
                        
                        main_acc = VECTOR_ADD<float, M8>(main_acc, ic_acc, vl);  // __riscv_vfadd_vv_f32m8
                    }
                    
                    VECTOR_STORE<float, M8>(output_channel + oh * output_w + ow, main_acc, vl);  // __riscv_vse32_v_f32m8
                    ow += vl;
                }
            }
        }
    }
}
