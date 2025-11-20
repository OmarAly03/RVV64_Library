#include <riscv_vector.h>
#include <stddef.h>

extern "C" {

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
            
            // Vectorize the spatial plane (H*W) using M8
            while (cnt > 0) {
                size_t vl = __riscv_vsetvl_e32m8(cnt);
                vfloat32m8_t v_in = __riscv_vle32_v_f32m8(in_ptr, vl);
                vfloat32m8_t v_out = __riscv_vfadd_vf_f32m8(v_in, b_val, vl);
                __riscv_vse32_v_f32m8(out_ptr, v_out, vl);
                
                in_ptr += vl;
                out_ptr += vl;
                cnt -= vl;
            }
        }
    }
}

void bias_add_scalar(const float* input, const float* bias, float* output,
                       size_t batch_size, size_t channels, size_t height, size_t width) {
    size_t channel_size = height * width;
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            float b_val = bias[c];
            size_t offset = (b * channels + c) * channel_size;
            for (size_t i = 0; i < channel_size; ++i) {
                output[offset + i] = input[offset + i] + b_val;
            }
        }
    }
}

}