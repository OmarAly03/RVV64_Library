#include <cstddef>
#include <riscv_vector.h>

using namespace std;

/*********************************** Scalar Version ************************************/

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


/********************************* Vectorized Versions *********************************/

void bias_add_e32m1(const float* input, const float* bias, float* output,
                      size_t batch_size, size_t channels,
                      size_t height, size_t width) {
    
    size_t channel_size = height * width;
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            float b_val = bias[c]; // Scalar bias
            size_t offset = (b * channels + c) * channel_size;
            
            const float* in_ptr = input + offset;
            float* out_ptr = output + offset;
            
            size_t cnt = channel_size;
            size_t vl;
            
            while (cnt > 0) {
                vl = __riscv_vsetvl_e32m1(cnt);
                
                // Load vector from input
                vfloat32m1_t v_input = __riscv_vle32_v_f32m1(in_ptr, vl);
                
                // Add scalar bias value to the vector
                vfloat32m1_t v_output = __riscv_vfadd_vf_f32m1(v_input, b_val, vl);
                
                // Store result in output
                __riscv_vse32_v_f32m1(out_ptr, v_output, vl);
                
                in_ptr += vl;
                out_ptr += vl;
                cnt -= vl;
            }
        }
    }
}

void bias_add_e32m2(const float* input, const float* bias, float* output,
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
                vl = __riscv_vsetvl_e32m2(cnt);
                vfloat32m2_t v_input = __riscv_vle32_v_f32m2(in_ptr, vl);
                vfloat32m2_t v_output = __riscv_vfadd_vf_f32m2(v_input, b_val, vl);
                __riscv_vse32_v_f32m2(out_ptr, v_output, vl);
                
                in_ptr += vl;
                out_ptr += vl;
                cnt -= vl;
            }
        }
    }
}

void bias_add_e32m4(const float* input, const float* bias, float* output,
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
                vl = __riscv_vsetvl_e32m4(cnt);
                vfloat32m4_t v_input = __riscv_vle32_v_f32m4(in_ptr, vl);
                vfloat32m4_t v_output = __riscv_vfadd_vf_f32m4(v_input, b_val, vl);
                __riscv_vse32_v_f32m4(out_ptr, v_output, vl);
                
                in_ptr += vl;
                out_ptr += vl;
                cnt -= vl;
            }
        }
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