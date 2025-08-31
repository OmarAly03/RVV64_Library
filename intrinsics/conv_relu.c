#include <riscv_vector.h>
#include <stddef.h>

void conv_layer(float *input, float *weights, float *biases,
                int in_h, int in_w, int in_c,
                int num_filters, int filter_size,
                int stride, int padding,
                float *output) {
    int out_h = (in_h + 2*padding - filter_size) / stride + 1;
    int out_w = (in_w + 2*padding - filter_size) / stride + 1;

    for (int oy = 0; oy < out_h; oy++) {
        for (int ox = 0; ox < out_w; ox++) {
            for (int f = 0; f < num_filters; f++) {
                float sum;
                // Initialize vector accumulator to zero across channels
                vfloat32m1_t v_acc;
                int rem_init = in_c;
                
                while (rem_init > 0) {
                    size_t vl = __riscv_vsetvl_e32m1(rem_init);
                    v_acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                    rem_init -= vl;
                }

                // Accumulate over filter window
                for (int ky = 0; ky < filter_size; ky++) {
                    for (int kx = 0; kx < filter_size; kx++) {
                        int in_y = oy * stride + ky - padding;
                        int in_x = ox * stride + kx - padding;
                        
                        // Skip if outside input boundaries (padding area)
                        if (in_y < 0 || in_y >= in_h || in_x < 0 || in_x >= in_w)
                            continue;
                            
                        float *in_ptr = input + (in_y * in_w + in_x) * in_c;
                        float *w_ptr = weights + ((f * filter_size + ky) * filter_size + kx) * in_c;
                        int rem = in_c;
                        
                        while (rem > 0) {
                            size_t vl = __riscv_vsetvl_e32m1(rem);
                            
                            // Load input and weight vectors
                            vfloat32m1_t v_in = __riscv_vle32_v_f32m1(in_ptr, vl);
                            vfloat32m1_t v_weight = __riscv_vle32_v_f32m1(w_ptr, vl);
                            
                            // Fused multiply-accumulate
                            v_acc = __riscv_vfmacc_vv_f32m1(v_acc, v_in, v_weight, vl);
                            
                            in_ptr += vl;
                            w_ptr += vl;
                            rem -= vl;
                        }
                    }
                }
                
                // Reduce accumulator to scalar sum
                vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
                vfloat32m1_t v_sum = __riscv_vfredusum_vs_f32m1_f32m1(v_acc, v_zero, in_c);
                sum = __riscv_vfmv_f_s_f32m1_f32(v_sum);
                
                // Add bias once per filter output
                sum += biases[f];

                // Store result
                output[(oy * out_w + ox) * num_filters + f] = sum;
            }
        }
    }
}

void relu_activation(float *input, int h, int w, int c, float *output) {
    int total = h * w * c;
    int idx = 0;
    const float zero = 0.0f;
    
    while (idx < total) {
        // Determine vector length for this iteration
        size_t vl = __riscv_vsetvl_e32m1(total - idx);
        
        // Load input values
        vfloat32m1_t v_input = __riscv_vle32_v_f32m1(input + idx, vl);
        
        // Create vector with zeros
        vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(zero, vl);
        
        // Compute max(input, 0)
        vfloat32m1_t v_result = __riscv_vfmax_vv_f32m1(v_input, v_zero, vl);
        
        // Store result
        __riscv_vse32_v_f32m1(output + idx, v_result, vl);
        
        // Move to next chunk
        idx += vl;
    }
}