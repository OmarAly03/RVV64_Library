#include <riscv_vector.h>
#include <stddef.h>

extern "C" {

    // =========================================================
    // Vectorized Dense Layer (e32m8)
    // Strategy: Vectorized Dot Product (Reduction)
    // Layout Assumption: Weights are [Out_Dim, In_Dim] (Row-Major)
    // =========================================================
    void dense_e32m8(const float* input, const float* weights, const float* bias, float* output, 
                     int in_dim, int out_dim) {

        for (int m = 0; m < out_dim; ++m) {
            
            // 1. Initialize sum with bias (scalar load)
            float sum = bias ? bias[m] : 0.0f;
            
            // Pointer to the start of this neuron's weights
            const float* w_row = weights + m * in_dim;

            // 2. Vectorized Dot Product
            // Loop over the input dimension 'in_dim'
            int k = 0;
            
            // Vector accumulator for the sum
            vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            vfloat32m8_t v_sum_vec = __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvlmax_e32m8());

            for (; k < in_dim; ) {
                size_t vl = __riscv_vsetvl_e32m8(in_dim - k);

                // Unit-stride loads (Fastest)
                vfloat32m8_t v_in = __riscv_vle32_v_f32m8(input + k, vl);
                vfloat32m8_t v_w = __riscv_vle32_v_f32m8(w_row + k, vl);

                // Fused Multiply-Add
                v_sum_vec = __riscv_vfmacc_vv_f32m8(v_sum_vec, v_in, v_w, vl);

                k += vl;
            }

            // 3. Reduction
            // Sum all elements in the vector register down to a scalar
            // Note: vfredusum is unordered reduction (faster), vfredosum is ordered
            vfloat32m1_t v_res = __riscv_vfredusum_vs_f32m8_f32m1(v_sum_vec, v_zero, __riscv_vsetvlmax_e32m8());
            
            sum += __riscv_vfmv_f_s_f32m1_f32(v_res);

            // 4. Store Result
            output[m] = sum;
        }
    }

    // Scalar Reference
    void dense_scalar(const float* input, const float* weights, const float* bias, float* output, 
                      int in_dim, int out_dim) {
        for (int m = 0; m < out_dim; ++m) {
            float sum = bias ? bias[m] : 0.0f;
            for (int k = 0; k < in_dim; ++k) {
                sum += input[k] * weights[m * in_dim + k];
            }
            output[m] = sum;
        }
    }
}