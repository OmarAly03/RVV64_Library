#include <cstddef>
#include <riscv_vector.h>

using namespace std;

/*********************************** Scalar Version (Non-Batched) ************************************/

// This is your original scalar version, slightly adapted to be non-batched
void dense_scalar(const float* input, const float* weights, const float* bias,
                        float* output, size_t in_features, size_t out_features) {
    // Implements Y = A*B^T + C, where A=input, B=weights, C=bias
    // A shape: [in_features]
    // B shape: [out_features, in_features]
    // C shape: [out_features]
    // Y shape: [out_features]
    for (size_t out_f = 0; out_f < out_features; ++out_f) {
        float sum = 0.0f;
        for (size_t in_f = 0; in_f < in_features; ++in_f) {
            sum += input[in_f] * weights[out_f * in_features + in_f];
        }
        output[out_f] = sum + bias[out_f];
    }
}


/********************************* Vectorized Versions (Non-Batched) *********************************/

void dense_e32m1(const float* input, const float* weights, const float* bias,
                   float* output, size_t in_features, size_t out_features) {
    for (size_t out_f = 0; out_f < out_features; ++out_f) {
        
        // Pointers are simpler: input is the start of the vector.
        const float* a_ptr = input;
        const float* b_ptr = &weights[out_f * in_features];
        size_t cnt = in_features;
        size_t vl;

        vfloat32m1_t v_sum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(in_features));

        for (; cnt > 0; cnt -= vl) {
            vl = __riscv_vsetvl_e32m1(cnt);
            
            vfloat32m1_t v_a = __riscv_vle32_v_f32m1(a_ptr, vl);
            vfloat32m1_t v_b = __riscv_vle32_v_f32m1(b_ptr, vl);
            v_sum = __riscv_vfmacc_vv_f32m1(v_sum, v_a, v_b, vl);
            
            a_ptr += vl;
            b_ptr += vl;
        }

        vfloat32m1_t v_scalar_sum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
        v_scalar_sum = __riscv_vfredusum_vs_f32m1_f32m1(v_sum, v_scalar_sum, __riscv_vsetvl_e32m1(in_features));
        float sum = __riscv_vfmv_f_s_f32m1_f32(v_scalar_sum);

        // Output is simpler: just [out_f]
        output[out_f] = sum + bias[out_f];
    }
}

void dense_e32m2(const float* input, const float* weights, const float* bias,
                   float* output, size_t in_features, size_t out_features) {
    for (size_t out_f = 0; out_f < out_features; ++out_f) {
        
        const float* a_ptr = input;
        const float* b_ptr = &weights[out_f * in_features];
        size_t cnt = in_features;
        size_t vl;

        vfloat32m2_t v_sum = __riscv_vfmv_v_f_f32m2(0.0f, __riscv_vsetvl_e32m2(in_features));

        for (; cnt > 0; cnt -= vl) {
            vl = __riscv_vsetvl_e32m2(cnt);
            vfloat32m2_t v_a = __riscv_vle32_v_f32m2(a_ptr, vl);
            vfloat32m2_t v_b = __riscv_vle32_v_f32m2(b_ptr, vl);
            v_sum = __riscv_vfmacc_vv_f32m2(v_sum, v_a, v_b, vl);
            a_ptr += vl;
            b_ptr += vl;
        }

        vfloat32m1_t v_scalar_sum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
        v_scalar_sum = __riscv_vfredusum_vs_f32m2_f32m1(v_sum, v_scalar_sum, __riscv_vsetvl_e32m2(in_features));
        float sum = __riscv_vfmv_f_s_f32m1_f32(v_scalar_sum);

        output[out_f] = sum + bias[out_f];
    }
}

void dense_e32m4(const float* input, const float* weights, const float* bias,
                   float* output, size_t in_features, size_t out_features) {
    for (size_t out_f = 0; out_f < out_features; ++out_f) {
        
        const float* a_ptr = input;
        const float* b_ptr = &weights[out_f * in_features];
        size_t cnt = in_features;
        size_t vl;

        vfloat32m4_t v_sum = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvl_e32m4(in_features));

        for (; cnt > 0; cnt -= vl) {
            vl = __riscv_vsetvl_e32m4(cnt);
            vfloat34m1_t v_a = __riscv_vle32_v_f32m4(a_ptr, vl);
            vfloat34m1_t v_b = __riscv_vle32_v_f32m4(b_ptr, vl);
            v_sum = __riscv_vfmacc_vv_f32m4(v_sum, v_a, v_b, vl);
            a_ptr += vl;
            b_ptr += vl;
        }

        vfloat32m1_t v_scalar_sum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
        v_scalar_sum = __riscv_vfredusum_vs_f32m4_f32m1(v_sum, v_scalar_sum, __riscv_vsetvl_e32m4(in_features));
        float sum = __riscv_vfmv_f_s_f32m1_f32(v_scalar_sum);

        output[out_f] = sum + bias[out_f];
    }
}

void dense_e32m8(const float* input, const float* weights, const float* bias,
                   float* output, size_t in_features, size_t out_features) {
    for (size_t out_f = 0; out_f < out_features; ++out_f) {
        
        const float* a_ptr = input;
        const float* b_ptr = &weights[out_f * in_features];
        size_t cnt = in_features;
        size_t vl;

        vfloat32m8_t v_sum = __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvl_e32m8(in_features));

        for (; cnt > 0; cnt -= vl) {
            vl = __riscv_vsetvl_e32m8(cnt);
            vfloat32m8_t v_a = __riscv_vle32_v_f32m8(a_ptr, vl);
            vfloat32m8_t v_b = __riscv_vle32_v_f32m8(b_ptr, vl);
            v_sum = __riscv_vfmacc_vv_f32m8(v_sum, v_a, v_b, vl);
            a_ptr += vl;
            b_ptr += vl;
        }

        vfloat32m1_t v_scalar_sum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
        v_scalar_sum = __riscv_vfredusum_vs_f32m8_f32m1(v_sum, v_scalar_sum, __riscv_vsetvl_e32m8(in_features));
        float sum = __riscv_vfmv_f_s_f32m1_f32(v_scalar_sum);

        output[out_f] = sum + bias[out_f];
    }
}