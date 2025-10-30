#include <cstddef>
#include <riscv_vector.h>
#include <cmath>      // For expf
#include <cfloat>     // For -FLT_MAX / __builtin_inff

using namespace std;

/*********************************** Scalar Version ************************************/

void softmax_scalar(float* input, float* output, size_t size) {
    // Pass 1: Find Max
    float max_val = -__builtin_inff();
    for (size_t i = 0; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // Pass 2: Calculate Exponentials and Sum
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    // Pass 3: Divide by Sum
    for (size_t i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

/********************************* Vectorized Versions *********************************/

// Helper function for the vectorized exponential and sum pass (Pass 3)
// This pass remains scalar because expf() is not a vector intrinsic.
float scalar_exp_and_sum(float* data, size_t size) {
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        data[i] = expf(data[i]);
        sum += data[i];
    }
    return sum;
}


void softmax_e32m1(float* input, float* output, size_t size) {
    float* in_ptr = input;
    float* out_ptr = output;
    size_t cnt = size;
    size_t vl;

    // === Pass 1: Find Max (Reduction) ===
    float max_val = -__builtin_inff();
    // Init max vector (m1)
    vfloat32m1_t v_max = __riscv_vfmv_v_f_f32m1(max_val, __riscv_vsetvl_e32m1(1)); 
    
    for (cnt = size; cnt > 0; cnt -= vl) {
        vl = __riscv_vsetvl_e32m1(cnt);
        vfloat32m1_t v_in = __riscv_vle32_v_f32m1(in_ptr, vl);
        // Correct argument order: (source_vector, scalar_dest, vl)
        v_max = __riscv_vfredmax_vs_f32m1_f32m1(v_in, v_max, vl);
        in_ptr += vl;
    }
    max_val = __riscv_vfmv_f_s_f32m1_f32(v_max); // Extract scalar max

    // === Pass 2: Subtract Max (Element-wise) ===
    in_ptr = input;
    out_ptr = output;
    vfloat32m1_t v_max_splat = __riscv_vfmv_v_f_f32m1(max_val, __riscv_vsetvl_e32m1(size)); 
    
    for (cnt = size; cnt > 0; cnt -= vl) {
        vl = __riscv_vsetvl_e32m1(cnt);
        vfloat32m1_t v_in = __riscv_vle32_v_f32m1(in_ptr, vl);
        vfloat32m1_t v_sub = __riscv_vfsub_vv_f32m1(v_in, v_max_splat, vl);
        __riscv_vse32_v_f32m1(out_ptr, v_sub, vl); 
        in_ptr += vl;
        out_ptr += vl;
    }

    // === Pass 3: Exp & Sum (Scalar) ===
    float sum = scalar_exp_and_sum(output, size);

    // === Pass 4: Divide by Sum (Element-wise) ===
    out_ptr = output;
    vfloat32m1_t v_sum_splat = __riscv_vfmv_v_f_f32m1(sum, __riscv_vsetvl_e32m1(size)); 

    for (cnt = size; cnt > 0; cnt -= vl) {
        vl = __riscv_vsetvl_e32m1(cnt);
        vfloat32m1_t v_exp = __riscv_vle32_v_f32m1(out_ptr, vl);
        vfloat32m1_t v_out = __riscv_vfdiv_vv_f32m1(v_exp, v_sum_splat, vl);
        __riscv_vse32_v_f32m1(out_ptr, v_out, vl);
        out_ptr += vl;
    }
}

void softmax_e32m2(float* input, float* output, size_t size) {
    float* in_ptr = input;
    float* out_ptr = output;
    size_t cnt = size;
    size_t vl;

    // === Pass 1: Find Max (Reduction) ===
    float max_val = -__builtin_inff();
    // Reduction destination must be m1
    vfloat32m1_t v_max = __riscv_vfmv_v_f_f32m1(max_val, __riscv_vsetvl_e32m1(1)); 
    for (cnt = size; cnt > 0; cnt -= vl) {
        vl = __riscv_vsetvl_e32m2(cnt);
        vfloat32m2_t v_in = __riscv_vle32_v_f32m2(in_ptr, vl);
        // Use m2->m1 reduction intrinsic
        v_max = __riscv_vfredmax_vs_f32m2_f32m1(v_in, v_max, vl);
        in_ptr += vl;
    }
    max_val = __riscv_vfmv_f_s_f32m1_f32(v_max); 

    // === Pass 2: Subtract Max (Element-wise) ===
    in_ptr = input;
    out_ptr = output;
    vfloat32m2_t v_max_splat = __riscv_vfmv_v_f_f32m2(max_val, __riscv_vsetvl_e32m2(size));
    for (cnt = size; cnt > 0; cnt -= vl) {
        vl = __riscv_vsetvl_e32m2(cnt);
        vfloat32m2_t v_in = __riscv_vle32_v_f32m2(in_ptr, vl);
        vfloat32m2_t v_sub = __riscv_vfsub_vv_f32m2(v_in, v_max_splat, vl);
        __riscv_vse32_v_f32m2(out_ptr, v_sub, vl);
        in_ptr += vl;
        out_ptr += vl;
    }

    // === Pass 3: Exp & Sum (Scalar) ===
    float sum = scalar_exp_and_sum(output, size);

    // === Pass 4: Divide by Sum (Element-wise) ===
    out_ptr = output;
    vfloat32m2_t v_sum_splat = __riscv_vfmv_v_f_f32m2(sum, __riscv_vsetvl_e32m2(size));
    for (cnt = size; cnt > 0; cnt -= vl) {
        vl = __riscv_vsetvl_e32m2(cnt);
        vfloat32m2_t v_exp = __riscv_vle32_v_f32m2(out_ptr, vl);
        vfloat32m2_t v_out = __riscv_vfdiv_vv_f32m2(v_exp, v_sum_splat, vl);
        __riscv_vse32_v_f32m2(out_ptr, v_out, vl);
        out_ptr += vl;
    }
}

void softmax_e32m4(float* input, float* output, size_t size) {
    float* in_ptr = input;
    float* out_ptr = output;
    size_t cnt = size;
    size_t vl;

    // === Pass 1: Find Max (Reduction) ===
    float max_val = -__builtin_inff();
    // Reduction destination must be m1
    vfloat32m1_t v_max = __riscv_vfmv_v_f_f32m1(max_val, __riscv_vsetvl_e32m1(1)); 
    for (cnt = size; cnt > 0; cnt -= vl) {
        vl = __riscv_vsetvl_e32m4(cnt);
        vfloat32m4_t v_in = __riscv_vle32_v_f32m4(in_ptr, vl);
        // Use m4->m1 reduction intrinsic
        v_max = __riscv_vfredmax_vs_f32m4_f32m1(v_in, v_max, vl);
        in_ptr += vl;
    }
    max_val = __riscv_vfmv_f_s_f32m1_f32(v_max); 

    // === Pass 2: Subtract Max (Element-wise) ===
    in_ptr = input;
    out_ptr = output;
    vfloat32m4_t v_max_splat = __riscv_vfmv_v_f_f32m4(max_val, __riscv_vsetvl_e32m4(size));
    for (cnt = size; cnt > 0; cnt -= vl) {
        vl = __riscv_vsetvl_e32m4(cnt);
        vfloat32m4_t v_in = __riscv_vle32_v_f32m4(in_ptr, vl);
        vfloat32m4_t v_sub = __riscv_vfsub_vv_f32m4(v_in, v_max_splat, vl);
        __riscv_vse32_v_f32m4(out_ptr, v_sub, vl);
        in_ptr += vl;
        out_ptr += vl;
    }

    // === Pass 3: Exp & Sum (Scalar) ===
    float sum = scalar_exp_and_sum(output, size);

    // === Pass 4: Divide by Sum (Element-wise) ===
    out_ptr = output;
    vfloat32m4_t v_sum_splat = __riscv_vfmv_v_f_f32m4(sum, __riscv_vsetvl_e32m4(size));
    for (cnt = size; cnt > 0; cnt -= vl) {
        vl = __riscv_vsetvl_e32m4(cnt);
        vfloat32m4_t v_exp = __riscv_vle32_v_f32m4(out_ptr, vl);
        vfloat32m4_t v_out = __riscv_vfdiv_vv_f32m4(v_exp, v_sum_splat, vl);
        __riscv_vse32_v_f32m4(out_ptr, v_out, vl);
        out_ptr += vl;
    }
}

void softmax_e32m8(float* input, float* output, size_t size) {
    float* in_ptr = input;
    float* out_ptr = output;
    size_t cnt = size;
    size_t vl;

    // === Pass 1: Find Max (Reduction) ===
    float max_val = -__builtin_inff();
    // Reduction destination must be m1
    vfloat32m1_t v_max = __riscv_vfmv_v_f_f32m1(max_val, __riscv_vsetvl_e32m1(1)); 
    for (cnt = size; cnt > 0; cnt -= vl) {
        vl = __riscv_vsetvl_e32m8(cnt);
        vfloat32m8_t v_in = __riscv_vle32_v_f32m8(in_ptr, vl);
        // Use m8->m1 reduction intrinsic
        v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_in, v_max, vl);
        in_ptr += vl;
    }
    max_val = __riscv_vfmv_f_s_f32m1_f32(v_max); 

    // === Pass 2: Subtract Max (Element-wise) ===
    in_ptr = input;
    out_ptr = output;
    vfloat32m8_t v_max_splat = __riscv_vfmv_v_f_f32m8(max_val, __riscv_vsetvl_e32m8(size));
    for (cnt = size; cnt > 0; cnt -= vl) {
        vl = __riscv_vsetvl_e32m8(cnt);
        vfloat32m8_t v_in = __riscv_vle32_v_f32m8(in_ptr, vl);
        vfloat32m8_t v_sub = __riscv_vfsub_vv_f32m8(v_in, v_max_splat, vl);
        __riscv_vse32_v_f32m8(out_ptr, v_sub, vl);
        in_ptr += vl;
        out_ptr += vl;
    }

    // === Pass 3: Exp & Sum (Scalar) ===
    float sum = scalar_exp_and_sum(output, size);

    // === Pass 4: Divide by Sum (Element-wise) ===
    out_ptr = output;
    vfloat32m8_t v_sum_splat = __riscv_vfmv_v_f_f32m8(sum, __riscv_vsetvl_e32m8(size));
    for (cnt = size; cnt > 0; cnt -= vl) {
        vl = __riscv_vsetvl_e32m8(cnt);
        vfloat32m8_t v_exp = __riscv_vle32_v_f32m8(out_ptr, vl);
        vfloat32m8_t v_out = __riscv_vfdiv_vv_f32m8(v_exp, v_sum_splat, vl);
        __riscv_vse32_v_f32m8(out_ptr, v_out, vl);
        out_ptr += vl;
    }
}