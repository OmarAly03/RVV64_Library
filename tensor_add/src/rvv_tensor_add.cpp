#include <cstddef>
#include <riscv_vector.h>

using namespace std;

/*********************************** Scalar Version ************************************/

void tensor_add_scalar(const float* input_a, const float* input_b, float* output,
                           size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = input_a[i] + input_b[i];
    }
}

/********************************* Vectorized Versions *********************************/

void tensor_add_e32m1(const float* input_a, const float* input_b, float* output,
                           size_t size) {
    const float* in_a_ptr = input_a;
    const float* in_b_ptr = input_b;
    float* out_ptr = output;
    
    size_t cnt = size;
    size_t vl;

    while (cnt > 0) {
        vl = __riscv_vsetvl_e32m1(cnt);
        
        vfloat32m1_t v_a = __riscv_vle32_v_f32m1(in_a_ptr, vl);
        vfloat32m1_t v_b = __riscv_vle32_v_f32m1(in_b_ptr, vl);
        
        // v_out = v_a + v_b
        vfloat32m1_t v_out = __riscv_vfadd_vv_f32m1(v_a, v_b, vl);
        
        __riscv_vse32_v_f32m1(out_ptr, v_out, vl);
        
        in_a_ptr += vl;
        in_b_ptr += vl;
        out_ptr += vl;
        cnt -= vl;
    }
}

void tensor_add_e32m2(const float* input_a, const float* input_b, float* output,
                           size_t size) {
    const float* in_a_ptr = input_a;
    const float* in_b_ptr = input_b;
    float* out_ptr = output;
    
    size_t cnt = size;
    size_t vl;

    while (cnt > 0) {
        vl = __riscv_vsetvl_e32m2(cnt);
        vfloat32m2_t v_a = __riscv_vle32_v_f32m2(in_a_ptr, vl);
        vfloat32m2_t v_b = __riscv_vle32_v_f32m2(in_b_ptr, vl);
        vfloat32m2_t v_out = __riscv_vfadd_vv_f32m2(v_a, v_b, vl);
        __riscv_vse32_v_f32m2(out_ptr, v_out, vl);
        in_a_ptr += vl;
        in_b_ptr += vl;
        out_ptr += vl;
        cnt -= vl;
    }
}

void tensor_add_e32m4(const float* input_a, const float* input_b, float* output,
                           size_t size) {
    const float* in_a_ptr = input_a;
    const float* in_b_ptr = input_b;
    float* out_ptr = output;
    
    size_t cnt = size;
    size_t vl;

    while (cnt > 0) {
        vl = __riscv_vsetvl_e32m4(cnt);
        vfloat32m4_t v_a = __riscv_vle32_v_f32m4(in_a_ptr, vl);
        vfloat32m4_t v_b = __riscv_vle32_v_f32m4(in_b_ptr, vl);
        vfloat32m4_t v_out = __riscv_vfadd_vv_f32m4(v_a, v_b, vl);
        __riscv_vse32_v_f32m4(out_ptr, v_out, vl);
        in_a_ptr += vl;
        in_b_ptr += vl;
        out_ptr += vl;
        cnt -= vl;
    }
}

void tensor_add_e32m8(const float* input_a, const float* input_b, float* output,
                           size_t size) {
    const float* in_a_ptr = input_a;
    const float* in_b_ptr = input_b;
    float* out_ptr = output;
    
    size_t cnt = size;
    size_t vl;

    while (cnt > 0) {
        vl = __riscv_vsetvl_e32m8(cnt);
        vfloat32m8_t v_a = __riscv_vle32_v_f32m8(in_a_ptr, vl);
        vfloat32m8_t v_b = __riscv_vle32_v_f32m8(in_b_ptr, vl);
        vfloat32m8_t v_out = __riscv_vfadd_vv_f32m8(v_a, v_b, vl);
        __riscv_vse32_v_f32m8(out_ptr, v_out, vl);
        in_a_ptr += vl;
        in_b_ptr += vl;
        out_ptr += vl;
        cnt -= vl;
    }
}