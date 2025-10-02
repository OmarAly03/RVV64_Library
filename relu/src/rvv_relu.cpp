#include <cstddef>
#include <riscv_vector.h>

using namespace std;

void relu_e32m1(float* input, float* output, size_t size) {
    float* in_ptr = input;
    float* out_ptr = output;
    
    for (size_t cnt = size; cnt > 0; ) {
        size_t vl = __riscv_vsetvl_e32m1(cnt);
        vfloat32m1_t v_input = __riscv_vle32_v_f32m1(in_ptr, vl);
        vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t v_result = __riscv_vfmax_vv_f32m1(v_input, v_zero, vl);
        __riscv_vse32_v_f32m1(out_ptr, v_result, vl);
        
        cnt -= vl;
        in_ptr += vl;
        out_ptr += vl;
    }
}

void relu_e32m2(float* input, float* output, size_t size) {
    float* in_ptr = input;
    float* out_ptr = output;
    
    for (size_t cnt = size; cnt > 0; ) {
        size_t vl = __riscv_vsetvl_e32m2(cnt);
        vfloat32m2_t v_input = __riscv_vle32_v_f32m2(in_ptr, vl);
        vfloat32m2_t v_zero = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t v_result = __riscv_vfmax_vv_f32m2(v_input, v_zero, vl);
        __riscv_vse32_v_f32m2(out_ptr, v_result, vl);
        
        cnt -= vl;
        in_ptr += vl;
        out_ptr += vl;
    }
}

void relu_e32m4(float* input, float* output, size_t size) {
    float* in_ptr = input;
    float* out_ptr = output;
    
    for (size_t cnt = size; cnt > 0; ) {
        size_t vl = __riscv_vsetvl_e32m4(cnt);
        vfloat32m4_t v_input = __riscv_vle32_v_f32m4(in_ptr, vl);
        vfloat32m4_t v_zero = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t v_result = __riscv_vfmax_vv_f32m4(v_input, v_zero, vl);
        __riscv_vse32_v_f32m4(out_ptr, v_result, vl);
        
        cnt -= vl;
        in_ptr += vl;
        out_ptr += vl;
    }
}

void relu_e32m8(float* input, float* output, size_t size) {
    float* in_ptr = input;
    float* out_ptr = output;
    
    for (size_t cnt = size; cnt > 0; ) {
        size_t vl = __riscv_vsetvl_e32m8(cnt);
        vfloat32m8_t v_input = __riscv_vle32_v_f32m8(in_ptr, vl);
        vfloat32m8_t v_zero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        vfloat32m8_t v_result = __riscv_vfmax_vv_f32m8(v_input, v_zero, vl);
        __riscv_vse32_v_f32m8(out_ptr, v_result, vl);
        
        cnt -= vl;
        in_ptr += vl;
        out_ptr += vl;
    }
}

void relu_scalar(float* input, float* output, size_t size) {
    for (size_t i = 0; i < size; i++) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}