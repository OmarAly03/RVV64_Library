#include <riscv_vector.h>
#include <stddef.h>

extern "C" {

void tensor_add_e32m8(const float* input_a, const float* input_b, float* output, size_t size) {
    const float* a = input_a;
    const float* b = input_b;
    float* out = output;
    size_t cnt = size;
    
    while (cnt > 0) {
        size_t vl = __riscv_vsetvl_e32m8(cnt);
        vfloat32m8_t v_a = __riscv_vle32_v_f32m8(a, vl);
        vfloat32m8_t v_b = __riscv_vle32_v_f32m8(b, vl);
        vfloat32m8_t v_o = __riscv_vfadd_vv_f32m8(v_a, v_b, vl);
        __riscv_vse32_v_f32m8(out, v_o, vl);
        
        a += vl; b += vl; out += vl; cnt -= vl;
    }
}

void tensor_add_scalar(const float* input_a, const float* input_b, float* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = input_a[i] + input_b[i];
    }
}

}