#include <riscv_vector.h>
#include <cstddef>

using namespace std;

// Scalar implementation of 1D NMS with a window of 3
void nms_scalar(float* input, float* output, size_t size) {
    if (size == 0) return;

    if (size == 1) {
        output[0] = input[0];
        return;
    }

    // First element
    output[0] = (input[0] >= input[1]) ? input[0] : 0.0f;

    // Middle elements
    for (size_t i = 1; i < size - 1; i++) {
        float center = input[i];
        if (center >= input[i-1] && center >= input[i+1]) {
            output[i] = center;
        } else {
            output[i] = 0.0f;
        }
    }

    // Last element
    output[size - 1] = (input[size - 1] >= input[size - 2]) ? input[size - 1] : 0.0f;
}


void nms_e32m1(float* input, float* output, size_t size) {
    if (size < 3) {
        nms_scalar(input, output, size);
        return;
    }

    output[0] = (input[0] >= input[1]) ? input[0] : 0.0f;
    output[size - 1] = (input[size - 1] >= input[size - 2]) ? input[size - 1] : 0.0f;

    float* in_ptr = input + 1;
    float* out_ptr = output + 1;
    size_t cnt = size - 2;

    while (cnt > 0) {
        size_t vl = __riscv_vsetvl_e32m1(cnt);
        vfloat32m1_t v_center = __riscv_vle32_v_f32m1(in_ptr, vl);
        vfloat32m1_t v_left = __riscv_vle32_v_f32m1(in_ptr - 1, vl);
        vfloat32m1_t v_right = __riscv_vle32_v_f32m1(in_ptr + 1, vl);

        vbool32_t is_ge_left = __riscv_vmfge_vv_f32m1_b32(v_center, v_left, vl);
        vbool32_t is_ge_right = __riscv_vmfge_vv_f32m1_b32(v_center, v_right, vl);
        vbool32_t is_max = __riscv_vmand_mm_b32(is_ge_left, is_ge_right, vl);

        vfloat32m1_t v_result = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        v_result = __riscv_vmerge_vvm_f32m1(v_result, v_center, is_max, vl);

        __riscv_vse32_v_f32m1(out_ptr, v_result, vl);

        cnt -= vl;
        in_ptr += vl;
        out_ptr += vl;
    }
}

void nms_e32m2(float* input, float* output, size_t size) {
    if (size < 3) {
        nms_scalar(input, output, size);
        return;
    }

    output[0] = (input[0] >= input[1]) ? input[0] : 0.0f;
    output[size - 1] = (input[size - 1] >= input[size - 2]) ? input[size - 1] : 0.0f;

    float* in_ptr = input + 1;
    float* out_ptr = output + 1;
    size_t cnt = size - 2;

    while (cnt > 0) {
        size_t vl = __riscv_vsetvl_e32m2(cnt);
        vfloat32m2_t v_center = __riscv_vle32_v_f32m2(in_ptr, vl);
        vfloat32m2_t v_left = __riscv_vle32_v_f32m2(in_ptr - 1, vl);
        vfloat32m2_t v_right = __riscv_vle32_v_f32m2(in_ptr + 1, vl);

        vbool16_t is_ge_left = __riscv_vmfge_vv_f32m2_b16(v_center, v_left, vl);
        vbool16_t is_ge_right = __riscv_vmfge_vv_f32m2_b16(v_center, v_right, vl);
        vbool16_t is_max = __riscv_vmand_mm_b16(is_ge_left, is_ge_right, vl);

        vfloat32m2_t v_result = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        v_result = __riscv_vmerge_vvm_f32m2(v_result, v_center, is_max, vl);

        __riscv_vse32_v_f32m2(out_ptr, v_result, vl);

        cnt -= vl;
        in_ptr += vl;
        out_ptr += vl;
    }
}

void nms_e32m4(float* input, float* output, size_t size) {
    if (size < 3) {
        nms_scalar(input, output, size);
        return;
    }

    output[0] = (input[0] >= input[1]) ? input[0] : 0.0f;
    output[size - 1] = (input[size - 1] >= input[size - 2]) ? input[size - 1] : 0.0f;

    float* in_ptr = input + 1;
    float* out_ptr = output + 1;
    size_t cnt = size - 2;

    while (cnt > 0) {
        size_t vl = __riscv_vsetvl_e32m4(cnt);
        vfloat32m4_t v_center = __riscv_vle32_v_f32m4(in_ptr, vl);
        vfloat32m4_t v_left = __riscv_vle32_v_f32m4(in_ptr - 1, vl);
        vfloat32m4_t v_right = __riscv_vle32_v_f32m4(in_ptr + 1, vl);

        vbool8_t is_ge_left = __riscv_vmfge_vv_f32m4_b8(v_center, v_left, vl);
        vbool8_t is_ge_right = __riscv_vmfge_vv_f32m4_b8(v_center, v_right, vl);
        vbool8_t is_max = __riscv_vmand_mm_b8(is_ge_left, is_ge_right, vl);

        vfloat32m4_t v_result = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        v_result = __riscv_vmerge_vvm_f32m4(v_result, v_center, is_max, vl);

        __riscv_vse32_v_f32m4(out_ptr, v_result, vl);

        cnt -= vl;
        in_ptr += vl;
        out_ptr += vl;
    }
}

void nms_e32m8(float* input, float* output, size_t size) {
    if (size < 3) {
        nms_scalar(input, output, size);
        return;
    }

    output[0] = (input[0] >= input[1]) ? input[0] : 0.0f;
    output[size - 1] = (input[size - 1] >= input[size - 2]) ? input[size - 1] : 0.0f;

    float* in_ptr = input + 1;
    float* out_ptr = output + 1;
    size_t cnt = size - 2;

    while (cnt > 0) {
        size_t vl = __riscv_vsetvl_e32m8(cnt);
        vfloat32m8_t v_center = __riscv_vle32_v_f32m8(in_ptr, vl);
        vfloat32m8_t v_left = __riscv_vle32_v_f32m8(in_ptr - 1, vl);
        vfloat32m8_t v_right = __riscv_vle32_v_f32m8(in_ptr + 1, vl);

        vbool4_t is_ge_left = __riscv_vmfge_vv_f32m8_b4(v_center, v_left, vl);
        vbool4_t is_ge_right = __riscv_vmfge_vv_f32m8_b4(v_center, v_right, vl);
        vbool4_t is_max = __riscv_vmand_mm_b4(is_ge_left, is_ge_right, vl);

        vfloat32m8_t v_result = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        v_result = __riscv_vmerge_vvm_f32m8(v_result, v_center, is_max, vl);

        __riscv_vse32_v_f32m8(out_ptr, v_result, vl);

        cnt -= vl;
        in_ptr += vl;
        out_ptr += vl;
    }
}
