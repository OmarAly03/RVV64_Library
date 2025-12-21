#include <cstddef>
#include <riscv_vector.h>
#include "rvv_defs.hpp"

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
        vl = SET_VECTOR_LENGTH<float, M1>(cnt);
        auto v_a = VECTOR_LOAD<float, M1>(in_a_ptr, vl);
        auto v_b = VECTOR_LOAD<float, M1>(in_b_ptr, vl);
        auto v_out = VECTOR_ADD<float, M1>(v_a, v_b, vl);
        VECTOR_STORE<float, M1>(out_ptr, v_out, vl);
        
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
        vl = SET_VECTOR_LENGTH<float, M2>(cnt);
        auto v_a = VECTOR_LOAD<float, M2>(in_a_ptr, vl);
        auto v_b = VECTOR_LOAD<float, M2>(in_b_ptr, vl);
        auto v_out = VECTOR_ADD<float, M2>(v_a, v_b, vl);
        VECTOR_STORE<float, M2>(out_ptr, v_out, vl);

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
        vl = SET_VECTOR_LENGTH<float, M4>(cnt);
        auto v_a = VECTOR_LOAD<float, M4>(in_a_ptr, vl);
        auto v_b = VECTOR_LOAD<float, M4>(in_b_ptr, vl);
        auto v_out = VECTOR_ADD<float, M4>(v_a, v_b, vl);
        VECTOR_STORE<float, M4>(out_ptr, v_out, vl);

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
        vl = SET_VECTOR_LENGTH<float, M8>(cnt);
        auto v_a = VECTOR_LOAD<float, M8>(in_a_ptr, vl);
        auto v_b = VECTOR_LOAD<float, M8>(in_b_ptr, vl);
        auto v_out = VECTOR_ADD<float, M8>(v_a, v_b, vl);
        VECTOR_STORE<float, M8>(out_ptr, v_out, vl);

        in_a_ptr += vl;
        in_b_ptr += vl;
        out_ptr += vl;
        cnt -= vl;
    }
}
