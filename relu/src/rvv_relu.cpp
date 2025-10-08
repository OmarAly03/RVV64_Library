#include <cstddef>
#include <riscv_vector.h>
#include "rvv_defs.hpp"

using namespace std;

void relu_e32m1(float* input, float* output, size_t size) {
    float* in_ptr = input;
    float* out_ptr = output;
    
    for (size_t cnt = size; cnt > 0; ) {
        size_t vl = SET_VECTOR_LENGTH<float, M1>(cnt);
        auto v_input = VECTOR_LOAD<float, M1>(in_ptr, vl);
        auto v_zero = VECTOR_MOVE<float, M1>(0.0f, vl);
        auto v_result = VECTOR_MAX<float, M1>(v_input, v_zero, vl);
        VECTOR_STORE<float, M1>(out_ptr, v_result, vl);
        
        cnt -= vl;
        in_ptr += vl;
        out_ptr += vl;
    }
}

void relu_e32m2(float* input, float* output, size_t size) {
	float* in_ptr = input;
	float* out_ptr = output;
	
	for (size_t cnt = size; cnt > 0; ) {
		size_t vl = SET_VECTOR_LENGTH<float, M2>(cnt);
		auto v_input = VECTOR_LOAD<float, M2>(in_ptr, vl);
		auto v_zero = VECTOR_MOVE<float, M2>(0.0f, vl);
		auto v_result = VECTOR_MAX<float, M2>(v_input, v_zero, vl);
		VECTOR_STORE<float, M2>(out_ptr, v_result, vl);
		
		cnt -= vl;
		in_ptr += vl;
		out_ptr += vl;
	}
}

void relu_e32m4(float* input, float* output, size_t size) {
	float* in_ptr = input;
	float* out_ptr = output;
	
	for (size_t cnt = size; cnt > 0; ) {
		size_t vl = SET_VECTOR_LENGTH<float, M4>(cnt);
		auto v_input = VECTOR_LOAD<float, M4>(in_ptr, vl);
		auto v_zero = VECTOR_MOVE<float, M4>(0.0f, vl);
		auto v_result = VECTOR_MAX<float, M4>(v_input, v_zero, vl);
		VECTOR_STORE<float, M4>(out_ptr, v_result, vl);
		
		cnt -= vl;
		in_ptr += vl;
		out_ptr += vl;
	}
}

void relu_e32m8(float* input, float* output, size_t size) {
	float* in_ptr = input;
	float* out_ptr = output;
	
	for (size_t cnt = size; cnt > 0; ) {
		size_t vl = SET_VECTOR_LENGTH<float, M8>(cnt);
		auto v_input = VECTOR_LOAD<float, M8>(in_ptr, vl);
		auto v_zero = VECTOR_MOVE<float, M8>(0.0f, vl);
		auto v_result = VECTOR_MAX<float, M8>(v_input, v_zero, vl);
		VECTOR_STORE<float, M8>(out_ptr, v_result, vl);
		
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