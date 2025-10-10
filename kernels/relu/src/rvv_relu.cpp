#include <cstddef>
#include <riscv_vector.h>
#include "defs_relu.h"
#include "rvv_defs.hpp"

using namespace std;

void relu_e32m1(int32_t* input, int32_t* output, size_t size) {
	int32_t* in_ptr = input;
	int32_t* out_ptr = output;
	
	for (size_t cnt = size; cnt > 0; ) {
		size_t vl = SET_VECTOR_LENGTH<int32_t, M1>(cnt);
		vint32m1_t v_input = VECTOR_LOAD<int32_t, M1>(in_ptr, vl);
		vint32m1_t v_zero = VECTOR_MOVE<int32_t, M1>(0, vl);
		vint32m1_t v_result = VECTOR_MAX<int32_t, M1>(v_input, v_zero, vl);
		VECTOR_STORE<int32_t, M1>(out_ptr, v_result, vl);
		
		cnt -= vl;
		in_ptr += vl;
		out_ptr += vl;
	}
}

void relu_e32m2(int32_t* input, int32_t* output, size_t size) {
	int32_t* in_ptr = input;
	int32_t* out_ptr = output;
	
	for (size_t cnt = size; cnt > 0; ) {
		size_t vl = __riscv_vsetvl_e32m2(cnt);
		vint32m2_t v_input = __riscv_vle32_v_i32m2(in_ptr, vl);
		vint32m2_t v_zero = __riscv_vmv_v_x_i32m2(0, vl);
		vint32m2_t v_result = __riscv_vmax_vv_i32m2(v_input, v_zero, vl);
		__riscv_vse32_v_i32m2(out_ptr, v_result, vl);
		
		cnt -= vl;
		in_ptr += vl;
		out_ptr += vl;
	}
}

void relu_e32m4(int32_t* input, int32_t* output, size_t size) {
	int32_t* in_ptr = input;
	int32_t* out_ptr = output;
	
	for (size_t cnt = size; cnt > 0; ) {
		size_t vl = __riscv_vsetvl_e32m4(cnt);
		vint32m4_t v_input = __riscv_vle32_v_i32m4(in_ptr, vl);
		vint32m4_t v_zero = __riscv_vmv_v_x_i32m4(0, vl);
		vint32m4_t v_result = __riscv_vmax_vv_i32m4(v_input, v_zero, vl);
		__riscv_vse32_v_i32m4(out_ptr, v_result, vl);
		
		cnt -= vl;
		in_ptr += vl;
		out_ptr += vl;
	}
}

void relu_e32m8(int32_t* input, int32_t* output, size_t size) {
	int32_t* in_ptr = input;
	int32_t* out_ptr = output;
	
	for (size_t cnt = size; cnt > 0; ) {
		size_t vl = __riscv_vsetvl_e32m8(cnt);
		vint32m8_t v_input = __riscv_vle32_v_i32m8(in_ptr, vl);
		vint32m8_t v_zero = __riscv_vmv_v_x_i32m8(0, vl);
		vint32m8_t v_result = __riscv_vmax_vv_i32m8(v_input, v_zero, vl);
		__riscv_vse32_v_i32m8(out_ptr, v_result, vl);
		
		cnt -= vl;
		in_ptr += vl;
		out_ptr += vl;
	}
}

void relu_scalar(int32_t* input, int32_t* output, size_t size) {
    for (size_t i = 0; i < size; i++) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}