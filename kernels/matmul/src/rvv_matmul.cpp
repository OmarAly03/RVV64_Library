#include <riscv_vector.h>
#include <cstddef>
#include "rvv_defs.hpp"

using namespace std;

void matmul_e32m1(int32_t *A, int32_t *B, int32_t *C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j_cnt = N; j_cnt > 0; ) {
            size_t vl = SET_VECTOR_LENGTH<int32_t, M1>(j_cnt);
            size_t j = N - j_cnt;
            
            auto acc = VECTOR_MOVE<int32_t, M1>(0, vl);
            
            for (size_t k = 0; k < K; k++) {
                auto a_elem = VECTOR_MOVE<int32_t, M1>(A[i * K + k], vl);
                auto b_vec = VECTOR_LOAD<int32_t, M1>(&B[k * N + j], vl);
                acc = VECTOR_FMACC<int32_t, M1>(acc, a_elem, b_vec, vl);
            }
            
            VECTOR_STORE<int32_t, M1>(&C[i * N + j], acc, vl);
            
            j_cnt -= vl;
        }
    }
}

void matmul_e32m2(int32_t *A, int32_t *B, int32_t *C, size_t M, size_t N, size_t K) {
	for (size_t i = 0; i < M; i++) {
		for (size_t j = 0; j < N; j += __riscv_vsetvlmax_e32m2()) {
			size_t vl = __riscv_vsetvl_e32m2(N - j);
			vint32m2_t acc = __riscv_vmv_v_x_i32m2(0, vl);
			
			for (size_t k = 0; k < K; k++) {
				vint32m2_t a_elem = __riscv_vmv_v_x_i32m2(A[i * K + k], vl);
				vint32m2_t b_vec = __riscv_vle32_v_i32m2(&B[k * N + j], vl);
				acc = __riscv_vmacc_vv_i32m2(acc, a_elem, b_vec, vl);
			}
			
			__riscv_vse32_v_i32m2(&C[i * N + j], acc, vl);
		}
	}
}

void matmul_e32m4(int32_t *A, int32_t *B, int32_t *C, size_t M, size_t N, size_t K) {
	for (size_t i = 0; i < M; i++) {
		for (size_t j = 0; j < N; j += __riscv_vsetvlmax_e32m4()) {
			size_t vl = __riscv_vsetvl_e32m4(N - j);
			vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);
			
			for (size_t k = 0; k < K; k++) {
				vint32m4_t a_elem = __riscv_vmv_v_x_i32m4(A[i * K + k], vl);
				vint32m4_t b_vec = __riscv_vle32_v_i32m4(&B[k * N + j], vl);
				acc = __riscv_vmacc_vv_i32m4(acc, a_elem, b_vec, vl);
			}
			
			__riscv_vse32_v_i32m4(&C[i * N + j], acc, vl);
		}
	}
}

void matmul_e32m8(int32_t *A, int32_t *B, int32_t *C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j += __riscv_vsetvlmax_e32m8()) {
            size_t vl = __riscv_vsetvl_e32m8(N - j);
            vint32m8_t acc = __riscv_vmv_v_x_i32m8(0, vl);  
            
            for (size_t k = 0; k < K; k++) {
                vint32m8_t a_elem = __riscv_vmv_v_x_i32m8(A[i * K + k], vl);  
                vint32m8_t b_vec = __riscv_vle32_v_i32m8(&B[k * N + j], vl);  
                acc = __riscv_vmacc_vv_i32m8(acc, a_elem, b_vec, vl);  
            }
            
            __riscv_vse32_v_i32m8(&C[i * N + j], acc, vl);  
        }
    }
}

void matmul_scalar(int32_t* A, int32_t* B, int32_t* C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            int32_t sum = 0;
            for (size_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}