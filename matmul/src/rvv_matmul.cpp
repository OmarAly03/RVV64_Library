#include <riscv_vector.h>
#include <cstddef>
#include "rvv_defs.hpp"

using namespace std;

void matmul_e32m1(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j_cnt = N; j_cnt > 0; ) {
            size_t vl = SET_VECTOR_LENGTH<float, M1>(j_cnt);
            size_t j = N - j_cnt;
            
            auto acc = VECTOR_MOVE<float, M1>(0.0f, vl);
            
            for (size_t k = 0; k < K; k++) {
                auto a_elem = VECTOR_MOVE<float, M1>(A[i * K + k], vl);
                auto b_vec = VECTOR_LOAD<float, M1>(&B[k * N + j], vl);
                acc = VECTOR_FMACC<float, M1>(acc, a_elem, b_vec, vl);
            }
            
            VECTOR_STORE<float, M1>(&C[i * N + j], acc, vl);
            
            j_cnt -= vl;
        }
    }
}

void matmul_e32m2(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
	for (size_t i = 0; i < M; i++) {
		for (size_t j_cnt = N; j_cnt > 0; ) {
			size_t vl = SET_VECTOR_LENGTH<float, M2>(j_cnt);
			size_t j = N - j_cnt;
			
			auto acc = VECTOR_MOVE<float, M2>(0.0f, vl);
			
			for (size_t k = 0; k < K; k++) {
				auto a_elem = VECTOR_MOVE<float, M2>(A[i * K + k], vl);
				auto b_vec = VECTOR_LOAD<float, M2>(&B[k * N + j], vl);
				acc = VECTOR_FMACC<float, M2>(acc, a_elem, b_vec, vl);
			}
			
			VECTOR_STORE<float, M2>(&C[i * N + j], acc, vl);
			
			j_cnt -= vl;
		}
	}
}

void matmul_e32m4(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
	for (size_t i = 0; i < M; i++) {
		for (size_t j_cnt = N; j_cnt > 0; ) {
			size_t vl = SET_VECTOR_LENGTH<float, M4>(j_cnt);
			size_t j = N - j_cnt;
			
			auto acc = VECTOR_MOVE<float, M4>(0.0f, vl);
			
			for (size_t k = 0; k < K; k++) {
				auto a_elem = VECTOR_MOVE<float, M4>(A[i * K + k], vl);
				auto b_vec = VECTOR_LOAD<float, M4>(&B[k * N + j], vl);
				acc = VECTOR_FMACC<float, M4>(acc, a_elem, b_vec, vl);
			}
			
			VECTOR_STORE<float, M4>(&C[i * N + j], acc, vl);
			
			j_cnt -= vl;
		}
	}
}

void matmul_e32m8(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
	for (size_t i = 0; i < M; i++) {
		for (size_t j_cnt = N; j_cnt > 0; ) {
			size_t vl = SET_VECTOR_LENGTH<float, M8>(j_cnt);
			size_t j = N - j_cnt;
			
			auto acc = VECTOR_MOVE<float, M8>(0.0f, vl);
			
			for (size_t k = 0; k < K; k++) {
				auto a_elem = VECTOR_MOVE<float, M8>(A[i * K + k], vl);
				auto b_vec = VECTOR_LOAD<float, M8>(&B[k * N + j], vl);
				acc = VECTOR_FMACC<float, M8>(acc, a_elem, b_vec, vl);
			}
			
			VECTOR_STORE<float, M8>(&C[i * N + j], acc, vl);
			
			j_cnt -= vl;
		}
	}
}

void matmul_scalar(float* A, float* B, float* C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}