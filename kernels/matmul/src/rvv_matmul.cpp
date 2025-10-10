#include <riscv_vector.h>
#include <cstddef>
#include "rvv_defs.hpp"

using namespace std;

void matmul_e32m1(int32_t *A, int32_t *B, int32_t *C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j_cnt = N; j_cnt > 0; ) {
            size_t vl = SET_VECTOR_LENGTH<int32_t, M1>(j_cnt);
            size_t j = N - j_cnt;
            
            auto acc = VECTOR_BROADCAST<int32_t, M1>(0, vl);
            
            for (size_t k = 0; k < K; k++) {
                auto a_elem = VECTOR_BROADCAST<int32_t, M1>(A[i * K + k], vl);
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
        for (size_t j_cnt = N; j_cnt > 0; ) {
            size_t vl = SET_VECTOR_LENGTH<int32_t, M2>(j_cnt);
            size_t j = N - j_cnt;
            
            auto acc = VECTOR_BROADCAST<int32_t, M2>(0, vl);
            
            for (size_t k = 0; k < K; k++) {
                auto a_elem = VECTOR_BROADCAST<int32_t, M2>(A[i * K + k], vl);
                auto b_vec = VECTOR_LOAD<int32_t, M2>(&B[k * N + j], vl);
                acc = VECTOR_FMACC<int32_t, M2>(acc, a_elem, b_vec, vl);
            }
            
            VECTOR_STORE<int32_t, M2>(&C[i * N + j], acc, vl);
            
            j_cnt -= vl;
        }
    }
}

void matmul_e32m4(int32_t *A, int32_t *B, int32_t *C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j_cnt = N; j_cnt > 0; ) {
            size_t vl = SET_VECTOR_LENGTH<int32_t, M4>(j_cnt);
            size_t j = N - j_cnt;
            
            auto acc = VECTOR_BROADCAST<int32_t, M4>(0, vl);
            
            for (size_t k = 0; k < K; k++) {
                auto a_elem = VECTOR_BROADCAST<int32_t, M4>(A[i * K + k], vl);
                auto b_vec = VECTOR_LOAD<int32_t, M4>(&B[k * N + j], vl);
                acc = VECTOR_FMACC<int32_t, M4>(acc, a_elem, b_vec, vl);
            }
            
            VECTOR_STORE<int32_t, M4>(&C[i * N + j], acc, vl);
            
            j_cnt -= vl;
        }
    }
}

void matmul_e32m8(int32_t *A, int32_t *B, int32_t *C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j_cnt = N; j_cnt > 0; ) {
            size_t vl = SET_VECTOR_LENGTH<int32_t, M8>(j_cnt);
            size_t j = N - j_cnt;
            
            auto acc = VECTOR_BROADCAST<int32_t, M8>(0, vl);
            
            for (size_t k = 0; k < K; k++) {
                auto a_elem = VECTOR_BROADCAST<int32_t, M8>(A[i * K + k], vl);
                auto b_vec = VECTOR_LOAD<int32_t, M8>(&B[k * N + j], vl);
                acc = VECTOR_FMACC<int32_t, M8>(acc, a_elem, b_vec, vl);
            }
            
            VECTOR_STORE<int32_t, M8>(&C[i * N + j], acc, vl);
            
            j_cnt -= vl;
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