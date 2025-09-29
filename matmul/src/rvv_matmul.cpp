#include <riscv_vector.h>
#include <cstddef>

using namespace std;

void matmul_e32m1(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j += __riscv_vsetvlmax_e32m1()) {
            size_t vl = __riscv_vsetvl_e32m1(N - j);
            vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            
            for (size_t k = 0; k < K; k++) {
                vfloat32m1_t a_elem = __riscv_vfmv_v_f_f32m1(A[i * K + k], vl);
                vfloat32m1_t b_vec = __riscv_vle32_v_f32m1(&B[k * N + j], vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, a_elem, b_vec, vl);
            }
            
            __riscv_vse32_v_f32m1(&C[i * N + j], acc, vl);
        }
    }
}

void matmul_e32m2(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j += __riscv_vsetvlmax_e32m2()) {
            size_t vl = __riscv_vsetvl_e32m2(N - j);
            vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            
            for (size_t k = 0; k < K; k++) {
                vfloat32m2_t a_elem = __riscv_vfmv_v_f_f32m2(A[i * K + k], vl);
                vfloat32m2_t b_vec = __riscv_vle32_v_f32m2(&B[k * N + j], vl);
                acc = __riscv_vfmacc_vv_f32m2(acc, a_elem, b_vec, vl);
            }
            
            __riscv_vse32_v_f32m2(&C[i * N + j], acc, vl);
        }
    }
}

void matmul_e32m4(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j += __riscv_vsetvlmax_e32m4()) {
            size_t vl = __riscv_vsetvl_e32m4(N - j);
            vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            
            for (size_t k = 0; k < K; k++) {
                vfloat32m4_t a_elem = __riscv_vfmv_v_f_f32m4(A[i * K + k], vl);
                vfloat32m4_t b_vec = __riscv_vle32_v_f32m4(&B[k * N + j], vl);
                acc = __riscv_vfmacc_vv_f32m4(acc, a_elem, b_vec, vl);
            }
            
            __riscv_vse32_v_f32m4(&C[i * N + j], acc, vl);
        }
    }
}

void matmul_e32m8(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j += __riscv_vsetvlmax_e32m8()) {
            size_t vl = __riscv_vsetvl_e32m8(N - j);
            vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            
            for (size_t k = 0; k < K; k++) {
                vfloat32m8_t a_elem = __riscv_vfmv_v_f_f32m8(A[i * K + k], vl);
                vfloat32m8_t b_vec = __riscv_vle32_v_f32m8(&B[k * N + j], vl);
                acc = __riscv_vfmacc_vv_f32m8(acc, a_elem, b_vec, vl);
            }
            
            __riscv_vse32_v_f32m8(&C[i * N + j], acc, vl);
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