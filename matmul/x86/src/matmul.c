#include <stddef.h>

// Generate the shared object file (.so) : gcc -shared -fPIC -O2 -o matmul.so matmul.c

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