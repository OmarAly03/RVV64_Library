#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

// Vectorized matrix transpose (RVV)
void transpose_rvv(int32_t *B, int32_t *BT, int rows_B, int cols_B) {
    int vl;
    asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(cols_B));
    for (int i = 0; i < rows_B; i++) {
        for (int j = 0; j < cols_B; ) {
            asm volatile("vle32.v v0, (%0)" :: "r"(&B[i * cols_B + j]) : "v0");
            asm volatile("vsse32.v v0, (%0), %1" :: "r"(&BT[j * rows_B + i]), "r"(rows_B * sizeof(int32_t)) : "v0");
            j += vl;
        }
    }
}

// Sequential matrix multiplication (A * B = C)
void matmul_sequential(int32_t *A, int32_t *B, int32_t *C, int rows_A, int cols_A, int cols_B) {
    memset(C, 0, rows_A * cols_B * sizeof(int32_t));
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            int32_t sum = 0;
            for (int k = 0; k < cols_A; k++) {
                sum += A[i * cols_A + k] * B[k * cols_B + j];
            }
            C[i * cols_B + j] = sum;
        }
    }
}

// Vectorized matrix multiplication (A * BT = C)
void vector_mul_rvv(int32_t *A, int32_t *BT, int32_t *C, int rows_A, int cols_A, int cols_B) {
    memset(C, 0, rows_A * cols_B * sizeof(int32_t));
    size_t vlmax;
    asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vlmax) : "r"(cols_A));
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            int32_t sum = 0;
            int32_t remaining = cols_A;
            int32_t *a_ptr = &A[i * cols_A];
            int32_t *bt_ptr = &BT[j * cols_A];
            while (remaining > 0) {
                size_t vl;
                int32_t temp_sum;
                asm volatile(
                    "vsetvli %0, %2, e32, m1, ta, ma;"
                    "vle32.v v0, (%3);"
                    "vle32.v v4, (%4);"
                    "vmul.vv v8, v0, v4;"
                    "vmv.v.i v9, 0;"
                    "vredsum.vs v8, v8, v9;"
                    "vmv.x.s %1, v8"
                    : "=r"(vl), "=r"(temp_sum)
                    : "r"(remaining), "r"(a_ptr), "r"(bt_ptr)
                    : "v0", "v4", "v8", "v9"
                );
                sum += temp_sum;
                a_ptr += vl;
                bt_ptr += vl;
                remaining -= vl;
            }
            C[i * cols_B + j] = sum;
        }
    }
}

static inline uint64_t rdinstret(void) {
    uint64_t val;
    asm volatile("rdinstret %0" : "=r"(val));
    return val;
}

int int_eq(int32_t a, int32_t b) {
    return a == b;
}

int test_matmul(void) {
    printf("\n===== MATRIX MULTIPLICATION PERFORMANCE TEST =====\n");
    
    int rows_A = (1<<7), cols_A = (1<<7), cols_B = (1<<7);
    int iterations = 1;

    size_t size_A = rows_A * cols_A * sizeof(int32_t);
    size_t size_B = cols_A * cols_B * sizeof(int32_t);
    size_t size_BT = cols_A * cols_B * sizeof(int32_t);
    size_t size_C = rows_A * cols_B * sizeof(int32_t);
    int32_t *A = malloc(size_A);
    int32_t *B = malloc(size_B);
    int32_t *BT = malloc(size_BT);
    int32_t *C_seq = malloc(size_C);
    int32_t *C_vec = malloc(size_C);

    srand(42);
    for (size_t i = 0; i < rows_A * cols_A; i++) A[i] = rand() % 10;
    for (size_t i = 0; i < cols_A * cols_B; i++) B[i] = rand() % 10;

    // Sequential: A * B = C
    uint64_t start_seq = rdinstret();
    clock_t time_start_seq = clock();
    for (int i = 0; i < iterations; i++) {
        matmul_sequential(A, B, C_seq, rows_A, cols_A, cols_B);
    }
    uint64_t end_seq = rdinstret();
    clock_t time_end_seq = clock();
    double time_seq = (double)(time_end_seq - time_start_seq) / CLOCKS_PER_SEC;

    // Vectorized: Transpose B to BT, then A * BT = C, count instructions for multiply only
    clock_t time_start_vec = clock();
    transpose_rvv(B, BT, cols_A, cols_B); // Transpose once
    uint64_t start_vec = rdinstret();
    for (int i = 0; i < iterations; i++) {
        vector_mul_rvv(A, BT, C_vec, rows_A, cols_A, cols_B);
    }
    uint64_t end_vec = rdinstret();
    clock_t time_end_vec = clock();
    double time_vec = (double)(time_end_vec - time_start_vec) / CLOCKS_PER_SEC;

    // Verify correctness
    int mismatches = 0;
    for (size_t i = 0; i < rows_A * cols_B && mismatches < 5; i++) {
        if (!int_eq(C_seq[i], C_vec[i])) {
            printf("Matmul Mismatch at index %zu: sequential=%d, vector=%d\n", i, C_seq[i], C_vec[i]);
            mismatches++;
        }
    }

    printf("Matrix A: %dx%d, Matrix B: %dx%d, Matrix C: %dx%d\n", rows_A, cols_A, cols_A, cols_B, rows_A, cols_B);
    printf("Sequential Execution time (multiply, %d iterations): %.3f seconds\n", iterations, time_seq);
    printf("Sequential Instructions (multiply, %d iterations): %llu\n", iterations, end_seq - start_seq);
    printf("Vectorized Execution time (transpose+multiply, %d iterations): %.3f seconds\n", iterations, time_vec);
    printf("Vectorized Instructions (multiply only, %d iterations): %llu\n", iterations, end_vec - start_vec);
    printf("Speedup (Time): %.2fx\n", time_seq / time_vec);
    printf("Speedup (Instructions, multiply only): %.2fx\n", (double)(end_seq - start_seq) / (end_vec - start_vec));
    printf("Correctness: %s\n", mismatches == 0 ? "PASSED" : "FAILED");

    free(A);
    free(B);
    free(BT);
    free(C_seq);
    free(C_vec);
    return mismatches == 0;
}

int main(void) {
    printf("===================================\n");
    printf("MATRIX MULTIPLICATION PERFORMANCE TEST\n");
    printf("===================================\n");

    int pass_matmul = test_matmul();

    printf("\n===================================\n");
    printf("SUMMARY:\n");
    printf("Matmul Test: %s\n", pass_matmul ? "PASSED" : "FAILED");

    return 0;
}