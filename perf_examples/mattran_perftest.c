#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

// Sequential matrix transpose
void transpose_sequential(int32_t *B, int32_t *BT, int rows_B, int cols_B) {
    for (int i = 0; i < rows_B; i++) {
        for (int j = 0; j < cols_B; j++) {
            BT[j * rows_B + i] = B[i * cols_B + j];
        }
    }
}

// Vectorized matrix transpose (RVV)
void transpose_rvv(int32_t *B, int32_t *BT, int rows_B, int cols_B) {
    int vl;
    // Set max vector length once
    asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(cols_B));
    for (int i = 0; i < rows_B; i++) {
        for (int j = 0; j < cols_B; ) {
            asm volatile("vle32.v v0, (%0)" :: "r"(&B[i * cols_B + j]) : "v0");
            asm volatile("vsse32.v v0, (%0), %1" :: "r"(&BT[j * rows_B + i]), "r"(rows_B * sizeof(int32_t)) : "v0");
            j += vl;
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

int test_transpose(void) {
    printf("\n===== MATRIX TRANSPOSE PERFORMANCE TEST =====\n");
    
    int rows_B = (1<<11), cols_B = (1<<11); // Large matrix for RVV benefits
    int iterations = 1; // Match ReLU iterations

    size_t size = rows_B * cols_B * sizeof(int32_t);
    int32_t *B = malloc(size);
    int32_t *BT_seq = malloc(size);
    int32_t *BT_vec = malloc(size);

    srand(42);
    for (size_t i = 0; i < rows_B * cols_B; i++) {
        B[i] = rand() % 100;
    }

    // Sequential
    uint64_t start_seq = rdinstret();
    clock_t time_start_seq = clock();
    for (int i = 0; i < iterations; i++) {
        transpose_sequential(B, BT_seq, rows_B, cols_B);
    }
    uint64_t end_seq = rdinstret();
    clock_t time_end_seq = clock();
    double time_seq = (double)(time_end_seq - time_start_seq) / CLOCKS_PER_SEC;

    // Vectorized
    uint64_t start_vec = rdinstret();
    clock_t time_start_vec = clock();
    for (int i = 0; i < iterations; i++) {
        transpose_rvv(B, BT_vec, rows_B, cols_B);
    }
    uint64_t end_vec = rdinstret();
    clock_t time_end_vec = clock();
    double time_vec = (double)(time_end_vec - time_start_vec) / CLOCKS_PER_SEC;

    // Verify correctness
    int mismatches = 0;
    for (size_t i = 0; i < rows_B * cols_B && mismatches < 5; i++) {
        if (!int_eq(BT_seq[i], BT_vec[i])) {
            printf("Transpose Mismatch at index %zu: sequential=%d, vector=%d\n", i, BT_seq[i], BT_vec[i]);
            mismatches++;
        }
    }

    printf("Input Matrix: %dx%d\n", rows_B, cols_B);
    printf("Sequential Execution time (%d iterations): %.3f seconds\n", iterations, time_seq);
    printf("Sequential Instructions (%d iterations): %llu\n", iterations, end_seq - start_seq);
    printf("Vectorized Execution time (%d iterations): %.3f seconds\n", iterations, time_vec);
    printf("Vectorized Instructions (%d iterations): %llu\n", iterations, end_vec - start_vec);
    printf("Speedup (Time): %.2fx\n", time_seq / time_vec);
    printf("Speedup (Instructions): %.2fx\n", (double)(end_seq - start_seq) / (end_vec - start_vec));
    printf("Correctness: %s\n", mismatches == 0 ? "PASSED" : "FAILED");

    free(B);
    free(BT_seq);
    free(BT_vec);
    return mismatches == 0;
}

int main(void) {
    printf("===================================\n");
    printf("MATRIX TRANSPOSE PERFORMANCE TEST\n");
    printf("===================================\n");

    int pass_transpose = test_transpose();

    printf("\n===================================\n");
    printf("SUMMARY:\n");
    printf("Transpose Test: %s\n", pass_transpose ? "PASSED" : "FAILED");

    return 0;
}