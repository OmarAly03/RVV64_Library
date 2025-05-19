#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "../lib/defs.h"

int main() {
    clock_t start, end;
    double cpu_time_used;

    // 4x4 matrices: A is 4x4, B is 4x4, C is 4x4
    int rows_A = 4, cols_A = 4, cols_B = 4;

    int32_t *A = (int32_t *)malloc(rows_A * cols_A * sizeof(int32_t));
    int32_t *B = (int32_t *)malloc(cols_A * cols_B * sizeof(int32_t));
    int32_t *BT = (int32_t *)malloc(cols_B * cols_A * sizeof(int32_t));
    int32_t *C_scalar = (int32_t *)malloc(rows_A * cols_B * sizeof(int32_t));
    int32_t *C_rvv = (int32_t *)malloc(rows_A * cols_B * sizeof(int32_t));

    if (!A || !B || !BT || !C_scalar || !C_rvv) {
        fprintf(stderr, "Memory allocation failed\n");
        free(A); free(B); free(BT); free(C_scalar); free(C_rvv);
        return 1;
    }

    // Initialize matrices with test values
    int32_t temp_A[] = {1, 2, 3, 4, 
                        5, 6, 7, 8, 
                        1, 2, 3, 4, 
                        5, 6, 7, 8};

    int32_t temp_B[] = {2, 0, 0, 0, 
                        0, 2, 0, 0,
                        0, 0, 2, 0,
                        0, 0, 0, 2};
    memcpy(A, temp_A, rows_A * cols_A * sizeof(int32_t));
    memcpy(B, temp_B, cols_A * cols_B * sizeof(int32_t));

    // Print input matrices
    printf("Matrix A:\n");
    for(int i = 0; i < rows_A; i++) {
        for(int j = 0; j < cols_A; j++) {
            printf("%2d ", A[i * cols_A + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    printf("Matrix B:\n");
    for(int i = 0; i < cols_A; i++) {
        for(int j = 0; j < cols_B; j++) {
            printf("%2d ", B[i * cols_B + j]);
        }
        printf("\n");
    }
    printf("\n");

    // Transpose B
    start = clock();
    transpose_rvv(B, BT, cols_A, cols_B);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Transpose time: %f seconds\n", cpu_time_used);

    // Scalar vector multiplication
    start = clock();
    vector_mul_scalar(A, B, C_scalar, rows_A, cols_A, cols_B);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Scalar execution time: %f seconds\n", cpu_time_used);

    // RVV vector multiplication
    start = clock();
    vector_mul_rvv(A, BT, C_rvv, rows_A, cols_A, cols_B);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("RVV execution time: %f seconds\n", cpu_time_used);

    // Verify results - print full matrices
    printf("\nScalar Result Matrix:\n");
    for(int i = 0; i < rows_A; i++) {
        for(int j = 0; j < cols_B; j++) {
            printf("%2d ", C_scalar[i * cols_B + j]);
        }
        printf("\n");
    }

    printf("\nRVV Result Matrix:\n");
    for(int i = 0; i < rows_A; i++) {
        for(int j = 0; j < cols_B; j++) {
            printf("%2d ", C_rvv[i * cols_B + j]);
        }
        printf("\n");
    }

    // Check correctness
    int correct = 1;
    for (int i = 0; i < rows_A * cols_B; i++) {
        if (C_scalar[i] != C_rvv[i]) {
            printf("Mismatch at position [%d]: scalar=%d, rvv=%d\n", 
                   i, C_scalar[i], C_rvv[i]);
            correct = 0;
        }
    }
    printf("\nResults %s\n", correct ? "match" : "do not match");

    free(A);
    free(B);
    free(BT);
    free(C_scalar);
    free(C_rvv);

    return 0;
}