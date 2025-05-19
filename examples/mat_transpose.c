#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "../lib/defs.h"

void print_matrix(int32_t *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%4d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    // Define matrix dimensions
    int rows_B = 3;
    int cols_B = 4;

    // Allocate memory for input matrix B and output matrix BT
    int32_t *B = (int32_t *)malloc(rows_B * cols_B * sizeof(int32_t));
    int32_t *BT = (int32_t *)malloc(cols_B * rows_B * sizeof(int32_t));

    // Initialize matrix B with sample values
    int32_t value = 1;
    for (int i = 0; i < rows_B; i++) {
        for (int j = 0; j < cols_B; j++) {
            B[i * cols_B + j] = value++;
        }
    }

    // Print original matrix B
    printf("Original Matrix B (%d x %d):\n", rows_B, cols_B);
    print_matrix(B, rows_B, cols_B);

    // Call the transpose function
    transpose_rvv(B, BT, rows_B, cols_B);

    // Print transposed matrix BT
    printf("\nTransposed Matrix BT (%d x %d):\n", cols_B, rows_B);
    print_matrix(BT, cols_B, rows_B);

    // Verify the transposition
    int correct = 1;
    for (int i = 0; i < rows_B; i++) {
        for (int j = 0; j < cols_B; j++) {
            if (B[i * cols_B + j] != BT[j * rows_B + i]) {
                correct = 0;
                break;
            }
        }
    }
    printf("\nTransposition %s\n", correct ? "Correct" : "Incorrect");

    // Free allocated memory
    free(B);
    free(BT);

    return 0;
}