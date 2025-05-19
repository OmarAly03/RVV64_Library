#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "../lib/defs.h"

int main() {
    int rows_A = 2, cols_A = 3;
    int32_t A[6] = {1, 2, 3, 4, 5, 6};      // 2x3 matrix
    int32_t v[3] = {7, 8, 9};               // 3x1 vector
    int32_t C[2];                           // 2x1 result
    vector_mat_vec_rvv(A, v, C, rows_A, cols_A);
    printf("Result: %d %d\n", C[0], C[1]);
    return 0;
}