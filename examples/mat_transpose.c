#include "../lib/defs.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

void transpose_seq(int32_t *B, int32_t *BT, int rows, int cols) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            BT[j * rows + i] = B[i * cols + j];
}


double timediff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
}

void fill_matrix(int32_t *M, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        M[i] = rand() % 100; 
}

void run_test(int rows, int cols) {
    int32_t *B = malloc(sizeof(int32_t) * rows * cols);
    int32_t *BT_seq = malloc(sizeof(int32_t) * rows * cols);
    int32_t *BT_rvv = malloc(sizeof(int32_t) * rows * cols);

    fill_matrix(B, rows, cols);

    struct timespec start, end;
    double time_seq, time_rvv;

    clock_gettime(CLOCK_MONOTONIC, &start);
    transpose_seq(B, BT_seq, rows, cols);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_seq = timediff(start, end);

    clock_gettime(CLOCK_MONOTONIC, &start);
    transpose_rvv(B, BT_rvv, rows, cols);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_rvv = timediff(start, end);

    // Validate correctness
    int correct = 1;
    for (int i = 0; i < rows * cols; ++i) {
        if (BT_seq[i] != BT_rvv[i]) {
            correct = 0;
            break;
        }
    }

    printf("Size %dx%d | Seq: %.6f s | RVV: %.6f s | Speedup: %.2fx | %s\n",
        rows, cols, time_seq, time_rvv, time_seq / time_rvv, correct ? "Correct" : "Mismatch");

    free(B);
    free(BT_seq);
    free(BT_rvv);
}

int main() {
    srand(time(NULL));

    // Test small to large matrices
    int sizes[][2] = {
        {2048, 2048},
        {4096, 4096},
        {16384, 16384}
    };

    int num_tests = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_tests; i++) {
        int rows = sizes[i][0];
        int cols = sizes[i][1];
        run_test(rows, cols);
    }

    return 0;
}
