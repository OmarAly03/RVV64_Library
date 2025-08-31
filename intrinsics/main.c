#include <stdint.h>
#include <riscv_vector.h>
#include "defs.h"
#include <uart.h>


#define ROWS 3
#define COLS 4
#define VEC_LEN 4

int32_t A[ROWS * COLS] __attribute__ ((aligned (4))) = {
    0, 1, 2, 3,
    10, 11, 12, 13,
    20, 21, 22, 23
};

int32_t B[ROWS * COLS] __attribute__ ((aligned (4))) = {
    0, 1, 2, 3,
    10, 11, 12, 13,
    20, 21, 22, 23
};

int32_t BT[COLS * ROWS] __attribute__ ((aligned (4)));
int32_t C[ROWS * COLS] __attribute__ ((aligned (4)));
int32_t v[VEC_LEN] __attribute__ ((aligned (4))) = {1, 2, 3, 4};
int32_t Cv[ROWS] __attribute__ ((aligned (4)));
int32_t u[VEC_LEN] __attribute__ ((aligned (4))) = {1, 2, 3, 4};
int32_t w[VEC_LEN] __attribute__ ((aligned (4)));

int main(void) {
    uart_printf("Testing Vicuna Functions\n");

    // Test transpose
    transpose_scalar(B, BT, ROWS, COLS);
    uart_printf("Transposed matrix (scalar):\n");
    for (int i = 0; i < COLS; i++) {
        for (int j = 0; j < ROWS; j++) {
            uart_printf("%d ", BT[i * ROWS + j]);
        }
        uart_printf("\n");
    }

    for (int i = 0; i < COLS * ROWS; i++) BT[i] = 0; // Reset BT
    transpose_rvv(B, BT, ROWS, COLS);
    uart_printf("\nTransposed matrix (RVV):\n");
    for (int i = 0; i < COLS; i++) {
        for (int j = 0; j < ROWS; j++) {
            uart_printf("%d ", BT[i * ROWS + j]);
        }
        uart_printf("\n");
    }

    // Test matrix multiplication
    vector_mul_scalar(A, B, C, ROWS, COLS, COLS);
    uart_printf("\nMatrix multiplication (scalar):\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            uart_printf("%d ", C[i * COLS + j]);
        }
        uart_printf("\n");
    }

    for (int i = 0; i < ROWS * COLS; i++) C[i] = 0; // Reset C
    vector_mul_rvv(A, BT, C, ROWS, COLS, ROWS);
    uart_printf("\nMatrix multiplication (RVV):\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < ROWS; j++) {
            uart_printf("%d ", C[i * ROWS + j]);
        }
        uart_printf("\n");
    }

    // Test matrix-vector multiplication
    vector_mat_vec_scalar(A, v, Cv, ROWS, COLS);
    uart_printf("\nMatrix-vector multiplication (scalar):\n");
    for (int i = 0; i < ROWS; i++) {
        uart_printf("%d ", Cv[i]);
    }
    uart_printf("\n");

    for (int i = 0; i < ROWS; i++) Cv[i] = 0; // Reset Cv
    vector_mat_vec_rvv(A, v, Cv, ROWS, COLS);
    uart_printf("\nMatrix-vector multiplication (RVV):\n");
    for (int i = 0; i < ROWS; i++) {
        uart_printf("%d ", Cv[i]);
    }
    uart_printf("\n");

    // Test vector addition
    vector_add_scalar(u, v, w, VEC_LEN);
    uart_printf("\nVector addition (scalar):\n");
    for (int i = 0; i < VEC_LEN; i++) {
        uart_printf("%d ", w[i]);
    }
    uart_printf("\n");

    for (int i = 0; i < VEC_LEN; i++) w[i] = 0; // Reset w
    vector_add_rvv(u, v, w, VEC_LEN);
    uart_printf("\nVector addition (RVV):\n");
    for (int i = 0; i < VEC_LEN; i++) {
        uart_printf("%d ", w[i]);
    }
    uart_printf("\n");

    asm volatile("1: j 1b"); // Infinite loop to halt
    return 0;
}