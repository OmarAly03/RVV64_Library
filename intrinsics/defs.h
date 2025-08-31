#include <stdint.h>

void vector_mul_scalar(int32_t *A, int32_t *B, int32_t *C, int rows_A, int cols_A, int cols_B);
void transpose_scalar(int32_t *B, int32_t *BT, int rows_B, int cols_B);
void transpose_rvv(int32_t *B, int32_t *BT, int rows_B, int cols_B);
void vector_mul_rvv(int32_t *A, int32_t *BT, int32_t *C, int rows_A, int cols_A, int cols_B);
void vector_mat_vec_scalar(int32_t *A, int32_t *v, int32_t *C, int rows_A, int cols_A);
void vector_mat_vec_rvv(int32_t *A, int32_t *v, int32_t *C, int rows_A, int cols_A);
void vector_add_scalar(int32_t *u, int32_t *v, int32_t *w, int n);
void vector_add_rvv(int32_t *u, int32_t *v, int32_t *w, int n);