#include <stdlib.h>
#include <stdint.h>

void train_step_1v(double *x, double *y, size_t n, double *w, double *b, double lr);

void train_step_2v(double *x1, double *x2, double *y, size_t n, double *w1, double *w2, double *b, double lr);

void train_step_3v(double *x1, double *x2, double *x3, double *y, size_t n, double *w1, double *w2, double *w3, double *b, double lr);

double calculate_mse_1v(double *x, double *y, size_t n, double w, double b);

double calculate_mse_2v(double *x1, double *x2, double *y, size_t n, double w1, double w2, double b);

double calculate_mse_3v(double *x1, double *x2, double *x3, double *y, size_t n, double w1, double w2, double w3, double b);

void vector_mul_scalar(int32_t *A, int32_t *B, int32_t *C, int rows_A, int cols_A, int cols_B);
void transpose_rvv(int32_t *B, int32_t *BT, int rows_B, int cols_B);
void vector_mul_rvv(int32_t *A, int32_t *BT, int32_t *C, int rows_A, int cols_A, int cols_B);
void vector_add_rvv(int32_t *u, int32_t *v, int32_t *w, int n);
void vector_mat_vec_rvv(int32_t *A, int32_t *v, int32_t *C, int rows_A, int cols_A);

void relu_activation(float *input, int h, int w, int c, float *output);
void conv_layer(float *input, float *weights, float *biases,
    int in_h, int in_w, int in_c,
    int num_filters, int filter_size,
    int stride, int padding,
    float *output);