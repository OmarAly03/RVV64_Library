#include <stdlib.h>

void train_step_1v(double *x, double *y, size_t n, double *w, double *b, double lr);

void train_step_2v(double *x1, double *x2, double *y, size_t n, double *w1, double *w2, double *b, double lr);

void train_step_3v(double *x1, double *x2, double *x3, double *y, size_t n, double *w1, double *w2, double *w3, double *b, double lr);

double calculate_mse_1v(double *x, double *y, size_t n, double w, double b);

double calculate_mse_2v(double *x1, double *x2, double *y, size_t n, double w1, double w2, double b);

double calculate_mse_3v(double *x1, double *x2, double *x3, double *y, size_t n, double w1, double w2, double w3, double b);

