#ifndef DEFS_H
#define DEFS_H

#include <cstddef>

void relu_e32m1(float* input, float* output, std::size_t size);
void relu_e32m2(float* input, float* output, std::size_t size);
void relu_e32m4(float* input, float* output, std::size_t size);
void relu_e32m8(float* input, float* output, std::size_t size);

void relu_scalar(float* input, float* output, std::size_t size);

void write_matrix_to_file(const char* filename, float* matrix, std::size_t rows, std::size_t cols);
void write_matrix_binary(const char* filename, float* matrix, std::size_t count);

#endif