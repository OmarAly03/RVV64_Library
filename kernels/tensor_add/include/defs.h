#ifndef DEFS_H
#define DEFS_H

#include <cstddef>

// --- Tensor Add ---
void tensor_add_scalar(const float* input_a, const float* input_b, float* output,
                           size_t size);
void tensor_add_e32m1(const float* input_a, const float* input_b, float* output,
                           size_t size);
void tensor_add_e32m2(const float* input_a, const float* input_b, float* output,
                           size_t size);
void tensor_add_e32m4(const float* input_a, const float* input_b, float* output,
                           size_t size);
void tensor_add_e32m8(const float* input_a, const float* input_b, float* output,
                           size_t size);


// --- Utils ---
void write_matrix_to_file(const char* filename, float* matrix, std::size_t rows, std::size_t cols);
void write_matrix_binary(const char* filename, float* matrix, std::size_t count);

#endif