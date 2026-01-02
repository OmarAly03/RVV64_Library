#ifndef DEFS_H
#define DEFS_H

#include <cstddef>

// --- Bias Add Versions---
void bias_add_scalar(const float* input, const float* bias, float* output,
                       size_t batch_size, size_t channels,
                       size_t height, size_t width);
					   
void bias_add_e32m1(const float* input, const float* bias, float* output,
					size_t channels, size_t channel_size);
void bias_add_e32m2(const float* input, const float* bias, float* output,
					size_t channels, size_t channel_size);
void bias_add_e32m4(const float* input, const float* bias, float* output,
					size_t channels, size_t channel_size);
void bias_add_e32m8(const float* input, const float* bias, float* output,
					size_t channels, size_t channel_size);

// --- Utils ---
void write_matrix_to_file(const char* filename, float* matrix, std::size_t rows, std::size_t cols);
void write_matrix_binary(const char* filename, float* matrix, std::size_t count);

#endif