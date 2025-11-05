#ifndef DEFS_H
#define DEFS_H

#include <cstddef>

// --- Dense (GEMM) ---
void dense_scalar(const float* input, const float* weights, const float* bias,
	float* output, size_t in_features, size_t out_features);

void dense_e32m1(const float* input, const float* weights, const float* bias,
   float* output, size_t in_features, size_t out_features);

void dense_e32m2(const float* input, const float* weights, const float* bias,
   float* output, size_t in_features, size_t out_features);

void dense_e32m4(const float* input, const float* weights, const float* bias,
   float* output, size_t in_features, size_t out_features);

void dense_e32m8(const float* input, const float* weights, const float* bias,
   float* output, size_t in_features, size_t out_features);


// --- Utils ---
void write_matrix_to_file(const char* filename, float* matrix, std::size_t rows, std::size_t cols);
void write_matrix_binary(const char* filename, float* matrix, std::size_t count);

#endif