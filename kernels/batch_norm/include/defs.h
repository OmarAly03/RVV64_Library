#ifndef DEFS_H
#define DEFS_H

#include <cstddef>

// BatchNorm functions - all take (input, output, scale, bias, mean, variance, N, C, H, W, epsilon, [tile])
void batch_norm_scalar(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon);

void batch_norm_tiled_scalar(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon);

void batch_norm_e32m1(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon);
void batch_norm_e32m2(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon);
void batch_norm_e32m4(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon);
void batch_norm_e32m8(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon);

void batch_norm_tiled_e32m1(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon);
void batch_norm_tiled_e32m2(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon);
void batch_norm_tiled_e32m4(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon);
void batch_norm_tiled_e32m8(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon);

// Utility functions
void write_matrix_to_file(const char* filename, float* matrix, std::size_t rows, std::size_t cols);
void write_matrix_binary(const char* filename, float* matrix, std::size_t count);

#endif