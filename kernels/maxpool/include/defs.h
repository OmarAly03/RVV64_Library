#ifndef DEFS_H
#define DEFS_H

#include <cstddef>
#include <cstdint>

void maxpool_scalar(const float* input, float* output,
	int batch, int channels,
	int in_h, int in_w,
	int k_h, int k_w,
	int stride_h, int stride_w,
	int pad_h, int pad_w);

void maxpool_e32m1(const float* input, float* output,
	int batch, int channels,
	int in_h, int in_w,
	int k_h, int k_w,
	int stride_h, int stride_w,
	int pad_h, int pad_w);

void maxpool_e32m2(const float* input, float* output,
	int batch, int channels,
	int in_h, int in_w,
	int k_h, int k_w,
	int stride_h, int stride_w,
	int pad_h, int pad_w);

void maxpool_e32m4(const float* input, float* output,
	int batch, int channels,
	int in_h, int in_w,
	int k_h, int k_w,
	int stride_h, int stride_w,
	int pad_h, int pad_w);

void maxpool_e32m8(const float* input, float* output,
	int batch, int channels,
	int in_h, int in_w,
	int k_h, int k_w,
	int stride_h, int stride_w,
	int pad_h, int pad_w);

void write_matrix_to_file(const char* filename, float* matrix, std::size_t rows, std::size_t cols);
void write_matrix_binary(const char* filename, float* matrix, std::size_t count);

#endif // DEFS_H
