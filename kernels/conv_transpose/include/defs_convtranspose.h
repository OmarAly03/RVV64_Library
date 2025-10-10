#ifndef DEFS_H
#define DEFS_H

#include <cstddef>
#include <cstdint>

static inline uint32_t read_mcycle(void) {
    uint32_t cycles;
    asm volatile("csrr %0, mcycle" : "=r"(cycles));
    return cycles;
}

void conv_transpose_2d_e32m1(
    const int32_t* input, const int32_t* kernel, int32_t* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w);

void conv_transpose_2d_e32m2(
	const int32_t* input, const int32_t* kernel, int32_t* output,
	int batch_size, int in_channels, int out_channels,
	int input_h, int input_w, int kernel_h, int kernel_w,
	int stride_h, int stride_w, int pad_h, int pad_w);

void conv_transpose_2d_e32m4(
	const int32_t* input, const int32_t* kernel, int32_t* output,
	int batch_size, int in_channels, int out_channels,
	int input_h, int input_w, int kernel_h, int kernel_w,
	int stride_h, int stride_w, int pad_h, int pad_w);

void conv_transpose_2d_e32m8(
	const int32_t* input, const int32_t* kernel, int32_t* output,
	int batch_size, int in_channels, int out_channels,
	int input_h, int input_w, int kernel_h, int kernel_w,
	int stride_h, int stride_w, int pad_h, int pad_w);

void conv_transpose_2d_scalar(
	const int32_t* input, const int32_t* kernel, int32_t* output,
	int batch_size, int in_channels, int out_channels,
	int input_h, int input_w, int kernel_h, int kernel_w,
	int stride_h, int stride_w, int pad_h, int pad_w);

#endif
