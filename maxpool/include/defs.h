#ifndef DEFS_H
#define DEFS_H

#include <cstddef>
#include <cstdint> 

// Calculate output dimension helper macro, now with ceil_mode support
#define CALC_OUT_DIM(in_dim, kernel, stride, ceil_mode) \
    (ceil_mode ? ((in_dim) + (stride) - (kernel) + (stride) - 1) / (stride) : ((in_dim) - (kernel)) / (stride) + 1)

// Function prototypes for MaxPool kernels
void maxpool_scalar(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode);
void maxpool_e32m1(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode);
void maxpool_e32m2(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode);
void maxpool_e32m4(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode);
void maxpool_e32m8(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode);

// Utility function prototypes
void read_tensor_binary(const char* filename, float* tensor, size_t count);
void write_tensor_binary_float(const char* filename, const float* tensor, size_t count);
void write_tensor_binary_int64(const char* filename, const int64_t* tensor, size_t count);

#endif // DEFS_H
