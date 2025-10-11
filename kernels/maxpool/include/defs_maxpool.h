#ifndef DEFS_MAXPOOL_H
#define DEFS_MAXPOOL_H

#include <cstddef>
#include <cstdint>

static inline uint32_t read_mcycle(void) {
    uint32_t cycles;
    asm volatile("csrr %0, mcycle" : "=r"(cycles));
    return cycles;
}

// Calculate output dimension helper macro
#define CALC_OUT_DIM(in_dim, kernel, stride, ceil_mode) \
    (ceil_mode ? ((in_dim) + (stride) - (kernel) + (stride) - 1) / (stride) : ((in_dim) - (kernel)) / (stride) + 1)

// Function prototypes for MaxPool kernels
void maxpool_scalar(const int32_t* X, int32_t* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode);
void maxpool_e32m1(const int32_t* X, int32_t* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode);
void maxpool_e32m2(const int32_t* X, int32_t* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode);
void maxpool_e32m4(const int32_t* X, int32_t* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode);
void maxpool_e32m8(const int32_t* X, int32_t* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode);

// Utility function prototypes
void read_tensor_binary(const char* filename, float* tensor, size_t count);
void write_tensor_binary_float(const char* filename, const float* tensor, size_t count);
void write_tensor_binary_int64(const char* filename, const int64_t* tensor, size_t count);

#endif // DEFS_MAXPOOL_H
