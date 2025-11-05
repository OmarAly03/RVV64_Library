#ifndef DEFS_H
#define DEFS_H

#include <cstddef>
#include <cstdint>

// New scalar softmax prototype
void softmax(const float *i, float *o, float *buf,
             uint64_t channels, uint64_t innerSize);

// New vector softmax prototype
void softmax_vec(const float *i, float *o, uint64_t channels,
                 uint64_t innerSize);


// Utility functions (unchanged)
void write_matrix_to_file(const char* filename, float* matrix, std::size_t rows, std::size_t cols);
void write_matrix_binary(const char* filename, float* matrix, std::size_t count);

#endif