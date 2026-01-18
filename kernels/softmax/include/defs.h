#ifndef DEFS_H
#define DEFS_H

#include <cstddef>
#include <cstdint>


void softmax(
    const float* input,
    float* output,
    size_t n
);

// Utility functions 
void write_matrix_to_file(const char* filename, float* matrix, std::size_t rows, std::size_t cols);
void write_matrix_binary(const char* filename, float* matrix, std::size_t count);

#endif