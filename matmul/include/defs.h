#ifndef DEFS_H
#define DEFS_H

#include <cstddef>

void matmul_e32m1(float *A, float *B, float *C, std::size_t M, std::size_t N, std::size_t K);
void matmul_e32m2(float *A, float *B, float *C, std::size_t M, std::size_t N, std::size_t K);
void matmul_e32m4(float *A, float *B, float *C, std::size_t M, std::size_t N, std::size_t K);
void matmul_e32m8(float *A, float *B, float *C, std::size_t M, std::size_t N, std::size_t K);

void matmul_scalar(float* A, float* B, float* C, std::size_t M, std::size_t N, std::size_t K);

void write_matrix_to_file(const char* filename, float* matrix, std::size_t rows, std::size_t cols);
void write_matrix_binary(const char* filename, float* matrix, std::size_t count);

#endif