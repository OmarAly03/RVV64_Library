#ifndef DEFS_H
#define DEFS_H

#include <cstddef>

void matmul_e32m1(float *A, float *B, float *C, std::size_t M, std::size_t N, std::size_t K);
void matmul_e32m2(float *A, float *B, float *C, std::size_t M, std::size_t N, std::size_t K);
void matmul_e32m4(float *A, float *B, float *C, std::size_t M, std::size_t N, std::size_t K);
void matmul_e32m8(float *A, float *B, float *C, std::size_t M, std::size_t N, std::size_t K);
void matmul_scalar(float* A, float* B, float* C, std::size_t M, std::size_t N, std::size_t K);

void compute_tile_scalar(const float* A, const float* B, float* C,
	std::size_t M, std::size_t N, std::size_t K,
	std::size_t i_start, std::size_t i_end,
	std::size_t j_start, std::size_t j_end,
	std::size_t k_start, std::size_t k_end);

void matmul_tiled_scalar(const float* A, const float* B, float* C, 
	std::size_t M, std::size_t N, std::size_t K,
	std::size_t tile_m, std::size_t tile_n, std::size_t tile_k);

void compute_tile_e32m1(const float* A, const float* B, float* C,
	size_t M, size_t N, size_t K,
	size_t i_start, size_t i_end,
	size_t j_start, size_t j_end,
	size_t k_start, size_t k_end);

void compute_tile_e32m2(const float* A, const float* B, float* C,
	size_t M, size_t N, size_t K,
	size_t i_start, size_t i_end,
	size_t j_start, size_t j_end,
	size_t k_start, size_t k_end);

void compute_tile_e32m4(const float* A, const float* B, float* C,
	size_t M, size_t N, size_t K,
	size_t i_start, size_t i_end,
	size_t j_start, size_t j_end,
	size_t k_start, size_t k_end);

void compute_tile_e32m8(const float* A, const float* B, float* C,
	size_t M, size_t N, size_t K,
	size_t i_start, size_t i_end,
	size_t j_start, size_t j_end,
	size_t k_start, size_t k_end);

void matmul_tiled_e32m1(const float* A, const float* B, float* C, 
	size_t M, size_t N, size_t K,
	size_t tile_m, size_t tile_n, size_t tile_k);

void matmul_tiled_e32m2(const float* A, const float* B, float* C, 
	size_t M, size_t N, size_t K,
	size_t tile_m, size_t tile_n, size_t tile_k);

void matmul_tiled_e32m4(const float* A, const float* B, float* C, 
	size_t M, size_t N, size_t K,
	size_t tile_m, size_t tile_n, size_t tile_k);

void matmul_tiled_e32m8(const float* A, const float* B, float* C, 
	size_t M, size_t N, size_t K,
	size_t tile_m, size_t tile_n, size_t tile_k);


void write_matrix_to_file(const char* filename, float* matrix, std::size_t rows, std::size_t cols);
void write_matrix_binary(const char* filename, float* matrix, std::size_t count);

#endif