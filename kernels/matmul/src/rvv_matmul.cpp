#include <riscv_vector.h>
#include <cstddef>
#include <cstring>
#include <algorithm> // Add this for std::min
#include "rvv_defs.hpp"

using namespace std;

/*********************************** Scalar ************************************/

void matmul_scalar(float *A, float *B, float *C, size_t M, size_t N, size_t K)
{
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			float sum = 0.0f;
			for (size_t k = 0; k < K; k++)
			{
				sum += A[i * K + k] * B[k * N + j];
			}
			C[i * N + j] = sum;
		}
	}
}

/********************************* Vectorized *********************************/

void matmul_e32m1(float *A, float *B, float *C, size_t M, size_t N, size_t K)
{
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j_cnt = N; j_cnt > 0;)
		{
			size_t vl = SET_VECTOR_LENGTH<float, M1>(j_cnt);
			size_t j = N - j_cnt;

			auto acc = VECTOR_MOVE<float, M1>(0.0f, vl);

			for (size_t k = 0; k < K; k++)
			{
				auto a_elem = VECTOR_MOVE<float, M1>(A[i * K + k], vl);
				auto b_vec = VECTOR_LOAD<float, M1>(&B[k * N + j], vl);
				acc = VECTOR_FMACC<float, M1>(acc, a_elem, b_vec, vl);
			}

			VECTOR_STORE<float, M1>(&C[i * N + j], acc, vl);

			j_cnt -= vl;
		}
	}
}

void matmul_e32m2(float *A, float *B, float *C, size_t M, size_t N, size_t K)
{
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j_cnt = N; j_cnt > 0;)
		{
			size_t vl = SET_VECTOR_LENGTH<float, M2>(j_cnt);
			size_t j = N - j_cnt;

			auto acc = VECTOR_MOVE<float, M2>(0.0f, vl);

			for (size_t k = 0; k < K; k++)
			{
				auto a_elem = VECTOR_MOVE<float, M2>(A[i * K + k], vl);
				auto b_vec = VECTOR_LOAD<float, M2>(&B[k * N + j], vl);
				acc = VECTOR_FMACC<float, M2>(acc, a_elem, b_vec, vl);
			}

			VECTOR_STORE<float, M2>(&C[i * N + j], acc, vl);

			j_cnt -= vl;
		}
	}
}

void matmul_e32m4(float *A, float *B, float *C, size_t M, size_t N, size_t K)
{
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j_cnt = N; j_cnt > 0;)
		{
			size_t vl = SET_VECTOR_LENGTH<float, M4>(j_cnt);
			size_t j = N - j_cnt;

			auto acc = VECTOR_MOVE<float, M4>(0.0f, vl);

			for (size_t k = 0; k < K; k++)
			{
				auto a_elem = VECTOR_MOVE<float, M4>(A[i * K + k], vl);
				auto b_vec = VECTOR_LOAD<float, M4>(&B[k * N + j], vl);
				acc = VECTOR_FMACC<float, M4>(acc, a_elem, b_vec, vl);
			}

			VECTOR_STORE<float, M4>(&C[i * N + j], acc, vl);

			j_cnt -= vl;
		}
	}
}

void matmul_e32m8(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
	for (size_t i = 0; i < M; i++) {
		size_t j_idx = 0;
		size_t j_left = N;
		
		while (j_left > 0) {
			size_t vl = SET_VECTOR_LENGTH<float, M8>(j_left);
			
			// Initialize accumulator vector with zeros
			auto v_acc = VECTOR_MOVE<float, M8>(0.0f, vl);

			for (size_t k = 0; k < K; k++) {
				// Broadcast A[i][k]
				float a_val = A[i * K + k];
				
				// Load row segment from B
				auto v_b = VECTOR_LOAD<float, M8>(&B[k * N + j_idx], vl);
				
				// Multiply-Accumulate: v_acc += a_val * v_b
				v_acc = VECTOR_FMACC<float, M8>(v_acc, a_val, v_b, vl);
			}

			// Store result segment to C
			VECTOR_STORE<float, M8>(&C[i * N + j_idx], v_acc, vl);

			j_idx += vl;
			j_left -= vl;
		}
	}
}

/********************************* Vectorized Unrolled *********************************/

void matmul_e32m1_unroll(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
	for (size_t i = 0; i + 7 < M; i += 8) {
		size_t j_idx = 0;
		size_t j_left = N;
		while (j_left > 0) {
			size_t vl = SET_VECTOR_LENGTH<float, M1>(j_left);
			
			// 8 Accumulators
			auto v_acc0 = VECTOR_MOVE<float, M1>(0.0f, vl);
			auto v_acc1 = VECTOR_MOVE<float, M1>(0.0f, vl);
			auto v_acc2 = VECTOR_MOVE<float, M1>(0.0f, vl);
			auto v_acc3 = VECTOR_MOVE<float, M1>(0.0f, vl);
			auto v_acc4 = VECTOR_MOVE<float, M1>(0.0f, vl);
			auto v_acc5 = VECTOR_MOVE<float, M1>(0.0f, vl);
			auto v_acc6 = VECTOR_MOVE<float, M1>(0.0f, vl);
			auto v_acc7 = VECTOR_MOVE<float, M1>(0.0f, vl);

			for (size_t k = 0; k < K; k++) {
				// Load 1 vector from B
				auto v_b = VECTOR_LOAD<float, M1>(&B[k * N + j_idx], vl);
				
				// 8 FMACCs using scalars from 8 rows of A
				v_acc0 = VECTOR_FMACC<float, M1>(v_acc0, A[(i+0)*K+k], v_b, vl);
				v_acc1 = VECTOR_FMACC<float, M1>(v_acc1, A[(i+1)*K+k], v_b, vl);
				v_acc2 = VECTOR_FMACC<float, M1>(v_acc2, A[(i+2)*K+k], v_b, vl);
				v_acc3 = VECTOR_FMACC<float, M1>(v_acc3, A[(i+3)*K+k], v_b, vl);
				v_acc4 = VECTOR_FMACC<float, M1>(v_acc4, A[(i+4)*K+k], v_b, vl);
				v_acc5 = VECTOR_FMACC<float, M1>(v_acc5, A[(i+5)*K+k], v_b, vl);
				v_acc6 = VECTOR_FMACC<float, M1>(v_acc6, A[(i+6)*K+k], v_b, vl);
				v_acc7 = VECTOR_FMACC<float, M1>(v_acc7, A[(i+7)*K+k], v_b, vl);
			}

			// Stores...
			VECTOR_STORE<float, M1>(&C[(i+0)*N + j_idx], v_acc0, vl);
			VECTOR_STORE<float, M1>(&C[(i+1)*N + j_idx], v_acc1, vl);
			VECTOR_STORE<float, M1>(&C[(i+2)*N + j_idx], v_acc2, vl);
			VECTOR_STORE<float, M1>(&C[(i+3)*N + j_idx], v_acc3, vl);
			VECTOR_STORE<float, M1>(&C[(i+4)*N + j_idx], v_acc4, vl);
			VECTOR_STORE<float, M1>(&C[(i+5)*N + j_idx], v_acc5, vl);
			VECTOR_STORE<float, M1>(&C[(i+6)*N + j_idx], v_acc6, vl);
			VECTOR_STORE<float, M1>(&C[(i+7)*N + j_idx], v_acc7, vl);

			j_idx += vl;
			j_left -= vl;
		}
	}
}

void matmul_e32m2_unroll(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
	// Process 4 rows of A at a time
	for (size_t i = 0; i + 3 < M; i += 4) {
		size_t j_idx = 0;
		size_t j_left = N;
		
		while (j_left > 0) {
			size_t vl = SET_VECTOR_LENGTH<float, M2>(j_left);
			
			// 4 Accumulators for 4 different rows of C
			auto v_acc0 = VECTOR_MOVE<float, M2>(0.0f, vl);
			auto v_acc1 = VECTOR_MOVE<float, M2>(0.0f, vl);
			auto v_acc2 = VECTOR_MOVE<float, M2>(0.0f, vl);
			auto v_acc3 = VECTOR_MOVE<float, M2>(0.0f, vl);

			for (size_t k = 0; k < K; k++) {
				// Load 4 scalar values from 4 different rows of A
				float a0 = A[(i + 0) * K + k];
				float a1 = A[(i + 1) * K + k];
				float a2 = A[(i + 2) * K + k];
				float a3 = A[(i + 3) * K + k];
				
				// Load 1 vector from B (Shared by all 4 rows)
				auto v_b = VECTOR_LOAD<float, M2>(&B[k * N + j_idx], vl);
				
				// 4 FMACCs: Ara can pipeline these together
				v_acc0 = VECTOR_FMACC<float, M2>(v_acc0, a0, v_b, vl);
				v_acc1 = VECTOR_FMACC<float, M2>(v_acc1, a1, v_b, vl);
				v_acc2 = VECTOR_FMACC<float, M2>(v_acc2, a2, v_b, vl);
				v_acc3 = VECTOR_FMACC<float, M2>(v_acc3, a3, v_b, vl);
			}

			// Store 4 result segments to 4 different rows of C
			VECTOR_STORE<float, M2>(&C[(i + 0) * N + j_idx], v_acc0, vl);
			VECTOR_STORE<float, M2>(&C[(i + 1) * N + j_idx], v_acc1, vl);
			VECTOR_STORE<float, M2>(&C[(i + 2) * N + j_idx], v_acc2, vl);
			VECTOR_STORE<float, M2>(&C[(i + 3) * N + j_idx], v_acc3, vl);

			j_idx += vl;
			j_left -= vl;
		}
	}
}

void matmul_e32m4_unroll(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
	for (size_t i = 0; i + 3 < M; i += 4) {
		size_t j_idx = 0;
		size_t j_left = N;
		while (j_left > 0) {
			size_t vl = SET_VECTOR_LENGTH<float, M4>(j_left);
			
			auto v_acc0 = VECTOR_MOVE<float, M4>(0.0f, vl);
			auto v_acc1 = VECTOR_MOVE<float, M4>(0.0f, vl);
			auto v_acc2 = VECTOR_MOVE<float, M4>(0.0f, vl);
			auto v_acc3 = VECTOR_MOVE<float, M4>(0.0f, vl);

			for (size_t k = 0; k < K; k++) {
				float a0 = A[(i + 0) * K + k];
				float a1 = A[(i + 1) * K + k];
				float a2 = A[(i + 2) * K + k];
				float a3 = A[(i + 3) * K + k];
				
				auto v_b = VECTOR_LOAD<float, M4>(&B[k * N + j_idx], vl);
				
				v_acc0 = VECTOR_FMACC<float, M4>(v_acc0, a0, v_b, vl);
				v_acc1 = VECTOR_FMACC<float, M4>(v_acc1, a1, v_b, vl);
				v_acc2 = VECTOR_FMACC<float, M4>(v_acc2, a2, v_b, vl);
				v_acc3 = VECTOR_FMACC<float, M4>(v_acc3, a3, v_b, vl);
			}

			VECTOR_STORE<float, M4>(&C[(i + 0) * N + j_idx], v_acc0, vl);
			VECTOR_STORE<float, M4>(&C[(i + 1) * N + j_idx], v_acc1, vl);
			VECTOR_STORE<float, M4>(&C[(i + 2) * N + j_idx], v_acc2, vl);
			VECTOR_STORE<float, M4>(&C[(i + 3) * N + j_idx], v_acc3, vl);

			j_idx += vl;
			j_left -= vl;
		}
	}
}

void matmul_e32m8_unroll(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
	for (size_t i = 0; i + 1 < M; i += 2) {
		size_t j_idx = 0;
		size_t j_left = N;
		while (j_left > 0) {
			size_t vl = SET_VECTOR_LENGTH<float, M8>(j_left);
			
			auto v_acc0 = VECTOR_MOVE<float, M8>(0.0f, vl);
			auto v_acc1 = VECTOR_MOVE<float, M8>(0.0f, vl);

			for (size_t k = 0; k < K; k++) {
				float a0 = A[(i + 0) * K + k];
				float a1 = A[(i + 1) * K + k];
				
				auto v_b = VECTOR_LOAD<float, M8>(&B[k * N + j_idx], vl);
				
				v_acc0 = VECTOR_FMACC<float, M8>(v_acc0, a0, v_b, vl);
				v_acc1 = VECTOR_FMACC<float, M8>(v_acc1, a1, v_b, vl);
			}

			VECTOR_STORE<float, M8>(&C[(i + 0) * N + j_idx], v_acc0, vl);
			VECTOR_STORE<float, M8>(&C[(i + 1) * N + j_idx], v_acc1, vl);

			j_idx += vl;
			j_left -= vl;
		}
	}
}


/*********************************** Tiled *************************************/

// Low-level scalar function: computes a single tile
void compute_tile_scalar(const float *A, const float *B, float *C,
							size_t M, size_t N, size_t K,
							size_t i_start, size_t i_end,
							size_t j_start, size_t j_end,
							size_t k_start, size_t k_end)
{

	// Compute this tile using scalar operations
	for (size_t ii = i_start; ii < i_end; ii++)
	{
		for (size_t jj = j_start; jj < j_end; jj++)
		{
			float sum = 0.0f;
			for (size_t kk = k_start; kk < k_end; kk++)
			{
				sum += A[ii * K + kk] * B[kk * N + jj];
			}
			C[ii * N + jj] += sum; // Accumulate (important for tiling!)
		}
	}
}

// Low-level vectorized function: computes a single tile using e32m1
void compute_tile_e32m1(const float *A, const float *B, float *C,
						size_t M, size_t N, size_t K,
						size_t i_start, size_t i_end,
						size_t j_start, size_t j_end,
						size_t k_start, size_t k_end)
{

	// Compute this tile using vectorized operations (same logic as matmul_e32m1)
	for (size_t ii = i_start; ii < i_end; ii++)
	{
		// Calculate remaining columns in this tile row
		size_t j_total = j_end - j_start;

		for (size_t j_cnt = j_total; j_cnt > 0;)
		{
			size_t vl = SET_VECTOR_LENGTH<float, M1>(j_cnt);
			size_t jj = j_start + (j_total - j_cnt); // Actual column index

			// Load existing values from C (important for accumulation in tiling!)
			auto acc = VECTOR_LOAD<float, M1>(&C[ii * N + jj], vl);

			for (size_t kk = k_start; kk < k_end; kk++)
			{
				auto a_elem = VECTOR_MOVE<float, M1>(A[ii * K + kk], vl);
				auto b_vec = VECTOR_LOAD<float, M1>(&B[kk * N + jj], vl);
				acc = VECTOR_FMACC<float, M1>(acc, a_elem, b_vec, vl);
			}

			VECTOR_STORE<float, M1>(&C[ii * N + jj], acc, vl);

			j_cnt -= vl;
		}
	}
}

// Low-level vectorized function: computes a single tile using e32m2
void compute_tile_e32m2(const float *A, const float *B, float *C,
						size_t M, size_t N, size_t K,
						size_t i_start, size_t i_end,
						size_t j_start, size_t j_end,
						size_t k_start, size_t k_end)
{

	// Compute this tile using vectorized operations (same logic as matmul_e32m2)
	for (size_t ii = i_start; ii < i_end; ii++)
	{
		// Calculate remaining columns in this tile row
		size_t j_total = j_end - j_start;

		for (size_t j_cnt = j_total; j_cnt > 0;)
		{
			size_t vl = SET_VECTOR_LENGTH<float, M2>(j_cnt);
			size_t jj = j_start + (j_total - j_cnt); // Actual column index

			// Load existing values from C (important for accumulation in tiling!)
			auto acc = VECTOR_LOAD<float, M2>(&C[ii * N + jj], vl);

			for (size_t kk = k_start; kk < k_end; kk++)
			{
				auto a_elem = VECTOR_MOVE<float, M2>(A[ii * K + kk], vl);
				auto b_vec = VECTOR_LOAD<float, M2>(&B[kk * N + jj], vl);
				acc = VECTOR_FMACC<float, M2>(acc, a_elem, b_vec, vl);
			}

			VECTOR_STORE<float, M2>(&C[ii * N + jj], acc, vl);

			j_cnt -= vl;
		}
	}
}

// Low-level vectorized function: computes a single tile using e32m4
void compute_tile_e32m4(const float *A, const float *B, float *C,
						size_t M, size_t N, size_t K,
						size_t i_start, size_t i_end,
						size_t j_start, size_t j_end,
						size_t k_start, size_t k_end)
{

	// Compute this tile using vectorized operations (same logic as matmul_e32m4)
	for (size_t ii = i_start; ii < i_end; ii++)
	{
		// Calculate remaining columns in this tile row
		size_t j_total = j_end - j_start;

		for (size_t j_cnt = j_total; j_cnt > 0;)
		{
			size_t vl = SET_VECTOR_LENGTH<float, M4>(j_cnt);
			size_t jj = j_start + (j_total - j_cnt); // Actual column index

			// Load existing values from C (important for accumulation in tiling!)
			auto acc = VECTOR_LOAD<float, M4>(&C[ii * N + jj], vl);

			for (size_t kk = k_start; kk < k_end; kk++)
			{
				auto a_elem = VECTOR_MOVE<float, M4>(A[ii * K + kk], vl);
				auto b_vec = VECTOR_LOAD<float, M4>(&B[kk * N + jj], vl);
				acc = VECTOR_FMACC<float, M4>(acc, a_elem, b_vec, vl);
			}

			VECTOR_STORE<float, M4>(&C[ii * N + jj], acc, vl);

			j_cnt -= vl;
		}
	}
}

// Low-level vectorized function: computes a single tile using e32m8
void compute_tile_e32m8(const float *A, const float *B, float *C,
						size_t M, size_t N, size_t K,
						size_t i_start, size_t i_end,
						size_t j_start, size_t j_end,
						size_t k_start, size_t k_end)
{

	// Compute this tile using vectorized operations (same logic as matmul_e32m8)
	for (size_t ii = i_start; ii < i_end; ii++)
	{
		// Calculate remaining columns in this tile row
		size_t j_total = j_end - j_start;

		for (size_t j_cnt = j_total; j_cnt > 0;)
		{
			size_t vl = SET_VECTOR_LENGTH<float, M8>(j_cnt);
			size_t jj = j_start + (j_total - j_cnt); // Actual column index

			// Load existing values from C (important for accumulation in tiling!)
			auto acc = VECTOR_LOAD<float, M8>(&C[ii * N + jj], vl);

			for (size_t kk = k_start; kk < k_end; kk++)
			{
				auto a_elem = VECTOR_MOVE<float, M8>(A[ii * K + kk], vl);
				auto b_vec = VECTOR_LOAD<float, M8>(&B[kk * N + jj], vl);
				acc = VECTOR_FMACC<float, M8>(acc, a_elem, b_vec, vl);
			}

			VECTOR_STORE<float, M8>(&C[ii * N + jj], acc, vl);

			j_cnt -= vl;
		}
	}
}


// High-level function: handles tiling logic
void matmul_tiled_scalar(const float *A, const float *B, float *C,
							size_t M, size_t N, size_t K,
							size_t tile_m, size_t tile_n, size_t tile_k)
{

	// Initialize C to zero
	memset(C, 0, M * N * sizeof(float));

	// Tile the computation with proper bounds checking
	for (size_t i = 0; i < M; i += tile_m)
	{
		for (size_t j = 0; j < N; j += tile_n)
		{
			for (size_t k = 0; k < K; k += tile_k)
			{

				// Calculate actual tile boundaries (handle partial tiles)
				size_t i_end = min(i + tile_m, M);
				size_t j_end = min(j + tile_n, N);
				size_t k_end = min(k + tile_k, K);

				// Call the low-level compute function for this tile
				compute_tile_scalar(A, B, C, M, N, K, i, i_end, j, j_end, k, k_end);
			}
		}
	}
}

    
// High-level function using vectorized tiles
void matmul_tiled_e32m1(const float *A, const float *B, float *C,
						size_t M, size_t N, size_t K,
						size_t tile_m, size_t tile_n, size_t tile_k)
{

	// Initialize C to zero
	for (size_t i = 0; i < M * N; i++)
	{
		C[i] = 0.0f;
	}

	// Tile the computation with proper bounds checking
	for (size_t i = 0; i < M; i += tile_m)
	{
		for (size_t j = 0; j < N; j += tile_n)
		{
			for (size_t k = 0; k < K; k += tile_k)
			{

				// Calculate actual tile boundaries (handle partial tiles)
				size_t i_end = min(i + tile_m, M);
				size_t j_end = min(j + tile_n, N);
				size_t k_end = min(k + tile_k, K);

				// Call the low-level vectorized compute function for this tile
				compute_tile_e32m1(A, B, C, M, N, K, i, i_end, j, j_end, k, k_end);
			}
		}
	}
}

// High-level function using vectorized tiles
void matmul_tiled_e32m2(const float *A, const float *B, float *C,
						size_t M, size_t N, size_t K,
						size_t tile_m, size_t tile_n, size_t tile_k)
{

	// Initialize C to zero
	for (size_t i = 0; i < M * N; i++)
	{
		C[i] = 0.0f;
	}

	// Tile the computation with proper bounds checking
	for (size_t i = 0; i < M; i += tile_m)
	{
		for (size_t j = 0; j < N; j += tile_n)
		{
			for (size_t k = 0; k < K; k += tile_k)
			{

				// Calculate actual tile boundaries (handle partial tiles)
				size_t i_end = min(i + tile_m, M);
				size_t j_end = min(j + tile_n, N);
				size_t k_end = min(k + tile_k, K);

				// Call the low-level vectorized compute function for this tile
				compute_tile_e32m2(A, B, C, M, N, K, i, i_end, j, j_end, k, k_end);
			}
		}
	}
}

// High-level function using vectorized tiles
void matmul_tiled_e32m4(const float *A, const float *B, float *C,
						size_t M, size_t N, size_t K,
						size_t tile_m, size_t tile_n, size_t tile_k)
{

	// Initialize C to zero
	for (size_t i = 0; i < M * N; i++)
	{
		C[i] = 0.0f;
	}

	// Tile the computation with proper bounds checking
	for (size_t i = 0; i < M; i += tile_m)
	{
		for (size_t j = 0; j < N; j += tile_n)
		{
			for (size_t k = 0; k < K; k += tile_k)
			{

				// Calculate actual tile boundaries (handle partial tiles)
				size_t i_end = min(i + tile_m, M);
				size_t j_end = min(j + tile_n, N);
				size_t k_end = min(k + tile_k, K);

				// Call the low-level vectorized compute function for this tile
				compute_tile_e32m4(A, B, C, M, N, K, i, i_end, j, j_end, k, k_end);
			}
		}
	}
}

// High-level function using vectorized tiles
void matmul_tiled_e32m8(const float *A, const float *B, float *C,
						size_t M, size_t N, size_t K,
						size_t tile_m, size_t tile_n, size_t tile_k)
{

	// Initialize C to zero
	for (size_t i = 0; i < M * N; i++)
	{
		C[i] = 0.0f;
	}

	// Tile the computation with proper bounds checking
	for (size_t i = 0; i < M; i += tile_m)
	{
		for (size_t j = 0; j < N; j += tile_n)
		{
			for (size_t k = 0; k < K; k += tile_k)
			{

				// Calculate actual tile boundaries (handle partial tiles)
				size_t i_end = min(i + tile_m, M);
				size_t j_end = min(j + tile_n, N);
				size_t k_end = min(k + tile_k, K);

				// Call the low-level vectorized compute function for this tile
				compute_tile_e32m8(A, B, C, M, N, K, i, i_end, j, j_end, k, k_end);
			}
		}
	}
}



