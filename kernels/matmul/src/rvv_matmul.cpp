#include <riscv_vector.h>
#include <cstddef>
#include <cstring>
#include <algorithm>  // Add this for std::min
#include "rvv_defs.hpp"

using namespace std;

extern "C" {

void matmul_e32m1(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j_cnt = N; j_cnt > 0; ) {
            size_t vl = SET_VECTOR_LENGTH<float, M1>(j_cnt);
            size_t j = N - j_cnt;
            
            auto acc = VECTOR_MOVE<float, M1>(0.0f, vl);
            
            for (size_t k = 0; k < K; k++) {
                auto a_elem = VECTOR_MOVE<float, M1>(A[i * K + k], vl);
                auto b_vec = VECTOR_LOAD<float, M1>(&B[k * N + j], vl);
                acc = VECTOR_FMACC<float, M1>(acc, a_elem, b_vec, vl);
            }
            
            VECTOR_STORE<float, M1>(&C[i * N + j], acc, vl);
            
            j_cnt -= vl;
        }
    }
}

void matmul_e32m2(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
	for (size_t i = 0; i < M; i++) {
		for (size_t j_cnt = N; j_cnt > 0; ) {
			size_t vl = SET_VECTOR_LENGTH<float, M2>(j_cnt);
			size_t j = N - j_cnt;
			
			auto acc = VECTOR_MOVE<float, M2>(0.0f, vl);
			
			for (size_t k = 0; k < K; k++) {
				auto a_elem = VECTOR_MOVE<float, M2>(A[i * K + k], vl);
				auto b_vec = VECTOR_LOAD<float, M2>(&B[k * N + j], vl);
				acc = VECTOR_FMACC<float, M2>(acc, a_elem, b_vec, vl);
			}
			
			VECTOR_STORE<float, M2>(&C[i * N + j], acc, vl);
			
			j_cnt -= vl;
		}
	}
}

void matmul_e32m4(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
	for (size_t i = 0; i < M; i++) {
		for (size_t j_cnt = N; j_cnt > 0; ) {
			size_t vl = SET_VECTOR_LENGTH<float, M4>(j_cnt);
			size_t j = N - j_cnt;
			
			auto acc = VECTOR_MOVE<float, M4>(0.0f, vl);
			
			for (size_t k = 0; k < K; k++) {
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
		for (size_t j_cnt = N; j_cnt > 0; ) {
			size_t vl = SET_VECTOR_LENGTH<float, M8>(j_cnt);
			size_t j = N - j_cnt;
			
			auto acc = VECTOR_MOVE<float, M8>(0.0f, vl);
			
			for (size_t k = 0; k < K; k++) {
				auto a_elem = VECTOR_MOVE<float, M8>(A[i * K + k], vl);
				auto b_vec = VECTOR_LOAD<float, M8>(&B[k * N + j], vl);
				acc = VECTOR_FMACC<float, M8>(acc, a_elem, b_vec, vl);
			}
			
			VECTOR_STORE<float, M8>(&C[i * N + j], acc, vl);
			
			j_cnt -= vl;
		}
	}
}

void matmul_scalar(float* A, float* B, float* C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Low-level function: computes a single tile
void compute_tile_scalar(const float* A, const float* B, float* C,
                        size_t M, size_t N, size_t K,
                        size_t i_start, size_t i_end,
                        size_t j_start, size_t j_end,
                        size_t k_start, size_t k_end) {
    
    // Compute this tile using scalar operations
    for (size_t ii = i_start; ii < i_end; ii++) {
        for (size_t jj = j_start; jj < j_end; jj++) {
            float sum = 0.0f;
            for (size_t kk = k_start; kk < k_end; kk++) {
                sum += A[ii * K + kk] * B[kk * N + jj];
            }
            C[ii * N + jj] += sum;  // Accumulate (important for tiling!)
        }
    }
}

// High-level function: handles tiling logic
void matmul_tiled_scalar(const float* A, const float* B, float* C, 
                        size_t M, size_t N, size_t K,
                        size_t tile_m, size_t tile_n, size_t tile_k) {
    
    // Initialize C to zero
    memset(C, 0, M * N * sizeof(float));
    
    // Tile the computation with proper bounds checking
    for (size_t i = 0; i < M; i += tile_m) {
        for (size_t j = 0; j < N; j += tile_n) {
            for (size_t k = 0; k < K; k += tile_k) {
                
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

// Low-level vectorized function: computes a single tile using e32m1
void compute_tile_e32m1(const float* A, const float* B, float* C,
						size_t M, size_t N, size_t K,
						size_t i_start, size_t i_end,
						size_t j_start, size_t j_end,
						size_t k_start, size_t k_end) {
	
	// Compute this tile using vectorized operations (same logic as matmul_e32m1)
	for (size_t ii = i_start; ii < i_end; ii++) {
		// Calculate remaining columns in this tile row
		size_t j_total = j_end - j_start;
		
		for (size_t j_cnt = j_total; j_cnt > 0; ) {
			size_t vl = SET_VECTOR_LENGTH<float, M1>(j_cnt);
			size_t jj = j_start + (j_total - j_cnt);  // Actual column index
			
			// Load existing values from C (important for accumulation in tiling!)
			auto acc = VECTOR_LOAD<float, M1>(&C[ii * N + jj], vl);
			
			for (size_t kk = k_start; kk < k_end; kk++) {
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
void compute_tile_e32m2(const float* A, const float* B, float* C,
						size_t M, size_t N, size_t K,
						size_t i_start, size_t i_end,
						size_t j_start, size_t j_end,
						size_t k_start, size_t k_end) {
	
	// Compute this tile using vectorized operations (same logic as matmul_e32m2)
	for (size_t ii = i_start; ii < i_end; ii++) {
		// Calculate remaining columns in this tile row
		size_t j_total = j_end - j_start;
		
		for (size_t j_cnt = j_total; j_cnt > 0; ) {
			size_t vl = SET_VECTOR_LENGTH<float, M2>(j_cnt);
			size_t jj = j_start + (j_total - j_cnt);  // Actual column index
			
			// Load existing values from C (important for accumulation in tiling!)
			auto acc = VECTOR_LOAD<float, M2>(&C[ii * N + jj], vl);
			
			for (size_t kk = k_start; kk < k_end; kk++) {
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
void compute_tile_e32m4(const float* A, const float* B, float* C,
						size_t M, size_t N, size_t K,
						size_t i_start, size_t i_end,
						size_t j_start, size_t j_end,
						size_t k_start, size_t k_end) {
	
	// Compute this tile using vectorized operations (same logic as matmul_e32m4)
	for (size_t ii = i_start; ii < i_end; ii++) {
		// Calculate remaining columns in this tile row
		size_t j_total = j_end - j_start;
		
		for (size_t j_cnt = j_total; j_cnt > 0; ) {
			size_t vl = SET_VECTOR_LENGTH<float, M4>(j_cnt);
			size_t jj = j_start + (j_total - j_cnt);  // Actual column index
			
			// Load existing values from C (important for accumulation in tiling!)
			auto acc = VECTOR_LOAD<float, M4>(&C[ii * N + jj], vl);
			
			for (size_t kk = k_start; kk < k_end; kk++) {
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
void compute_tile_e32m8(const float* A, const float* B, float* C,
                        size_t M, size_t N, size_t K,
                        size_t i_start, size_t i_end,
                        size_t j_start, size_t j_end,
                        size_t k_start, size_t k_end) {
    
    // Compute this tile using vectorized operations (same logic as matmul_e32m8)
    for (size_t ii = i_start; ii < i_end; ii++) {
        // Calculate remaining columns in this tile row
        size_t j_total = j_end - j_start;
        
        for (size_t j_cnt = j_total; j_cnt > 0; ) {
            size_t vl = SET_VECTOR_LENGTH<float, M8>(j_cnt);
            size_t jj = j_start + (j_total - j_cnt);  // Actual column index
            
            // Load existing values from C (important for accumulation in tiling!)
            auto acc = VECTOR_LOAD<float, M8>(&C[ii * N + jj], vl);
            
            for (size_t kk = k_start; kk < k_end; kk++) {
                auto a_elem = VECTOR_MOVE<float, M8>(A[ii * K + kk], vl);
                auto b_vec = VECTOR_LOAD<float, M8>(&B[kk * N + jj], vl);
                acc = VECTOR_FMACC<float, M8>(acc, a_elem, b_vec, vl);
            }
            
            VECTOR_STORE<float, M8>(&C[ii * N + jj], acc, vl);
            
            j_cnt -= vl;
        }
    }
}

// High-level function using vectorized tiles
void matmul_tiled_e32m1(const float* A, const float* B, float* C, 
                        size_t M, size_t N, size_t K,
                        size_t tile_m, size_t tile_n, size_t tile_k) {
    
	// Initialize C to zero
	for (size_t i = 0; i < M * N; i++) {
		C[i] = 0.0f;
	}
    
    // Tile the computation with proper bounds checking
    for (size_t i = 0; i < M; i += tile_m) {
        for (size_t j = 0; j < N; j += tile_n) {
            for (size_t k = 0; k < K; k += tile_k) {
                
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
void matmul_tiled_e32m2(const float* A, const float* B, float* C, 
                        size_t M, size_t N, size_t K,
                        size_t tile_m, size_t tile_n, size_t tile_k) {
    
	// Initialize C to zero
	for (size_t i = 0; i < M * N; i++) {
		C[i] = 0.0f;
	}
    
    // Tile the computation with proper bounds checking
    for (size_t i = 0; i < M; i += tile_m) {
        for (size_t j = 0; j < N; j += tile_n) {
            for (size_t k = 0; k < K; k += tile_k) {
                
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
void matmul_tiled_e32m4(const float* A, const float* B, float* C, 
                        size_t M, size_t N, size_t K,
                        size_t tile_m, size_t tile_n, size_t tile_k) {
    
	// Initialize C to zero
	for (size_t i = 0; i < M * N; i++) {
		C[i] = 0.0f;
	}
    
    // Tile the computation with proper bounds checking
    for (size_t i = 0; i < M; i += tile_m) {
        for (size_t j = 0; j < N; j += tile_n) {
            for (size_t k = 0; k < K; k += tile_k) {
                
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
void matmul_tiled_e32m8(const float* A, const float* B, float* C, 
                        size_t M, size_t N, size_t K,
                        size_t tile_m, size_t tile_n, size_t tile_k) {
    
	// Initialize C to zero
	for (size_t i = 0; i < M * N; i++) {
		C[i] = 0.0f;
	}
    
    // Tile the computation with proper bounds checking
    for (size_t i = 0; i < M; i += tile_m) {
        for (size_t j = 0; j < N; j += tile_n) {
            for (size_t k = 0; k < K; k += tile_k) {
                
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

}