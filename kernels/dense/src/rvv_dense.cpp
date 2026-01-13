#include <cstddef>
#include <riscv_vector.h>
#include "rvv_defs.hpp"

using namespace std;

/*********************************** Scalar Version (Non-Batched) ************************************/
// Helper: Dense = Matrix Mult + Bias Add
// input: [1 x in_features] or [batch x in_features]
// weights: [in_features x out_features]
// bias: [out_features]
// output: [batch x out_features]

void dense_scalar(const float* input, const float* weights, const float* bias,
    float* output, size_t in_features, size_t out_features) {
    
    size_t K = in_features;
    size_t N = out_features;

    // Weights are [N x K]
    for (size_t j = 0; j < N; j++) {
        float acc = bias[j];
        for (size_t k = 0; k < K; k++) {
            // Note the indexing change: weights[j * K + k]
            acc += input[k] * weights[j * K + k];
        }
        output[j] = acc;
    }
}

/********************************* Vectorized Versions (Non-Batched) *********************************/

void dense_e32m1(const float* input, const float* weights, const float* bias,
	float* output, size_t in_features, size_t out_features) {
	 
	 size_t K = in_features;
	 size_t N = out_features;
 
	 // We iterate through outputs (j)
	 for (size_t j_idx = 0; j_idx < N; ) {
		 size_t vl = SET_VECTOR_LENGTH<float, M1>(N - j_idx);
 
		 // 1. Initialize with bias
		 auto v_acc = VECTOR_LOAD<float, M1>(&bias[j_idx], vl);
 
		 // 2. Since weights are [OUT, IN], we can't easily broadcast input[k] 
		 // across weights[k*N+j] like we did before.
		 // Instead, we compute the dot product for each j in the vector.
		 
		 // REVISED LOGIC: For each j in the current vector segment
		 for (size_t k = 0; k < K; k++) {
			 // We need to load weights[j_idx...j_idx+vl][k]
			 // BUT: In [N x K] layout, the elements weights[j][k] and weights[j+1][k] 
			 // are NOT contiguous. They are K elements apart.
			 
			 // This requires a STRIDED LOAD
			 auto v_w = VECTOR_STRIDED_LOAD<float, M1>(&weights[j_idx * K + k], K * sizeof(float), vl);
			 auto v_in = VECTOR_MOVE<float, M1>(input[k], vl);
			 v_acc = VECTOR_FMACC<float, M1>(v_acc, v_in, v_w, vl);
		 }
 
		 VECTOR_STORE<float, M1>(&output[j_idx], v_acc, vl);
		 j_idx += vl;
	 }
 }

void dense_e32m2(const float* input, const float* weights, const float* bias,
	float* output, size_t in_features, size_t out_features) {
	 
	 size_t K = in_features;
	 size_t N = out_features;
 
	 // We iterate through outputs (j)
	 for (size_t j_idx = 0; j_idx < N; ) {
		 size_t vl = SET_VECTOR_LENGTH<float, M2>(N - j_idx);
 
		 // 1. Initialize with bias
		 auto v_acc = VECTOR_LOAD<float, M2>(&bias[j_idx], vl);
 
		 // 2. Since weights are [OUT, IN], we can't easily broadcast input[k] 
		 // across weights[k*N+j] like we did before.
		 // Instead, we compute the dot product for each j in the vector.
		 
		 // REVISED LOGIC: For each j in the current vector segment
		 for (size_t k = 0; k < K; k++) {
			 // We need to load weights[j_idx...j_idx+vl][k]
			 // BUT: In [N x K] layout, the elements weights[j][k] and weights[j+1][k] 
			 // are NOT contiguous. They are K elements apart.
			 
			 // This requires a STRIDED LOAD
			 auto v_w = VECTOR_STRIDED_LOAD<float, M2>(&weights[j_idx * K + k], K * sizeof(float), vl);
			 auto v_in = VECTOR_MOVE<float, M2>(input[k], vl);
			 v_acc = VECTOR_FMACC<float, M2>(v_acc, v_in, v_w, vl);
		 }
 
		 VECTOR_STORE<float, M2>(&output[j_idx], v_acc, vl);
		 j_idx += vl;
	 }
 }

void dense_e32m4(const float* input, const float* weights, const float* bias,
	float* output, size_t in_features, size_t out_features) {
	 
	 size_t K = in_features;
	 size_t N = out_features;
 
	 // We iterate through outputs (j)
	 for (size_t j_idx = 0; j_idx < N; ) {
		 size_t vl = SET_VECTOR_LENGTH<float, M4>(N - j_idx);
 
		 // 1. Initialize with bias
		 auto v_acc = VECTOR_LOAD<float, M4>(&bias[j_idx], vl);
 
		 // 2. Since weights are [OUT, IN], we can't easily broadcast input[k] 
		 // across weights[k*N+j] like we did before.
		 // Instead, we compute the dot product for each j in the vector.
		 
		 // REVISED LOGIC: For each j in the current vector segment
		 for (size_t k = 0; k < K; k++) {
			 // We need to load weights[j_idx...j_idx+vl][k]
			 // BUT: In [N x K] layout, the elements weights[j][k] and weights[j+1][k] 
			 // are NOT contiguous. They are K elements apart.
			 
			 // This requires a STRIDED LOAD
			 auto v_w = VECTOR_STRIDED_LOAD<float, M4>(&weights[j_idx * K + k], K * sizeof(float), vl);
			 auto v_in = VECTOR_MOVE<float, M4>(input[k], vl);
			 v_acc = VECTOR_FMACC<float, M4>(v_acc, v_in, v_w, vl);
		 }
 
		 VECTOR_STORE<float, M4>(&output[j_idx], v_acc, vl);
		 j_idx += vl;
	 }
 }

void dense_e32m8(const float* input, const float* weights, const float* bias,
	float* output, size_t in_features, size_t out_features) {
	 
	 size_t K = in_features;
	 size_t N = out_features;
 
	 // We iterate through outputs (j)
	 for (size_t j_idx = 0; j_idx < N; ) {
		 size_t vl = SET_VECTOR_LENGTH<float, M8>(N - j_idx);
 
		 // 1. Initialize with bias
		 auto v_acc = VECTOR_LOAD<float, M8>(&bias[j_idx], vl);
 
		 // 2. Since weights are [OUT, IN], we can't easily broadcast input[k] 
		 // across weights[k*N+j] like we did before.
		 // Instead, we compute the dot product for each j in the vector.
		 
		 // REVISED LOGIC: For each j in the current vector segment
		 for (size_t k = 0; k < K; k++) {
			 // We need to load weights[j_idx...j_idx+vl][k]
			 // BUT: In [N x K] layout, the elements weights[j][k] and weights[j+1][k] 
			 // are NOT contiguous. They are K elements apart.
			 
			 // This requires a STRIDED LOAD
			 auto v_w = VECTOR_STRIDED_LOAD<float, M8>(&weights[j_idx * K + k], K * sizeof(float), vl);
			 auto v_in = VECTOR_MOVE<float, M8>(input[k], vl);
			 v_acc = VECTOR_FMACC<float, M8>(v_acc, v_in, v_w, vl);
		 }
 
		 VECTOR_STORE<float, M8>(&output[j_idx], v_acc, vl);
		 j_idx += vl;
	 }
 }

 