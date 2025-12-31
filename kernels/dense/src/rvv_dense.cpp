#include <cstddef>
#include <riscv_vector.h>
#include "rvv_defs.hpp"

using namespace std;

/*********************************** Scalar Version (Non-Batched) ************************************/

// This is your original scalar version, slightly adapted to be non-batched
void dense_scalar(const float* input, const float* weights, const float* bias,
                        float* output, size_t in_features, size_t out_features) {
    // Implements Y = A*B^T + C, where A=input, B=weights, C=bias
    // A shape: [in_features]
    // B shape: [out_features, in_features]
    // C shape: [out_features]
    // Y shape: [out_features]
    for (size_t out_f = 0; out_f < out_features; ++out_f) {
        float sum = 0.0f;
        for (size_t in_f = 0; in_f < in_features; ++in_f) {
            sum += input[in_f] * weights[out_f * in_features + in_f];
        }
        output[out_f] = sum + bias[out_f];
    }
}


/********************************* Vectorized Versions (Non-Batched) *********************************/

void dense_e32m1(const float* input, const float* weights, const float* bias,
				   float* output, size_t in_features, size_t out_features) {
	for (size_t out_f = 0; out_f < out_features; ++out_f) {
		
		// Pointers are simpler: input is the start of the vector.
		const float* a_ptr = input;
		const float* b_ptr = &weights[out_f * in_features];
		size_t cnt = in_features;
		size_t vl;

		auto v_sum = VECTOR_BROADCAST<float, M1>(0.0f, SET_VECTOR_LENGTH<float, M1>(in_features));

		for (; cnt > 0; cnt -= vl) {
			vl = SET_VECTOR_LENGTH<float, M1>(cnt);
			
			auto v_a = VECTOR_LOAD<float, M1>(a_ptr, vl);
			auto v_b = VECTOR_LOAD<float, M1>(b_ptr, vl);
			v_sum = VECTOR_FMACC<float, M1>(v_sum, v_a, v_b, vl);
			
			a_ptr += vl;
			b_ptr += vl;
		}

		auto v_scalar_sum = VECTOR_BROADCAST<float, M1>(0.0f, SET_VECTOR_LENGTH<float, M1>(1));
		v_scalar_sum = VECTOR_VFREDSUM<float, M1>(v_sum, v_scalar_sum, SET_VECTOR_LENGTH<float, M1>(in_features));
		auto sum = VECTOR_EXTRACT_SCALAR<float, M1>(v_scalar_sum);

		// Output is simpler: just [out_f]
		output[out_f] = sum + bias[out_f];
	}
}

void dense_e32m2(const float* input, const float* weights, const float* bias,
				   float* output, size_t in_features, size_t out_features) {
	for (size_t out_f = 0; out_f < out_features; ++out_f) {
		
		const float* a_ptr = input;
		const float* b_ptr = &weights[out_f * in_features];
		size_t cnt = in_features;
		size_t vl;

		auto v_sum = VECTOR_BROADCAST<float, M2>(0.0f, SET_VECTOR_LENGTH<float, M2>(in_features));

		for (; cnt > 0; cnt -= vl) {
			vl = SET_VECTOR_LENGTH<float, M2>(cnt);
			auto v_a = VECTOR_LOAD<float, M2>(a_ptr, vl);
			auto v_b = VECTOR_LOAD<float, M2>(b_ptr, vl);
			v_sum = VECTOR_FMACC<float, M2>(v_sum, v_a, v_b, vl);
			a_ptr += vl;
			b_ptr += vl;
		}

		auto v_scalar_sum = VECTOR_BROADCAST<float, M1>(0.0f, SET_VECTOR_LENGTH<float, M1>(1));
		v_scalar_sum = VECTOR_VFREDSUM<float, M2>(v_sum, v_scalar_sum, SET_VECTOR_LENGTH<float, M2>(in_features));
		auto sum = VECTOR_EXTRACT_SCALAR<float, M1>(v_scalar_sum);

		output[out_f] = sum + bias[out_f];
	}
}

void dense_e32m4(const float* input, const float* weights, const float* bias,
				   float* output, size_t in_features, size_t out_features) {
	for (size_t out_f = 0; out_f < out_features; ++out_f) {
		
		const float* a_ptr = input;
		const float* b_ptr = &weights[out_f * in_features];
		size_t cnt = in_features;
		size_t vl;

		auto v_sum = VECTOR_BROADCAST<float, M4>(0.0f, SET_VECTOR_LENGTH<float, M4>(in_features));

		for (; cnt > 0; cnt -= vl) {
			vl = SET_VECTOR_LENGTH<float, M4>(cnt);
			auto v_a = VECTOR_LOAD<float, M4>(a_ptr, vl);
			auto v_b = VECTOR_LOAD<float, M4>(b_ptr, vl);
			v_sum = VECTOR_FMACC<float, M4>(v_sum, v_a, v_b, vl);
			a_ptr += vl;
			b_ptr += vl;
		}

		auto v_scalar_sum = VECTOR_BROADCAST<float, M1>(0.0f, SET_VECTOR_LENGTH<float, M1>(1));
		v_scalar_sum = VECTOR_VFREDSUM<float, M4>(v_sum, v_scalar_sum, SET_VECTOR_LENGTH<float, M4>(in_features));
		auto sum = VECTOR_EXTRACT_SCALAR<float, M1>(v_scalar_sum);

		output[out_f] = sum + bias[out_f];
	}
}

void dense_e32m8(const float* input, const float* weights, const float* bias,
				   float* output, size_t in_features, size_t out_features) {
	for (size_t out_f = 0; out_f < out_features; ++out_f) {
		
		const float* a_ptr = input;
		const float* b_ptr = &weights[out_f * in_features];
		size_t cnt = in_features;
		size_t vl;

		auto v_sum = VECTOR_BROADCAST<float, M8>(0.0f, SET_VECTOR_LENGTH<float, M8>(in_features));

		for (; cnt > 0; cnt -= vl) {
			vl = SET_VECTOR_LENGTH<float, M8>(cnt);
			auto v_a = VECTOR_LOAD<float, M8>(a_ptr, vl);
			auto v_b = VECTOR_LOAD<float, M8>(b_ptr, vl);
			v_sum = VECTOR_FMACC<float, M8>(v_sum, v_a, v_b, vl);
			a_ptr += vl;
			b_ptr += vl;
		}

		auto v_scalar_sum = VECTOR_BROADCAST<float, M1>(0.0f, SET_VECTOR_LENGTH<float, M1>(1));
		v_scalar_sum = VECTOR_VFREDSUM<float, M8>(v_sum, v_scalar_sum, SET_VECTOR_LENGTH<float, M8>(in_features));
		auto sum = VECTOR_EXTRACT_SCALAR<float, M1>(v_scalar_sum);

		output[out_f] = sum + bias[out_f];
	}
}