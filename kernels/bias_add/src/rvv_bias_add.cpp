#include <cstddef>
#include <riscv_vector.h>
#include "rvv_defs.hpp"

using namespace std;

/*********************************** Scalar Version ************************************/

void bias_add_scalar(const float* input, const float* bias, float* output,
                       size_t batch_size, size_t channels,
                       size_t height, size_t width) {
    
    // Size of one 2D feature map
    size_t channel_size = height * width; 
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            // Get the scalar bias value for this channel
            float b_val = bias[c]; 
            // Calculate the starting offset for this channel
            size_t offset = (b * channels + c) * channel_size;
            
            const float* in_ptr = input + offset;
            float* out_ptr = output + offset;
            
            // This is the loop we will vectorize
            for (size_t i = 0; i < channel_size; ++i) {
                out_ptr[i] = in_ptr[i] + b_val;
            }
        }
    }
}


/********************************* Vectorized Versions *********************************/

void bias_add_e32m1(const float* input, const float* bias, float* output,
					  size_t batch_size, size_t channels,
					  size_t height, size_t width) {
	
	size_t channel_size = height * width;
	
	for (size_t b = 0; b < batch_size; ++b) {
		for (size_t c = 0; c < channels; ++c) {
			float b_val = bias[c]; // Scalar bias
			size_t offset = (b * channels + c) * channel_size;
			
			const float* in_ptr = input + offset;
			float* out_ptr = output + offset;
			
			size_t cnt = channel_size;
			size_t vl;
			
			while (cnt > 0) {
				vl = SET_VECTOR_LENGTH<float, M1>(cnt);
				
				// Load vector from input
				auto v_input = VECTOR_LOAD<float, M1>(in_ptr, vl);
				
				// Add scalar bias value to the vector
				auto v_output = VECTOR_ADD<float, M1>(v_input, b_val, vl);
				
				// Store result in output
				VECTOR_STORE<float, M1>(out_ptr, v_output, vl);
				
				in_ptr += vl;
				out_ptr += vl;
				cnt -= vl;
			}
		}
	}
}

void bias_add_e32m2(const float* input, const float* bias, float* output,
					  size_t batch_size, size_t channels,
					  size_t height, size_t width) {
	
	size_t channel_size = height * width;
	
	for (size_t b = 0; b < batch_size; ++b) {
		for (size_t c = 0; c < channels; ++c) {
			float b_val = bias[c];
			size_t offset = (b * channels + c) * channel_size;
			
			const float* in_ptr = input + offset;
			float* out_ptr = output + offset;
			
			size_t cnt = channel_size;
			size_t vl;
			
			while (cnt > 0) {
				vl = SET_VECTOR_LENGTH<float, M2>(cnt);
				auto v_input = VECTOR_LOAD<float, M2>(in_ptr, vl);
				auto v_output = VECTOR_ADD<float, M2>(v_input, b_val, vl);
				VECTOR_STORE<float, M2>(out_ptr, v_output, vl);
				
				in_ptr += vl;
				out_ptr += vl;
				cnt -= vl;
			}
		}
	}
}

void bias_add_e32m4(const float* input, const float* bias, float* output,
					  size_t batch_size, size_t channels,
					  size_t height, size_t width) {
	
	size_t channel_size = height * width;
	
	for (size_t b = 0; b < batch_size; ++b) {
		for (size_t c = 0; c < channels; ++c) {
			float b_val = bias[c];
			size_t offset = (b * channels + c) * channel_size;
			
			const float* in_ptr = input + offset;
			float* out_ptr = output + offset;
			
			size_t cnt = channel_size;
			size_t vl;
			
			while (cnt > 0) {
				vl = SET_VECTOR_LENGTH<float, M4>(cnt);
				auto v_input = VECTOR_LOAD<float, M4>(in_ptr, vl);
				auto v_output = VECTOR_ADD<float, M4>(v_input, b_val, vl);
				VECTOR_STORE<float, M4>(out_ptr, v_output, vl);
				
				in_ptr += vl;
				out_ptr += vl;
				cnt -= vl;
			}
		}
	}
}

void bias_add_e32m8(const float* input, const float* bias, float* output,
					  size_t batch_size, size_t channels,
					  size_t height, size_t width) {
	
	size_t channel_size = height * width;
	
	for (size_t b = 0; b < batch_size; ++b) {
		for (size_t c = 0; c < channels; ++c) {
			float b_val = bias[c];
			size_t offset = (b * channels + c) * channel_size;
			
			const float* in_ptr = input + offset;
			float* out_ptr = output + offset;
			
			size_t cnt = channel_size;
			size_t vl;
			
			while (cnt > 0) {
				vl = SET_VECTOR_LENGTH<float, M8>(cnt);
				auto v_input = VECTOR_LOAD<float, M8>(in_ptr, vl);
				auto v_output = VECTOR_ADD<float, M8>(v_input, b_val, vl);
				VECTOR_STORE<float, M8>(out_ptr, v_output, vl);
				
				in_ptr += vl;
				out_ptr += vl;
				cnt -= vl;
			}
		}
	}
}