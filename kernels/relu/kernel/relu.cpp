// Copyright 2020 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Author: Gemini
//
// Description: Kernels for float32 ReLU

#include <stddef.h> // For size_t
#include "/home/omar/ara/lib/rvv_defs.hpp"  // For vector intrinsics/macros

/**
 * @brief Compute ReLU on a float32 array using RISC-V Vector Extensions (e32m8)
 *
 * @param input Pointer to the input array
 * @param output Pointer to the output array
 * @param size Number of elements in the arrays
 */
void relu_e32m8(float* input, float* output, size_t size) {
    float* in_ptr = input;
    float* out_ptr = output;

    // Loop over the array, processing 'vl' elements at a time
    for (size_t cnt = size; cnt > 0; ) {
        // Set the vector length (vl) for this iteration
        size_t vl = SET_VECTOR_LENGTH<float, M8>(cnt);

        // Load a vector of input data
        auto v_input = VECTOR_LOAD<float, M8>(in_ptr, vl);
        
        // Create a vector of zeros
        auto v_zero = VECTOR_MOVE<float, M8>(0.0f, vl);
        
        // Compute the maximum (input, 0.0)
        auto v_result = VECTOR_MAX<float, M8>(v_input, v_zero, vl);
        
        // Store the result vector to the output
        VECTOR_STORE<float, M8>(out_ptr, v_result, vl);

        // Decrement the remaining count and advance pointers
        cnt -= vl;
        in_ptr += vl;
        out_ptr += vl;
    }
}

/**
 * @brief Compute ReLU on a float32 array using scalar C++
 *
 * @param input Pointer to the input array
 * @param output Pointer to the output array
 * @param size Number of elements in the arrays
 */
void relu_scalar(float* input, float* output, size_t size) {
    // Simple scalar loop
    for (size_t i = 0; i < size; i++) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}