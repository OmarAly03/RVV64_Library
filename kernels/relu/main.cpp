// Copyright 2020 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may not obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Author: Matheus Cavalcante, ETH Zurich
//         Samuel Riedel, ETH Zurich
// Adapted for ReLU by Gemini

#include <stdint.h>
#include <string.h>
#include <math.h> // For fabsf

#include "runtime.h"
#include "util.h"

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

// Define the data type
#define DTYPE_FLOAT32 2
#define DTYPE DTYPE_FLOAT32
typedef float _DTYPE;

// Error threshold for verification
#define THRESHOLD 0.00001f

// Forward declarations for ReLU kernels
void relu_e32m8(float* input, float* output, size_t size);
void relu_scalar(float* input, float* output, size_t size);

// Define the kernel to benchmark
// #define _KERNEL relu_scalar

/**
 * @brief Verification function for ReLU
 *
 * @param result The output array from the kernel
 * @param golden The golden reference array
 * @param M Number of rows
 * @param N Number of columns
 * @return int 0 if passed, otherwise the index of the first mismatch (or -1)
 */
int relu_verify(_DTYPE *result, _DTYPE *golden, uint64_t M, uint64_t N) {
  uint64_t size = M * N;
  for (uint64_t i = 0; i < size; ++i) {
    if (fabsf(result[i] - golden[i]) > THRESHOLD) {
      // Return index of error. Add 1 to distinguish from 0 (success).
      // Use -1 for index 0 to distinguish from 0 (success).
      return i == 0 ? -1 : (int)i;
    }
  }
  return 0; // Success
}

#define _VERIFY relu_verify

// Define Matrix dimensions:
// G = ReLU(A) with A=[MxN], G=[MxN]
extern uint64_t M;
extern uint64_t N;

// Data arrays from gen_data.py
extern _DTYPE a[] __attribute__((aligned(32 * NR_LANES), section(".l2"))); // Input
extern _DTYPE c[] __attribute__((aligned(32 * NR_LANES), section(".l2"))); // Output
// extern _DTYPE g[] __attribute__((aligned(32 * NR_LANES), section(".l2"))); // Golden

int main() {
  printf("\n");
  printf("===========\n");
  printf("=  ReLU   =\n");
  printf("===========\n");
  printf("\n");
  printf("\n");

  printf("\n");
  printf("------------------------------------------------------------\n");
  printf("Calculating a (%d x %d) element-wise ReLU...\n", M, N);
  printf("------------------------------------------------------------\n");
  printf("\n");

  uint64_t size = M * N;

  // Matrices are initialized --> Start calculating
  printf("Calculating Scalar ReLU...\n");
//   // Warm-up run
//   int unsigned loop_cont = 1;
//   do {
//     _KERNEL(a, c, size);
//   } while (--loop_cont != 0);

  // Timed run
  start_timer();
  relu_scalar(a, c, size);
  stop_timer();

  // Metrics
  int runtime = get_timer();
  // Performance in Elements/cycle
//   float performance = (float)size / runtime;
  // Utilization assumes 1 element per cycle is 100% for a single lane
  // Adjust (1.0 * NR_LANES) if your architecture's peak is different
//   float utilization = 100 * performance / (1.0 * NR_LANES);

  printf("The execution took %d cycles.\n", runtime);
//   printf("The performance is %f ELEM/cycle (%f%% utilization).\n", performance,
//          utilization);

  // Verify the result
//   printf("Verifying result...\n");
//   int error = _VERIFY(c, g, M, N);
//   if (error != 0) {
//     unsigned int idx = error == -1 ? 0 : error;
//     printf("Error code %d at index %d\n", error, idx);
//     printf("c[%d] = %f\n", idx, c[idx]);
//     printf("g[%d] = %f\n", idx, g[idx]);
//     return 1;
//   } else {
//     printf("Passed.\n");
//   }

    // Matrices are initialized --> Start calculating
	printf("Calculating Vectorized ReLU...\n");
	// // Warm-up run
	// int unsigned loop_cont = 1;
	// do {
	//   _KERNEL(a, c, size);
	// } while (--loop_cont != 0);
  
	// Timed run
	start_timer();
	relu_e32m8(a, c, size);
	stop_timer();
  
	// Metrics
	runtime = get_timer();
	// Performance in Elements/cycle
	// performance = (float)size / runtime;
	// Utilization assumes 1 element per cycle is 100% for a single lane
	// Adjust (1.0 * NR_LANES) if your architecture's peak is different
	// utilization = 100 * performance / (1.0 * NR_LANES);
  
	printf("The execution took %d cycles.\n", runtime);
	// printf("The performance is %f ELEM/cycle (%f%% utilization).\n", performance,
	// 	   utilization);
  
	// Verify the result
	// printf("Verifying result...\n");
	// error = _VERIFY(c, g, M, N);
	// if (error != 0) {
	//   unsigned int idx = error == -1 ? 0 : error;
	//   printf("Error code %d at index %d\n", error, idx);
	//   printf("c[%d] = %f\n", idx, c[idx]);
	//   printf("g[%d] = %f\n", idx, g[idx]);
	//   return 1;
	// } else {
	//   printf("Passed.\n");
	// }

  return 0;
}