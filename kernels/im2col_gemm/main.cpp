#include <stdint.h>
#include <string.h>
#include <math.h>
#include "runtime.h"
#include "util.h"

#ifdef SPIKE
#include <stdio.h>
#else
#include "printf.h"
#endif

#define THRESHOLD 0.001f

extern "C" {
    extern uint64_t C_IN;
    extern uint64_t H_IN;
    extern uint64_t W_IN;
    extern uint64_t M_OUT;
    extern uint64_t K_H;
    extern uint64_t K_W;
    extern uint64_t PAD;
    extern uint64_t STRIDE;
    extern uint64_t HAS_BIAS;
    extern uint64_t CHK_COL_SIZE;
    extern uint64_t CHK_OUT_SIZE;

    extern float input_data[];
    extern float weights[];
    extern float bias[];
    extern float golden[];
    extern float output_data[];
}

// Scratchpad
#define MAX_COL_ELEMENTS (80 * 1024)

// 32k floats * 4 bytes = 128 KB
#define MAX_GEMM_ACC_ELEMENTS (32 * 1024)

float col_buffer[MAX_COL_ELEMENTS] __attribute__((aligned(256)));
float gemm_buffer[MAX_GEMM_ACC_ELEMENTS] __attribute__((aligned(256)));

// Declarations
void conv2d_scalar(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w);

void conv2d_e32m1(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w);

void conv2d_e32m8(
	const float* input, const float* kernel, float* output,
	int batch_size, int in_channels, int out_channels,
	int input_h, int input_w, int kernel_h, int kernel_w,
	int stride_h, int stride_w, int pad_h, int pad_w);

void conv2d_im2col_gemm_scalar(
    const float* input, const float* weights, const float* bias,
    float* output,
    float* col_buf, float* gemm_buf,
    int C, int H, int W, int M, int KH, int KW,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int has_bias);

void conv2d_im2col_gemm_vector(
    const float* input, const float* weights, const float* bias,
    float* output,
    float* col_buf, float* gemm_buf,
    int C, int H, int W, int M, int KH, int KW,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int has_bias);

void im2col_e32m8(const float* data_im, float* data_col,
	int channels, int height, int width,
	int kernel_h, int kernel_w,
	int pad_h, int pad_w,
	int stride_h, int stride_w);

	void conv2d_im2col_gemm_m8(
		const float* input, const float* kernel, const float* bias,
		float* output,
		float* col_buf, float* gemm_buf,
		int in_channels, int input_h, int input_w, 
		int out_channels, int kernel_h, int kernel_w,
		int pad_h, int pad_w, int stride_h, int stride_w,
		int has_bias);

// int verify(float* result, float* gold, size_t size) {
//     for (size_t i = 0; i < size; ++i) {
//         if (fabsf(result[i] - gold[i]) > THRESHOLD) {
//             return i == 0 ? -1 : (int)i;
//         }
//     }
//     return 0;
// }

int main() {
    printf("\n=== CONV2D BENCHMARK SUITE ===\n");
    printf("In: [%ld, %ld, %ld], Kern: [%ld, %ld, %ld, %ld]\n", 
           C_IN, H_IN, W_IN, M_OUT, C_IN, K_H, K_W);

    // Check Scratchpad size
    if (CHK_COL_SIZE > MAX_COL_ELEMENTS || CHK_OUT_SIZE > MAX_GEMM_ACC_ELEMENTS) {
        printf("ERROR: Scratchpad buffer too small!\n");
        return 1;
    }

    int runtime;

    // 1. Direct Scalar
    // printf("\n--- Direct Scalar ---\n");
    // start_timer();
    // conv2d_scalar(input_data, weights, output_data, 1, C_IN, M_OUT, H_IN, W_IN, K_H, K_W, STRIDE, STRIDE, PAD, PAD);
    // stop_timer();
    // runtime = get_timer();
    // printf("Cycles: %d\n", runtime);

    // 2. Direct Vector (M1)
    // printf("\n--- Direct Vector (M1) ---\n");
    // start_timer();
    // conv2d_e32m1(input_data, weights, output_data, 1, C_IN, M_OUT, H_IN, W_IN, K_H, K_W, STRIDE, STRIDE, PAD, PAD);
    // stop_timer();
    // runtime = get_timer();
    // printf("Cycles: %d\n", runtime);

	printf("\n--- Direct Vector (M8) ---\n");
    start_timer();
    conv2d_e32m8(input_data, weights, output_data, 1, C_IN, M_OUT, H_IN, W_IN, K_H, K_W, STRIDE, STRIDE, PAD, PAD);
    stop_timer();
    runtime = get_timer();
    printf("Cycles: %d\n", runtime);

    // 3. Im2Col Scalar
    // printf("\n--- Im2Col + GEMM Scalar ---\n");
    // start_timer();
    // conv2d_im2col_gemm_scalar(input_data, weights, bias, output_data, col_buffer, gemm_buffer,
    //                           C_IN, H_IN, W_IN, M_OUT, K_H, K_W, PAD, PAD, STRIDE, STRIDE, HAS_BIAS);
    // stop_timer();
    // runtime = get_timer();
    // printf("Cycles: %d\n", runtime);

    // 4. Im2Col Vector
    // printf("\n--- Im2Col + GEMM Vector (M8) ---\n");
    // start_timer();
    // conv2d_im2col_gemm_vector(input_data, weights, bias, output_data, col_buffer, gemm_buffer,
    //                           C_IN, H_IN, W_IN, M_OUT, K_H, K_W, PAD, PAD, STRIDE, STRIDE, HAS_BIAS);
    // stop_timer();
    // runtime = get_timer();
    // printf("Cycles: %d\n", runtime);

	printf("\n--- Im2Col + GEMM Vector (M8) ---\n");
    start_timer();
    conv2d_im2col_gemm_m8(input_data, weights, bias, output_data, col_buffer, gemm_buffer,
                              C_IN, H_IN, W_IN, M_OUT, K_H, K_W, PAD, PAD, STRIDE, STRIDE, HAS_BIAS);
    stop_timer();
    runtime = get_timer();
    printf("Cycles: %d\n", runtime);

	
    // Verify Final Result (Im2Col Vector)
    // printf("\nVerifying Final Result...\n");
    // int error = verify(output_data, golden, CHK_OUT_SIZE);
    // if (error != 0) {
    //     int idx = error == -1 ? 0 : error;
    //     printf("FAIL: Error at index %d. Exp %f, Got %f\n", idx, golden[idx], output_data[idx]);
    //     return 1;
    // }
    // printf("PASSED.\n");
    return 0;
}