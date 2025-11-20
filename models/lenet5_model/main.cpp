#include <stdint.h>
#include <string.h>
#include "runtime.h"
#include "util.h"
#include "./kernel/kernels.h"

#ifdef SPIKE
#include <stdio.h>
#else
#include "printf.h"
#endif

// --- IMPORT DATA ---
extern "C" {
    extern float image_data[];
    // C1
    extern float c1_w[]; extern float c1_b[];
    // C2 (Split)
    extern float c2_1_w[]; extern float c2_1_b[];
    extern float c2_2_w[]; extern float c2_2_b[];
    // C3
    extern float c3_w[]; extern float c3_b[];
    // F4
    extern float f4_w[]; extern float f4_b[];
    // F5
    extern float f5_w[]; extern float f5_b[];
}

// --- BUFFER ALLOCATION ---
#define BUF_C1_SIZE 5000
#define BUF_P1_SIZE 1200
#define BUF_C2_SIZE 1800
#define BUF_P2_SIZE 500
#define COL_BUF_SIZE (80 * 1024)

// Buffers
float buf_c1[BUF_C1_SIZE] __attribute__((aligned(256)));
float buf_p1[BUF_P1_SIZE] __attribute__((aligned(256)));

// C2 Split Buffers
float buf_c2_1[BUF_C2_SIZE] __attribute__((aligned(256)));
float buf_c2_2[BUF_C2_SIZE] __attribute__((aligned(256)));
float buf_p2_1[BUF_P2_SIZE] __attribute__((aligned(256)));
float buf_p2_2[BUF_P2_SIZE] __attribute__((aligned(256)));
float buf_p2_sum[BUF_P2_SIZE] __attribute__((aligned(256)));

// C3 Out: 120
float buf_c3[256] __attribute__((aligned(256)));
// F4 Out: 84
float buf_f4[256] __attribute__((aligned(256)));
// F5 Out: 10
float buf_f5[256] __attribute__((aligned(256)));
// Softmax Out: 10
float buf_final[256] __attribute__((aligned(256)));

// Scratchpads
float col_buf[COL_BUF_SIZE] __attribute__((aligned(256)));
float gemm_buf[COL_BUF_SIZE] __attribute__((aligned(256)));


int main() {
    printf("\n=== LeNet-5 INFERENCE START ===\n");
    
    int total_cycles = 0;
    int current_cycles = 0;

    // --- LAYER 1 ---
    printf("\n[Layer 1] C1 (1->6) + P1\n");
    start_timer();
    conv2d_e32m8_direct(image_data, c1_w, buf_c1, 1, 1, 6, 32, 32, 5, 5, 1, 1, 0, 0, 0);
    bias_add_e32m8(buf_c1, c1_b, buf_c1, 1, 6, 28, 28, 1); // Fused ReLU
    maxpool_e32m8(buf_c1, buf_p1, 1, 6, 28, 28, 2, 2, 2, 2);
    stop_timer();
    current_cycles = get_timer();
    printf("Layer 1 Cycles: %d\n", current_cycles);
    total_cycles += current_cycles;

    // --- LAYER 2 ---
    printf("\n[Layer 2] C2 Split (6->16) + P2 + Add\n");
    start_timer();
    // Branch 1
    conv2d_e32m8_im2col(buf_p1, c2_1_w, c2_1_b, buf_c2_1, col_buf, gemm_buf, 6, 14, 14, 16, 5, 5, 0, 0, 1, 1, 0, 0); 
    bias_add_e32m8(buf_c2_1, c2_1_b, buf_c2_1, 1, 16, 10, 10, 1); // Bias + Fused ReLU
    maxpool_e32m8(buf_c2_1, buf_p2_1, 1, 16, 10, 10, 2, 2, 2, 2);

    // Branch 2
    conv2d_e32m8_im2col(buf_p1, c2_2_w, c2_2_b, buf_c2_2, col_buf, gemm_buf, 6, 14, 14, 16, 5, 5, 0, 0, 1, 1, 0, 0);
    bias_add_e32m8(buf_c2_2, c2_2_b, buf_c2_2, 1, 16, 10, 10, 1); // Bias + Fused ReLU
    maxpool_e32m8(buf_c2_2, buf_p2_2, 1, 16, 10, 10, 2, 2, 2, 2);

    // Tensor Add
    tensor_add_e32m8(buf_p2_1, buf_p2_2, buf_p2_sum, 16*5*5);
    stop_timer();
    current_cycles = get_timer();
    printf("Layer 2 Cycles: %d\n", current_cycles);
    total_cycles += current_cycles;

    // --- LAYER 3 ---
    printf("\n[Layer 3] C3 (16->120)\n");
    start_timer();
    conv2d_e32m8_im2col(buf_p2_sum, c3_w, c3_b, buf_c3, col_buf, gemm_buf, 16, 5, 5, 120, 5, 5, 0, 0, 1, 1, 0, 0);
    bias_add_e32m8(buf_c3, c3_b, buf_c3, 1, 120, 1, 1, 1); // Bias + ReLU
    stop_timer();
    current_cycles = get_timer();
    printf("Layer 3 Cycles: %d\n", current_cycles);
    total_cycles += current_cycles;

    // --- LAYER 4 ---
    printf("\n[Layer 4] F4 (120->84)\n");
    start_timer();
    dense_e32m8(buf_c3, f4_w, f4_b, buf_f4, 120, 84, 1); // Fused ReLU
    stop_timer();
    current_cycles = get_timer();
    printf("Layer 4 Cycles: %d\n", current_cycles);
    total_cycles += current_cycles;

    // --- LAYER 5 ---
    printf("\n[Layer 5] F5 (84->10)\n");
    start_timer();
    dense_e32m8(buf_f4, f5_w, f5_b, buf_f5, 84, 10, 0); // No ReLU
    stop_timer();
    current_cycles = get_timer();
    printf("Layer 5 Cycles: %d\n", current_cycles);
    total_cycles += current_cycles;

    // --- LAYER 6 ---
    printf("\n[Layer 6] Softmax\n");
    start_timer();
    softmax_scalar(buf_f5, buf_final, 10);
    stop_timer();
    current_cycles = get_timer();
    printf("Layer 6 Cycles: %d\n", current_cycles);
    total_cycles += current_cycles;

    printf("\n=== PREDICTION COMPLETE ===\n");
    printf("Total Inference Cycles: %d\n", total_cycles);
    printf("Class Probabilities:\n");
    for(int i=0; i<10; i++) {
        printf("Class %d: %f\n", i, buf_final[i]);
    }

    return 0;
}