#include <cstdlib>
extern "C" {
    #include <uart.h>
}
#include "defs_maxpool.h"

int main(){
    // Test parameters for maxpool
    const size_t N = 1;      // batch size
    const size_t C = 4;      // channels (reduced for memory)
    const size_t H = 16;     // input height (reduced for memory)
    const size_t W = 16;     // input width (reduced for memory)
    const size_t K = 3;      // kernel size
    const size_t S = 2;      // stride
    const bool ceil_mode = false;
    
    const size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
    const size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
    
    const size_t input_size = N * C * H * W;   // 1*4*16*16 = 1024
    const size_t output_size = N * C * OH * OW; // 1*4*7*7 = 196
    
    uart_printf("==== Beginning MaxPool Benchmarking ====\n");
    uart_printf("Input: [%d,%d,%d,%d], Kernel: %dx%d, Stride: %d\n", N, C, H, W, K, K, S);
    uart_printf("Output: [%d,%d,%d,%d] (size: %d)\n", N, C, OH, OW, output_size);
    uart_printf("Total memory: %d bytes\n", (input_size + 10*output_size) * 4);
    uart_printf("=========================================\n\n");

    // Static arrays - input
    static int32_t input[1024];           // N*C*H*W = 1*4*16*16 = 1024
    
    // Static arrays - scalar reference
    static int32_t scalar_values[196];    // N*C*OH*OW = 1*4*7*7 = 196  
    static int64_t scalar_indices[196];   // Same size as output_values
    
    // Static arrays - M1 outputs
    static int32_t m1_values[196];
    static int64_t m1_indices[196];
    
    // Static arrays - M2 outputs
    static int32_t m2_values[196];
    static int64_t m2_indices[196];
    
    // Static arrays - M4 outputs
    static int32_t m4_values[196];
    static int64_t m4_indices[196];
    
    // Static arrays - M8 outputs
    static int32_t m8_values[196];
    static int64_t m8_indices[196];

    size_t start, end;

    // --- INITIALIZE INPUT ARRAY ---
    uart_printf("Initializing input data...\n");
    start = read_mcycle();
    for (size_t i = 0; i < input_size; i++) {
        input[i] = (int32_t)(i % 200) - 100; // Range: -100 to 99
    }
    end = read_mcycle();
    uart_printf("Input initialization time: %d cycles\n\n", end - start);

    // --- SCALAR IMPLEMENTATION ---
    uart_printf("Testing scalar implementation...\n");
    start = read_mcycle();
    maxpool_scalar(input, scalar_values, scalar_indices, N, C, H, W, K, S, ceil_mode);
    end = read_mcycle();
    size_t scalar_cycles = end - start;
    uart_printf("MaxPool scalar time: %d cycles\n", scalar_cycles);

    // --- VECTOR IMPLEMENTATION M1 ---
    uart_printf("Testing vector e32m1 implementation...\n");
    start = read_mcycle();
    maxpool_e32m1(input, m1_values, m1_indices, N, C, H, W, K, S, ceil_mode);
    end = read_mcycle();
    size_t m1_cycles = end - start;
    uart_printf("MaxPool e32m1 time: %d cycles\n", m1_cycles);
    
    // Verify M1 results
    bool m1_correct = true;
    size_t m1_errors = 0;
    for (size_t i = 0; i < output_size; i++) {
        if (m1_values[i] != scalar_values[i] || m1_indices[i] != scalar_indices[i]) {
            m1_correct = false;
            m1_errors++;
            if (m1_errors <= 5) { // Show first 5 errors only
                uart_printf("Mismatch at index %d: scalar_val=%d, m1_val=%d, scalar_idx=%d, m1_idx=%d\n", 
                    i, scalar_values[i], m1_values[i], (int)scalar_indices[i], (int)m1_indices[i]);
            }
        }
    }
    
    if (m1_errors > 5) {
        uart_printf("... and %d more errors\n", m1_errors - 5);
    }
    
    uart_printf("M1 results: %s (%d/%d correct)\n", 
               m1_correct ? "CORRECT" : "INCORRECT", 
               output_size - m1_errors, output_size);

    // --- VECTOR IMPLEMENTATION M2 ---
    uart_printf("Testing vector e32m2 implementation...\n");
    start = read_mcycle();
    maxpool_e32m2(input, m2_values, m2_indices, N, C, H, W, K, S, ceil_mode);
    end = read_mcycle();
    size_t m2_cycles = end - start;
    uart_printf("MaxPool e32m2 time: %d cycles\n", m2_cycles);
    
    // Verify M2 results
    bool m2_correct = true;
    size_t m2_errors = 0;
    for (size_t i = 0; i < output_size; i++) {
        if (m2_values[i] != scalar_values[i] || m2_indices[i] != scalar_indices[i]) {
            m2_correct = false;
            m2_errors++;
            if (m2_errors <= 5) { // Show first 5 errors only
                uart_printf("Mismatch at index %d: scalar_val=%d, m2_val=%d, scalar_idx=%d, m2_idx=%d\n", 
                    i, scalar_values[i], m2_values[i], (int)scalar_indices[i], (int)m2_indices[i]);
            }
        }
    }
    
    if (m2_errors > 5) {
        uart_printf("... and %d more errors\n", m2_errors - 5);
    }
    
    uart_printf("M2 results: %s (%d/%d correct)\n", 
               m2_correct ? "CORRECT" : "INCORRECT", 
               output_size - m2_errors, output_size);

    // --- VECTOR IMPLEMENTATION M4 ---
    uart_printf("Testing vector e32m4 implementation...\n");
    start = read_mcycle();
    maxpool_e32m4(input, m4_values, m4_indices, N, C, H, W, K, S, ceil_mode);
    end = read_mcycle();
    size_t m4_cycles = end - start;
    uart_printf("MaxPool e32m4 time: %d cycles\n", m4_cycles);
    
    // Verify M4 results
    bool m4_correct = true;
    size_t m4_errors = 0;
    for (size_t i = 0; i < output_size; i++) {
        if (m4_values[i] != scalar_values[i] || m4_indices[i] != scalar_indices[i]) {
            m4_correct = false;
            m4_errors++;
            if (m4_errors <= 5) { // Show first 5 errors only
                uart_printf("Mismatch at index %d: scalar_val=%d, m4_val=%d, scalar_idx=%d, m4_idx=%d\n", 
                    i, scalar_values[i], m4_values[i], (int)scalar_indices[i], (int)m4_indices[i]);
            }
        }
    }
    
    if (m4_errors > 5) {
        uart_printf("... and %d more errors\n", m4_errors - 5);
    }
    
    uart_printf("M4 results: %s (%d/%d correct)\n", 
               m4_correct ? "CORRECT" : "INCORRECT", 
               output_size - m4_errors, output_size);

    // --- VECTOR IMPLEMENTATION M8 ---
    uart_printf("Testing vector e32m8 implementation...\n");
    start = read_mcycle();
    maxpool_e32m8(input, m8_values, m8_indices, N, C, H, W, K, S, ceil_mode);
    end = read_mcycle();
    size_t m8_cycles = end - start;
    uart_printf("MaxPool e32m8 time: %d cycles\n", m8_cycles);
    
    // Verify M8 results
    bool m8_correct = true;
    size_t m8_errors = 0;
    for (size_t i = 0; i < output_size; i++) {
        if (m8_values[i] != scalar_values[i] || m8_indices[i] != scalar_indices[i]) {
            m8_correct = false;
            m8_errors++;
            if (m8_errors <= 5) { // Show first 5 errors only
                uart_printf("Mismatch at index %d: scalar_val=%d, m8_val=%d, scalar_idx=%d, m8_idx=%d\n", 
                    i, scalar_values[i], m8_values[i], (int)scalar_indices[i], (int)m8_indices[i]);
            }
        }
    }
    
    if (m8_errors > 5) {
        uart_printf("... and %d more errors\n", m8_errors - 5);
    }
    
    uart_printf("M8 results: %s (%d/%d correct)\n", 
               m8_correct ? "CORRECT" : "INCORRECT", 
               output_size - m8_errors, output_size);

    // Print some sample outputs for verification
    uart_printf("\nSample outputs (first 8):\n");
    uart_printf("Index | Scalar Val/Idx | M1 Val/Idx | M2 Val/Idx | M4 Val/Idx | M8 Val/Idx\n");
    uart_printf("------|----------------|------------|------------|------------|------------\n");
    for (size_t i = 0; i < 8 && i < output_size; i++) {
        uart_printf("  %2d  |   %3d / %3d   |  %3d / %3d |  %3d / %3d |  %3d / %3d |  %3d / %3d\n", 
            i, 
            scalar_values[i], (int)scalar_indices[i],
            m1_values[i], (int)m1_indices[i], 
            m2_values[i], (int)m2_indices[i],
            m4_values[i], (int)m4_indices[i],
            m8_values[i], (int)m8_indices[i]);
    }

    // Performance summary
    uart_printf("\n==== Performance Summary ====\n");
    uart_printf("Input size: %d elements (%d bytes)\n", input_size, input_size * 4);
    uart_printf("Output size: %d elements (%d bytes)\n", output_size, output_size * 4);
    uart_printf("Kernel: %dx%d, Stride: %d\n", K, K, S);
    uart_printf("Scalar cycles: %d\n", scalar_cycles);
    
    uart_printf("Vector M1 cycles: %d \n", m1_cycles);
    if (m1_correct && m1_cycles > 0) {
        uart_printf(" (%.2fx speedup)", (float)scalar_cycles / (float)m1_cycles);
    } else if (!m1_correct) {
        uart_printf(" (INCORRECT)");
    }
    uart_printf("\n");
    
    uart_printf("Vector M2 cycles: %d", m2_cycles);
    if (m2_correct && m2_cycles > 0) {
        uart_printf(" (%.2fx speedup)", (float)scalar_cycles / (float)m2_cycles);
    } else if (!m2_correct) {
        uart_printf(" (INCORRECT)");
    }
    uart_printf("\n");
    
    uart_printf("Vector M4 cycles: %d", m4_cycles);
    if (m4_correct && m4_cycles > 0) {
        uart_printf(" (%.2fx speedup)", (float)scalar_cycles / (float)m4_cycles);
    } else if (!m4_correct) {
        uart_printf(" (INCORRECT)");
    }
    uart_printf("\n");
    
    uart_printf("Vector M8 cycles: %d", m8_cycles);
    if (m8_correct && m8_cycles > 0) {
        uart_printf(" (%.2fx speedup)", (float)scalar_cycles / (float)m8_cycles);
    } else if (!m8_correct) {
        uart_printf(" (INCORRECT)");
    }
    uart_printf("\n");
    
    // Determine best implementation
    size_t best_cycles = scalar_cycles;
    const char* best_impl = "scalar";
    
    if (m1_correct && m1_cycles < best_cycles) {
        best_cycles = m1_cycles;
        best_impl = "M1";
    }
    if (m2_correct && m2_cycles < best_cycles) {
        best_cycles = m2_cycles;
        best_impl = "M2";
    }
    if (m4_correct && m4_cycles < best_cycles) {
        best_cycles = m4_cycles;
        best_impl = "M4";
    }
    if (m8_correct && m8_cycles < best_cycles) {
        best_cycles = m8_cycles;
        best_impl = "M8";
    }
    
    uart_printf("\nBest implementation: %s (%d cycles, %.2fx speedup)\n", 
               best_impl, best_cycles, (float)scalar_cycles / (float)best_cycles);
    
    uart_printf("==============================\n");

    asm volatile("ebreak");
    return 0;
}