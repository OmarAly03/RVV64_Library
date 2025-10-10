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
    uart_printf("Total memory: %d bytes\n", (input_size + 2*output_size) * 4);
    uart_printf("=========================================\n\n");

    // Static arrays
    static int32_t input[1024];           // N*C*H*W = 1*4*16*16 = 1024
    static int32_t output_values[196];    // N*C*OH*OW = 1*4*7*7 = 196  
    static int32_t output_indices[196];   // Same size as output_values
    static int32_t scalar_values[196];    // For verification
    static int32_t scalar_indices[196];   // For verification

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
    maxpool_scalar(input, output_values, output_indices, N, C, H, W, K, S, ceil_mode);
    end = read_mcycle();
    size_t scalar_cycles = end - start;
    uart_printf("MaxPool scalar time: %d cycles\n", scalar_cycles);
    
    // Save scalar results for verification
    for (size_t i = 0; i < output_size; i++) {
        scalar_values[i] = output_values[i];
        scalar_indices[i] = output_indices[i];
    }

    // --- VECTOR IMPLEMENTATION M1 ---
    uart_printf("Testing vector e32m1 implementation...\n");
    // Clear output arrays
    for (size_t i = 0; i < output_size; i++) {
        output_values[i] = 0;
        output_indices[i] = 0;
    }
    
    start = read_mcycle();
    maxpool_e32m1(input, output_values, output_indices, N, C, H, W, K, S, ceil_mode);
    end = read_mcycle();
    size_t vector_cycles = end - start;
    uart_printf("MaxPool e32m1 time: %d cycles\n", vector_cycles);
    
    // Verify results
    bool m1_correct = true;
    size_t errors = 0;
    for (size_t i = 0; i < output_size; i++) {
        if (output_values[i] != scalar_values[i] || output_indices[i] != scalar_indices[i]) {
            m1_correct = false;
            errors++;
            if (errors <= 5) { // Show first 5 errors only
                uart_printf("Mismatch at index %d: scalar_val=%d, m1_val=%d, scalar_idx=%d, m1_idx=%d\n", 
                           i, scalar_values[i], output_values[i], scalar_indices[i], output_indices[i]);
            }
        }
    }
    
    if (errors > 5) {
        uart_printf("... and %d more errors\n", errors - 5);
    }
    
    uart_printf("M1 results: %s (%d/%d correct)\n", 
               m1_correct ? "CORRECT" : "INCORRECT", 
               output_size - errors, output_size);

    // Print some sample outputs for verification
    uart_printf("\nSample outputs (first 8):\n");
    uart_printf("Index | Scalar Val/Idx | Vector Val/Idx\n");
    uart_printf("------|----------------|---------------\n");
    for (size_t i = 0; i < 8 && i < output_size; i++) {
        uart_printf("  %2d  |   %3d / %3d   |   %3d / %3d\n", 
                   i, scalar_values[i], scalar_indices[i], 
                   output_values[i], output_indices[i]);
    }

    // Performance summary
    uart_printf("\n==== Performance Summary ====\n");
    uart_printf("Input size: %d elements (%d bytes)\n", input_size, input_size * 4);
    uart_printf("Output size: %d elements (%d bytes)\n", output_size, output_size * 4);
    uart_printf("Kernel: %dx%d, Stride: %d\n", K, K, S);
    uart_printf("Scalar cycles: %d\n", scalar_cycles);
    uart_printf("Vector cycles: %d\n", vector_cycles);
    
    if (m1_correct && vector_cycles > 0) {
        uart_printf("Speedup: %.2fx\n", (float)scalar_cycles / (float)vector_cycles);
        uart_printf("Efficiency: %s\n", vector_cycles < scalar_cycles ? "GOOD" : "NEEDS OPTIMIZATION");
    } else if (!m1_correct) {
        uart_printf("Cannot calculate speedup - vector implementation has errors\n");
    }
    
    uart_printf("==============================\n");

    asm volatile("ebreak");
    return 0;
}