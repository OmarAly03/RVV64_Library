#include <cstdlib>
extern "C" {
    #include <uart.h>
}
#include "defs_convtranspose.h"

int main(){
    // Small but realistic conv_transpose parameters
    int batch_size = 1;
    int in_channels = 2;
    int out_channels = 2;
    int input_h = 8;
    int input_w = 8;
    int kernel_h = 3;
    int kernel_w = 3;
    int stride_h = 2;
    int stride_w = 2;
    int pad_h = 1;
    int pad_w = 1;
    
    // Calculate output dimensions
    int out_height = (input_h - 1) * stride_h - 2 * pad_h + kernel_h;  // 14
    int out_width = (input_w - 1) * stride_w - 2 * pad_w + kernel_w;   // 14
    
    // Calculate array sizes
    size_t input_size = batch_size * in_channels * input_h * input_w;     // 128
    size_t kernel_size = in_channels * out_channels * kernel_h * kernel_w; // 36
    size_t output_size = batch_size * out_channels * out_height * out_width; // 392
    
    size_t start, end;
    
    // Allocate arrays
    int32_t input[input_size];
    int32_t kernel[kernel_size];
    int32_t output[output_size];
    
    uart_printf("==== Beginning Conv Transpose Benchmarking ====\n");
    uart_printf("Input: %dx%dx%dx%d, Kernel: %dx%dx%dx%d, Output: %dx%dx%dx%d\n", 
                batch_size, in_channels, input_h, input_w,
                in_channels, out_channels, kernel_h, kernel_w,
                batch_size, out_channels, out_height, out_width);
    uart_printf("Total elements - Input: %d, Kernel: %d, Output: %d\n \n", 
                input_size, kernel_size, output_size);

    // --- INITIALIZE INPUT AND KERNEL ---
    start = read_mcycle();
    
    // Initialize input with simple pattern
    for (size_t i = 0; i < input_size; i++) {
        input[i] = (int32_t)(i % 10) + 1; // Values 1-10
    }
    
    // Initialize kernel with simple pattern
    for (size_t i = 0; i < kernel_size; i++) {
        kernel[i] = (int32_t)(i % 5) + 1; // Values 1-5
    }
    
    end = read_mcycle();
    uart_printf("input/kernel initialization time: %d \n", end - start);

    // --- SCALAR VERSION ---
    start = read_mcycle();
    conv_transpose_2d_scalar(input, kernel, output, 
                           batch_size, in_channels, out_channels,
                           input_h, input_w, kernel_h, kernel_w,
                           stride_h, stride_w, pad_h, pad_w);
    end = read_mcycle();
    uart_printf("conv_transpose time scalar: %d \n", end - start);

    // --- M1 VERSION ---
    start = read_mcycle();
    conv_transpose_2d_e32m1(input, kernel, output,
                          batch_size, in_channels, out_channels,
                          input_h, input_w, kernel_h, kernel_w,
                          stride_h, stride_w, pad_h, pad_w);
    end = read_mcycle();
    uart_printf("conv_transpose time m1: %d \n", end - start);

    // --- M2 VERSION ---
    start = read_mcycle();
    conv_transpose_2d_e32m2(input, kernel, output,
                          batch_size, in_channels, out_channels,
                          input_h, input_w, kernel_h, kernel_w,
                          stride_h, stride_w, pad_h, pad_w);
    end = read_mcycle();
    uart_printf("conv_transpose time m2: %d \n", end - start);

    // --- M4 VERSION ---
    start = read_mcycle();
    conv_transpose_2d_e32m4(input, kernel, output,
                          batch_size, in_channels, out_channels,
                          input_h, input_w, kernel_h, kernel_w,
                          stride_h, stride_w, pad_h, pad_w);
    end = read_mcycle();
    uart_printf("conv_transpose time m4: %d \n", end - start);

    // --- M8 VERSION ---
    start = read_mcycle();
    conv_transpose_2d_e32m8(input, kernel, output,
                          batch_size, in_channels, out_channels,
                          input_h, input_w, kernel_h, kernel_w,
                          stride_h, stride_w, pad_h, pad_w);
    end = read_mcycle();
    uart_printf("conv_transpose time m8: %d \n", end - start);

    uart_printf("================================================= \n");

    asm volatile("ebreak");
    return 0;
}