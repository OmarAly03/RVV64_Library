#include <cstdlib>
extern "C" {
    #include <uart.h>
}
#include "defs_relu.h"

int main(){
    size_t size = 31000; // array size for ReLU testing
    size_t start, end;
    int32_t input[size], output[size];
    uart_printf("==== Beginning ReLU Benchmarking (size = %d) ====\n \n", size);

    // --- INITIALIZE INPUT ARRAY ---
    start = read_mcycle();
    for (size_t i = 0; i < size; i++) {
        input[i] = (int32_t)(i % 100) - 50; // Mix of positive and negative values
    }
    end = read_mcycle();
    uart_printf("input array initialization time: %d \n", end - start);

    start = read_mcycle();
    relu_scalar(input, output, size);
    end = read_mcycle();
    uart_printf("relu time scalar: %d \n", end - start);

    start = read_mcycle();
    relu_e32m1(input, output, size);
    end = read_mcycle();
    uart_printf("relu time m1: %d \n", end - start);

    start = read_mcycle();
    relu_e32m2(input, output, size);
    end = read_mcycle();
    uart_printf("relu time m2: %d \n", end - start);

    start = read_mcycle();
    relu_e32m4(input, output, size);
    end = read_mcycle();
    uart_printf("relu time m4: %d \n", end - start);

    start = read_mcycle();
    relu_e32m8(input, output, size);
    end = read_mcycle();
    uart_printf("relu time m8: %d \n", end - start);

    uart_printf("================================================= \n");

    asm volatile("ebreak");
    return 0;
}