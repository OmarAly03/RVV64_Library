#include <cstdlib>
extern "C" {
	#include <uart.h>
}
#include "defs_matmul.h"

int main(){
	size_t M, N, K; // defaults
	M = N = K = 16;
	size_t start, end;
	int32_t A[M * K], B[K * N], C[M * N];
	uart_printf("==== Beginning Benchmarking (M = %d, N = %d, K = %d)====\n \n", M, N, K);

	// --- INITIALIZE MATRICES ---
	start = read_mcycle();
	for (size_t i = 0; i < M * K; i++) {
		A[i] = 2;
	}
	for (size_t i = 0; i < K * N; i++) {
		B[i] = 3;
	}

	end = read_mcycle();
	uart_printf("matrices initialization time: %d \n", end - start);

	start = read_mcycle();
	matmul_scalar(A, B, C, M, N, K);
	end = read_mcycle();
	uart_printf("matmul time scalar: %d \n", end - start);

	start = read_mcycle();
	matmul_e32m1(A, B, C, M, N, K);
	end = read_mcycle();
	uart_printf("matmul time m1: %d \n", end - start);

	start = read_mcycle();
	matmul_e32m2(A, B, C, M, N, K);
	end = read_mcycle();
	uart_printf("matmul time m2: %d \n", end - start);

	start = read_mcycle();
    matmul_e32m4(A, B, C, M, N, K);
	end = read_mcycle();
	uart_printf("matmul time m4: %d \n", end - start);

	start = read_mcycle();
	matmul_e32m8(A, B, C, M, N, K);
	end = read_mcycle();
	uart_printf("matmul time m8: %d \n", end - start);

	uart_printf("================================================= \n");

	asm volatile("ebreak");
	return 0;
}