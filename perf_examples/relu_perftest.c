#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

// Sequential ReLU
void relu_activation_sequential(float *input, int h, int w, int c, float *output) {
    int total = h * w * c;
    for (int i = 0; i < total; i++) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

// Optimized vectorized ReLU
void relu_activation(float *input, int h, int w, int c, float *output) {
    int total = h * w * c;
    int idx = 0;
    const float zero = 0.0f;
    int vl;

    // Set vector length once
    asm volatile("vsetvli %0, %1, e32, m8" : "=r"(vl) : "r"(total));

    while (idx < total) {
        asm volatile("vle32.v v0, (%0)" :: "r"(input + idx));
        asm volatile("vfmv.v.f v8, %0" :: "f"(zero));
        asm volatile("vfmax.vv v0, v0, v8");
        asm volatile("vse32.v v0, (%0)" :: "r"(output + idx));
        idx += vl;
    }
}

static inline uint64_t rdinstret(void) {
    uint64_t val;
    asm volatile("rdinstret %0" : "=r"(val));
    return val;
}

int float_eq(float a, float b, float eps) {
    return fabsf(a - b) < eps || fabsf(a - b) / fmaxf(fabsf(a), fabsf(b)) < eps;
}

int test_relu(void) {
    printf("\n===== RELU PERFORMANCE TEST =====\n");
    
    int h = 448, w = 448, c = 128;
    int iterations = 1;

    size_t size = h * w * c * sizeof(float);
    float *input = malloc(size);
    float *output_seq = malloc(size);
    float *output_vec = malloc(size);

    srand(42);
    for (size_t i = 0; i < h * w * c; i++) input[i] = (float)(rand() % 200 - 100) / 10.0f;

    // Sequential
    uint64_t start_seq = rdinstret();
    clock_t time_start_seq = clock();
    for (int i = 0; i < iterations; i++) {
        relu_activation_sequential(input, h, w, c, output_seq);
    }
    uint64_t end_seq = rdinstret();
    clock_t time_end_seq = clock();
    double time_seq = (double)(time_end_seq - time_start_seq) / CLOCKS_PER_SEC;

    // Vectorized
    uint64_t start_vec = rdinstret();
    clock_t time_start_vec = clock();
    for (int i = 0; i < iterations; i++) {
        relu_activation(input, h, w, c, output_vec);
    }
    uint64_t end_vec = rdinstret();
    clock_t time_end_vec = clock();
    double time_vec = (double)(time_end_vec - time_start_vec) / CLOCKS_PER_SEC;

    // Verify correctness
    int mismatches = 0;
    for (size_t i = 0; i < h * w * c && mismatches < 5; i++) {
        if (!float_eq(output_seq[i], output_vec[i], 1e-4)) {
            printf("ReLU Mismatch at index %zu: sequential=%.2f, vector=%.2f\n", i, output_seq[i], output_vec[i]);
            mismatches++;
        }
    }

    printf("Input: %dx%dx%d\n", h, w, c);
    printf("Sequential Execution time (%d iterations): %.3f seconds\n", iterations, time_seq);
    printf("Sequential Instructions (%d iterations): %llu\n", iterations, end_seq - start_seq);
    printf("Vectorized Execution time (%d iterations): %.3f seconds\n", iterations, time_vec);
    printf("Vectorized Instructions (%d iterations): %llu\n", iterations, end_vec - start_vec);
    printf("Speedup (Time): %.2fx\n", time_seq / time_vec);
    printf("Speedup (Instructions): %.2fx\n", (double)(end_seq - start_seq) / (end_vec - start_vec));
    printf("Correctness: %s\n", mismatches == 0 ? "PASSED" : "FAILED");

    free(input);
    free(output_seq);
    free(output_vec);
    return mismatches == 0;
}

int main(void) {
    printf("===================================\n");
    printf("RELU PERFORMANCE TEST\n");
    printf("===================================\n");

    int pass_relu = test_relu();

    printf("\n===================================\n");
    printf("SUMMARY:\n");
    printf("ReLU Test: %s\n", pass_relu ? "PASSED" : "FAILED");

    return 0;
}