#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Sequential softmax
void softmax_sequential(float *x, float *result, size_t n) {
    float max_val = -INFINITY;
    for (size_t i = 0; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum_exp = 0.0f;
    for (size_t i = 0; i < n; i++) {
        result[i] = expf(x[i] - max_val);
        sum_exp += result[i];
    }
    for (size_t i = 0; i < n; i++) {
        result[i] /= sum_exp;
    }
}

// Vectorized softmax (RVV)
void softmax_rvv(float *x, float *result, size_t n) {
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    size_t vl;

    // Phase 1: Find global maximum
    for (size_t i = 0; i < n; i += vl) {
        float chunk_max;
        asm volatile (
            "vsetvli %0, %3, e32, m1, ta, ma\n"
            "vle32.v v0, (%2)\n"
            "vfmv.s.f v1, %4\n"
            "vfredmax.vs v1, v0, v1\n"
            "vfmv.f.s %1, v1\n"
            : "=&r"(vl), "=&f"(chunk_max)
            : "r"(x + i), "r"(n - i), "f"(-INFINITY)
            : "v0", "v1"
        );
        if (chunk_max > max_val) max_val = chunk_max;
    }

    // Phase 2: Subtract max and compute exp
    for (size_t i = 0; i < n; i += vl) {
        asm volatile (
            "vsetvli %0, %4, e32, m1, ta, ma\n"
            "vle32.v v0, (%2)\n"
            "vfmv.v.f v1, %3\n"
            "vfsub.vv v0, v0, v1\n"
            "vse32.v v0, (%1)\n"
            : "=&r"(vl)
            : "r"(result + i), "r"(x + i), "f"(max_val), "r"(n - i)
            : "v0", "v1"
        );
        for (size_t j = 0; j < vl; j++) {
            result[i + j] = expf(result[i + j]);
        }
    }

    // Phase 3: Sum exponentials
    for (size_t i = 0; i < n; i += vl) {
        float chunk_sum;
        float zero = 0.0f;
        asm volatile (
            "vsetvli %0, %4, e32, m1, ta, ma\n"
            "vle32.v v0, (%2)\n"
            "vfmv.s.f v1, %3\n"
            "vfredusum.vs v1, v0, v1\n"
            "vfmv.f.s %1, v1\n"
            : "=&r"(vl), "=&f"(chunk_sum)
            : "r"(result + i), "f"(zero), "r"(n - i)
            : "v0", "v1"
        );
        sum_exp += chunk_sum;
    }

    // Phase 4: Normalize
    for (size_t i = 0; i < n; i += vl) {
        asm volatile (
            "vsetvli %0, %4, e32, m1, ta, ma\n"
            "vle32.v v0, (%2)\n"
            "vfmv.v.f v1, %3\n"
            "vfdiv.vv v0, v0, v1\n"
            "vse32.v v0, (%1)\n"
            : "=&r"(vl)
            : "r"(result + i), "r"(result + i), "f"(sum_exp), "r"(n - i)
            : "v0", "v1"
        );
    }
}

static inline uint64_t rdinstret(void) {
    uint64_t val;
    asm volatile("rdinstret %0" : "=r"(val));
    return val;
}

int float_eq(float a, float b) {
    return fabs(a - b) < 1e-5;
}

int test_softmax(void) {
    printf("\n===== SOFTMAX PERFORMANCE TEST =====\n");
    
    size_t n = (1<<17);
    size_t iterations = 1;

    size_t size = n * sizeof(float);
    float *x = malloc(size);
    float *result_seq = malloc(size);
    float *result_vec = malloc(size);

    srand(42);
    for (size_t i = 0; i < n; i++) x[i] = (float)(rand() % 10);

    // Sequential
    uint64_t start_seq = rdinstret();
    clock_t time_start_seq = clock();
    for (size_t i = 0; i < iterations; i++) {
        softmax_sequential(x, result_seq, n);
    }
    uint64_t end_seq = rdinstret();
    clock_t time_end_seq = clock();
    double time_seq = (double)(time_end_seq - time_start_seq) / CLOCKS_PER_SEC;

    // Vectorized
    uint64_t start_vec = rdinstret();
    clock_t time_start_vec = clock();
    for (size_t i = 0; i < iterations; i++) {
        softmax_rvv(x, result_vec, n);
    }
    uint64_t end_vec = rdinstret();
    clock_t time_end_vec = clock();
    double time_vec = (double)(time_end_vec - time_start_vec) / CLOCKS_PER_SEC;

    // Verify correctness
    int mismatches = 0;
    for (size_t i = 0; i < n && mismatches < 5; i++) {
        if (!float_eq(result_seq[i], result_vec[i])) {
            printf("Softmax Mismatch at index %zu: seq=%f, vec=%f\n", i, result_seq[i], result_vec[i]);
            mismatches++;
        }
    }

    printf("Input size: %zu\n", n);
    printf("Sequential Time (%zu iter): %.3f s\n", iterations, time_seq);
    printf("Sequential Instr (%zu iter): %llu\n", iterations, end_seq - start_seq);
    printf("Vectorized Time (%zu iter): %.3f s\n", iterations, time_vec);
    printf("Vectorized Instr (%zu iter): %llu\n", iterations, end_vec - start_vec);
    printf("Speedup (Time): %.2fx\n", time_seq / time_vec);
    printf("Speedup (Instr): %.2fx\n", (double)(end_seq - start_seq) / (end_vec - start_vec));
    printf("Correctness: %s\n", mismatches == 0 ? "PASSED" : "FAILED");

    free(x);
    free(result_seq);
    free(result_vec);
    return mismatches == 0;
}

int main(void) {
    printf("===================================\n");
    printf("SOFTMAX PERFORMANCE TEST\n");
    printf("===================================\n");

    int pass_softmax = test_softmax();

    printf("\n===================================\n");
    printf("SUMMARY:\n");
    printf("Softmax Test: %s\n", pass_softmax ? "PASSED" : "FAILED");

    return 0;
}