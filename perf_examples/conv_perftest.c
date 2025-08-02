#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

// Sequential convolution
void conv_layer_sequential(float *input, float *weights, float *biases,
                          int in_h, int in_w, int in_c,
                          int num_filters, int filter_size,
                          int stride, int padding,
                          float *output) {
    int out_h = (in_h + 2 * padding - filter_size) / stride + 1;
    int out_w = (in_w + 2 * padding - filter_size) / stride + 1;

    for (int oy = 0; oy < out_h; oy++) {
        for (int ox = 0; ox < out_w; ox++) {
            for (int f = 0; f < num_filters; f++) {
                float sum = 0.0f;
                for (int ky = 0; ky < filter_size; ky++) {
                    for (int kx = 0; kx < filter_size; kx++) {
                        int in_y = oy * stride + ky - padding;
                        int in_x = ox * stride + kx - padding;
                        if (in_y < 0 || in_y >= in_h || in_x < 0 || in_x >= in_w)
                            continue;
                        for (int c = 0; c < in_c; c++) {
                            float in_val = input[(in_y * in_w + in_x) * in_c + c];
                            float w_val = weights[((f * filter_size + ky) * filter_size + kx) * in_c + c];
                            sum += in_val * w_val;
                        }
                    }
                }
                sum += biases[f];
                output[(oy * out_w + ox) * num_filters + f] = sum;
            }
        }
    }
}

// Optimized vectorized convolution
void conv_layer(float *input, float *weights, float *biases,
                int in_h, int in_w, int in_c,
                int num_filters, int filter_size,
                int stride, int padding,
                float *output) {
    int out_h = (in_h + 2*padding - filter_size) / stride + 1;
    int out_w = (in_w + 2*padding - filter_size) / stride + 1;
    int vl;

    // Set vector length once per filter
    asm volatile("vsetvli %0, %1, e32, m8" : "=r"(vl) : "r"(in_c));

    for (int oy = 0; oy < out_h; oy++) {
        for (int ox = 0; ox < out_w; ox++) {
            for (int f = 0; f < num_filters; f++) {
                float sum = 0.0f;

                // Initialize accumulator
                asm volatile("vfmv.v.f v0, %0" :: "f"(0.0f));

                // Accumulate over filter window
                for (int ky = 0; ky < filter_size; ky++) {
                    for (int kx = 0; kx < filter_size; kx++) {
                        int in_y = oy * stride + ky - padding;
                        int in_x = ox * stride + kx - padding;
                        if (in_y < 0 || in_y >= in_h || in_x < 0 || in_x >= in_w)
                            continue;
                        float *in_ptr = input + (in_y * in_w + in_x) * in_c;
                        float *w_ptr = weights + ((f * filter_size + ky) * filter_size + kx) * in_c;
                        int rem = in_c;

                        while (rem > 0) {
                            asm volatile("vle32.v v8, (%0)" :: "r"(in_ptr));
                            asm volatile("vle32.v v16, (%0)" :: "r"(w_ptr));
                            asm volatile("vfmacc.vv v0, v8, v16");
                            in_ptr += vl;
                            w_ptr += vl;
                            rem -= vl;
                        }
                    }
                }

                // Reduce to scalar
                asm volatile("vfmv.v.f v8, %0" :: "f"(0.0f));
                asm volatile("vfredusum.vs v0, v0, v8");
                asm volatile("vfmv.f.s %0, v0" : "=f"(sum));

                sum += biases[f];
                output[(oy * out_w + ox) * num_filters + f] = sum;
            }
        }
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

int test_conv(void) {
    printf("\n===== CONVOLUTION PERFORMANCE TEST =====\n");
    
    int in_h = 224, in_w = 224, in_c = 256;
    int num_filters = 256, filter_size = 3, stride = 1, padding = 1;
    int out_h = (in_h + 2 * padding - filter_size) / stride + 1;
    int out_w = (in_w + 2 * padding - filter_size) / stride + 1;
    int iterations = 3;

    size_t input_size = in_h * in_w * in_c * sizeof(float);
    size_t weights_size = num_filters * filter_size * filter_size * in_c * sizeof(float);
    size_t biases_size = num_filters * sizeof(float);
    size_t output_size = out_h * out_w * num_filters * sizeof(float);

    float *input = malloc(input_size);
    float *weights = malloc(weights_size);
    float *biases = malloc(biases_size);
    float *output_seq = malloc(output_size);
    float *output_vec = malloc(output_size);

    srand(42);
    for (size_t i = 0; i < in_h * in_w * in_c; i++) input[i] = (float)(rand() % 100) / 10.0f;
    for (size_t i = 0; i < num_filters * filter_size * filter_size * in_c; i++) weights[i] = (float)(rand() % 10) / 10.0f;
    for (size_t i = 0; i < num_filters; i++) biases[i] = 0.0f;

    // Sequential
    uint64_t start_seq = rdinstret();
    clock_t time_start_seq = clock();
    for (int i = 0; i < iterations; i++) {
        conv_layer_sequential(input, weights, biases, in_h, in_w, in_c, num_filters, filter_size, stride, padding, output_seq);
    }
    uint64_t end_seq = rdinstret();
    clock_t time_end_seq = clock();
    double time_seq = (double)(time_end_seq - time_start_seq) / CLOCKS_PER_SEC;

    // Vectorized
    uint64_t start_vec = rdinstret();
    clock_t time_start_vec = clock();
    for (int i = 0; i < iterations; i++) {
        conv_layer(input, weights, biases, in_h, in_w, in_c, num_filters, filter_size, stride, padding, output_vec);
    }
    uint64_t end_vec = rdinstret();
    clock_t time_end_vec = clock();
    double time_vec = (double)(time_end_vec - time_start_vec) / CLOCKS_PER_SEC;

    // Verify correctness
    int mismatches = 0;
    for (size_t i = 0; i < out_h * out_w * num_filters && mismatches < 5; i++) {
        if (!float_eq(output_seq[i], output_vec[i], 1e-4)) {
            printf("Conv Mismatch at index %zu: sequential=%.2f, vector=%.2f\n", i, output_seq[i], output_vec[i]);
            mismatches++;
        }
    }

    printf("Input: %dx%dx%d, Filters: %d, Filter Size: %dx%d, Stride: %d, Padding: %d\n",
           in_h, in_w, in_c, num_filters, filter_size, filter_size, stride, padding);
    printf("Output: %dx%dx%d\n", out_h, out_w, num_filters);
    printf("Sequential Execution time (%d iterations): %.3f seconds\n", iterations, time_seq);
    printf("Sequential Instructions (%d iterations): %llu\n", iterations, end_seq - start_seq);
    printf("Vectorized Execution time (%d iterations): %.3f seconds\n", iterations, time_vec);
    printf("Vectorized Instructions (%d iterations): %llu\n", iterations, end_vec - start_vec);
    printf("Speedup (Time): %.2fx\n", time_seq / time_vec);
    printf("Speedup (Instructions): %.2fx\n", (double)(end_seq - start_seq) / (end_vec - start_vec));
    printf("Correctness: %s\n", mismatches == 0 ? "PASSED" : "FAILED");

    free(input);
    free(weights);
    free(biases);
    free(output_seq);
    free(output_vec);
    return mismatches == 0;
}

int main(void) {
    printf("===================================\n");
    printf("CONVOLUTION PERFORMANCE TEST\n");
    printf("===================================\n");

    int pass_conv = test_conv();

    printf("\n===================================\n");
    printf("SUMMARY:\n");
    printf("Convolution Test: %s\n", pass_conv ? "PASSED" : "FAILED");

    return 0;
}