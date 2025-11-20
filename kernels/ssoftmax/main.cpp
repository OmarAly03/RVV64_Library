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

extern "C" {
    extern uint32_t SIZE;
    extern float input_data[]; extern float golden_data[]; extern float output_data[];

    void softmax_vec(const float *i, float *o, uint64_t channels, uint64_t innerSize);
    void softmax_scalar(float* input, float* output, size_t size);
}

int verify(float* res, float* gold, size_t size) {
    int err = 0;
    for(size_t i=0; i<size; i++) {
        float diff = fabsf(res[i] - gold[i]);
        if(diff > 0.001f) {
            err++;
            if(err < 5) printf("Err @ %d: %f vs %f\n", i, res[i], gold[i]);
        }
    }
    return err;
}

int main() {
    printf("\n=== SOFTMAX [Size: %d] ===\n", SIZE);

    start_timer();
    softmax_scalar(input_data, output_data, SIZE);
    stop_timer();
    int t_s = get_timer();
    printf("Scalar: %d\n", t_s);

    memset(output_data, 0, SIZE * sizeof(float));

    // For 1D Softmax: Channels = SIZE, InnerSize = 1
    start_timer();
    softmax_vec(input_data, output_data, SIZE, 1);
    stop_timer();
    int t_v = get_timer();
    printf("Vector: %d (Speedup: %.2fx)\n", t_v, (float)t_s/t_v);

    if(verify(output_data, golden_data, SIZE) == 0) printf("PASSED\n");
    else printf("FAILED\n");

    return 0;
}