#include <stdint.h>
#include <string.h>
#include "runtime.h"
#include "util.h"
#ifdef SPIKE
#include <stdio.h>
#else
#include "printf.h"
#endif

extern "C" {
    extern uint32_t BATCH, CHANNELS, HEIGHT, WIDTH;
    extern float input_data[];
    extern float bias_data[];
    extern float golden_data[];
    extern float output_data[];

    void bias_add_e32m8(const float* i, const float* b, float* o, size_t bs, size_t c, size_t h, size_t w);
    void bias_add_scalar(const float* i, const float* b, float* o, size_t bs, size_t c, size_t h, size_t w);
}

int verify(float* res, float* gold, size_t size) {
    int err = 0;
    for(size_t i=0; i<size; i++) {
        if(res[i] != gold[i]) {
            err++;
            if(err < 5) printf("Err @ %d: %f vs %f\n", i, res[i], gold[i]);
        }
    }
    return err;
}

int main() {
    size_t total_size = BATCH * CHANNELS * HEIGHT * WIDTH;
    printf("\n=== BIAS ADD [%dx%dx%dx%d] ===\n", BATCH, CHANNELS, HEIGHT, WIDTH);

    start_timer();
    bias_add_scalar(input_data, bias_data, output_data, BATCH, CHANNELS, HEIGHT, WIDTH);
    stop_timer();
    int t_s = get_timer();
    printf("Scalar: %d\n", t_s);

    // Clear
    memset(output_data, 0, total_size * sizeof(float));

    start_timer();
    bias_add_e32m8(input_data, bias_data, output_data, BATCH, CHANNELS, HEIGHT, WIDTH);
    stop_timer();
    int t_v = get_timer();
    printf("Vector: %d (Speedup: %.2fx)\n", t_v, (float)t_s/t_v);

    if(verify(output_data, golden_data, total_size) == 0) printf("PASSED\n");
    else printf("FAILED\n");

    return 0;
}