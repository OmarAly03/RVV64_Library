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
    extern uint32_t SIZE;
    extern float input_a[]; extern float input_b[];
    extern float golden_data[]; extern float output_data[];

    void tensor_add_e32m8(const float* a, const float* b, float* o, size_t size);
    void tensor_add_scalar(const float* a, const float* b, float* o, size_t size);
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
    printf("\n=== TENSOR ADD [Size: %d] ===\n", SIZE);

    start_timer();
    tensor_add_scalar(input_a, input_b, output_data, SIZE);
    stop_timer();
    int t_s = get_timer();
    printf("Scalar: %d\n", t_s);

    memset(output_data, 0, SIZE * sizeof(float));

    start_timer();
    tensor_add_e32m8(input_a, input_b, output_data, SIZE);
    stop_timer();
    int t_v = get_timer();
    printf("Vector: %d (Speedup: %.2fx)\n", t_v, (float)t_s/t_v);

    if(verify(output_data, golden_data, SIZE) == 0) printf("PASSED\n");
    else printf("FAILED\n");

    return 0;
}