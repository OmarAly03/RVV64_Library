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
    extern uint32_t IN_DIM;
    extern uint32_t OUT_DIM;
    extern float input_data[];
    extern float weights[];
    extern float bias[];
    extern float golden_data[];
    extern float output_data[];
    
    void dense_e32m8(const float* input, const float* weights, const float* bias, float* output, int in, int out);
    void dense_scalar(const float* input, const float* weights, const float* bias, float* output, int in, int out);
}

// int verify(float* res, float* gold, int size) {
//     int err = 0;
//     for(int i=0; i<size; i++) {
//         float diff = res[i] - gold[i];
//         if(diff < 0) diff = -diff;
//         if(diff > 0.001f) {
//             err++;
//             if(err < 5) printf("Mismatch @ %d: Got %f Exp %f\n", i, res[i], gold[i]);
//         }
//     }
//     return err;
// }

int main() {
    int IN = IN_DIM;
    int OUT = OUT_DIM;

    printf("\n=== DENSE LAYER [In: %d, Out: %d] ===\n", IN, OUT);

    // Scalar
    start_timer();
    dense_scalar(input_data, weights, bias, output_data, IN, OUT);
    stop_timer();
    int t_scalar = get_timer();
    printf("Scalar Cycles: %d\n", t_scalar);

    // Reset output
    memset(output_data, 0, OUT * sizeof(float));

    // Vector
    start_timer();
    dense_e32m8(input_data, weights, bias, output_data, IN, OUT);
    stop_timer();
    int t_vector = get_timer();
    printf("Vector Cycles: %d\n", t_vector);
    
    printf("Speedup: %.2fx\n", (float)t_scalar / t_vector);

    // Verify
    // if(verify(output_data, golden_data, OUT) == 0) printf("Status: PASSED\n");
    // else printf("Status: FAILED\n");

    return 0;
}