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

// External Data from data.S
extern "C" {
    extern uint32_t C_IN;
    extern uint32_t H_IN;
    extern uint32_t W_IN;
    extern uint32_t K_SIZE;
    extern uint32_t STRIDE;
    
    extern float input_data[];
    extern float golden_data[];
}

// Function Prototypes
extern "C" void maxpool_e32m8(const float* input, float* output, int batch, int c, int h, int w, int kh, int kw, int sh, int sw);
extern "C" void maxpool_scalar(const float* input, float* output, int batch, int c, int h, int w, int kh, int kw, int sh, int sw);

// Output Buffer (Make it large enough for LeNet Pool1)
// Pool1 Output: 6 * 14 * 14 = 1176 floats. 10k is plenty.
#define MAX_OUT_ELEMENTS (10 * 1024)
float output_data[MAX_OUT_ELEMENTS] __attribute__((aligned(256)));

int verify(float* res, float* gold, int size) {
    int err = 0;
    for(int i=0; i<size; i++) {
        if(res[i] != gold[i]) {
            err++;
            if(err < 5) printf("Mismatch @ %d: Got %f Exp %f\n", i, res[i], gold[i]);
        }
    }
    return err;
}

int main() {
    int C = C_IN;
    int H = H_IN;
    int W = W_IN;
    int K = K_SIZE;
    int S = STRIDE;
    
    int out_h = (H - K) / S + 1;
    int out_w = (W - K) / S + 1;
    int out_size = C * out_h * out_w;

    printf("\n=== MAXPOOL BENCHMARK ===\n");
    printf("Shape: [%d, %d, %d] Pool: %dx%d Stride: %d\n", C, H, W, K, K, S);

    // 1. Run Scalar (Optional, for cycle comparison)
    start_timer();
    maxpool_scalar(input_data, output_data, 1, C, H, W, K, K, S, S);
    stop_timer();
    int t_scalar = get_timer();
    printf("Scalar Cycles: %d\n", t_scalar);

    // Clear Buffer
    memset(output_data, 0, out_size * sizeof(float));

    // 2. Run Vector M8
    start_timer();
    maxpool_e32m8(input_data, output_data, 1, C, H, W, K, K, S, S);
    stop_timer();
    int t_vector = get_timer();
    printf("Vector Cycles: %d\n", t_vector);
    
    // printf("Speedup: %.2fx\n", (float)t_scalar / (float)t_vector);

    // 3. Verify
    // printf("Verifying...\n");
    // int errors = verify(output_data, golden_data, out_size);
    // if(errors == 0) printf("PASSED.\n");
    // else printf("FAILED with %d errors.\n", errors);

    return 0;
}