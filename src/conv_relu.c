#include <stddef.h>

void conv_layer(float *input, float *weights, float *biases,
                int in_h, int in_w, int in_c,
                int num_filters, int filter_size,
                int stride, int padding,
                float *output) {
    int out_h = (in_h + 2*padding - filter_size) / stride + 1;
    int out_w = (in_w + 2*padding - filter_size) / stride + 1;

    for (int oy = 0; oy < out_h; oy++) {
        for (int ox = 0; ox < out_w; ox++) {
            for (int f = 0; f < num_filters; f++) {
                float sum;
                int vl;
                // Initialize vector accumulator v3 to zero across channels
                int rem_init = in_c;
                while (rem_init > 0) {
                    asm volatile("vsetvli %0, %1, e32, m1" : "=r"(vl) : "r"(rem_init));
                    asm volatile("vfmv.v.f v3, %0" :: "f"(0.0f));
                    rem_init -= vl;
                }

                // Accumulate over filter window
                for (int ky = 0; ky < filter_size; ky++) {
                    for (int kx = 0; kx < filter_size; kx++) {
                        int in_y = oy * stride + ky - padding;
                        int in_x = ox * stride + kx - padding;
                        if (in_y < 0 || in_y >= in_h || in_x < 0 || in_x >= in_w)
                            continue;
                        float *in_ptr = input + (in_y * in_w + in_x) * in_c;
                        float *w_ptr  = weights + ((f * filter_size + ky) * filter_size + kx) * in_c;
                        int rem = in_c;
                        while (rem > 0) {
                            asm volatile("vsetvli %0, %1, e32, m1" : "=r"(vl) : "r"(rem));
                            asm volatile("vle32.v v0, (%0)" :: "r"(in_ptr));
                            asm volatile("vle32.v v1, (%0)" :: "r"(w_ptr));
                            // fused multiply-accumulate into v3
                            asm volatile("vfmacc.vv v3, v0, v1");
                            in_ptr += vl;
                            w_ptr  += vl;
                            rem    -= vl;
                        }
                    }
                }
                // reduce accumulator v3 to scalar sum
                asm volatile("vfmv.v.f v0, %0" :: "f"(0.0f));
                asm volatile("vfredusum.vs v3, v3, v0");
                asm volatile("vfmv.f.s %0, v3" : "=f"(sum));
                // Add bias once per filter output
                sum += biases[f];

                output[(oy * out_w + ox) * num_filters + f] = sum;
            }
        }
    }
}

void relu_activation(float *input, int h, int w, int c, float *output){
    int total = h * w * c;
    int idx = 0;
    const float zero = 0.0f;
    while (idx < total) {
        int vl;
        asm volatile("vsetvli %0, %1, e32, m1" : "=r"(vl) : "r"(total - idx));
        asm volatile("vle32.v v0, (%0)" :: "r"(input + idx));
        // broadcast zero and compute max
        asm volatile("vfmv.v.f v1, %0" :: "f"(zero));
        asm volatile("vfmax.vv v0, v0, v1");
        asm volatile("vse32.v v0, (%0)" :: "r"(output + idx));
        idx += vl;
    }
}