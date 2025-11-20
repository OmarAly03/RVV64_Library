#include <riscv_vector.h>
#include <float.h>
#include <stddef.h>

// Helper for scalar fallback if needed
#define MIN(a,b) (((a)<(b))?(a):(b))

extern "C" {
    
    // =========================================================
    // Vectorized MaxPool (e32m8)
    // Strategy: Vectorize output width (ow) loop using Strided Loads
    // =========================================================
    void maxpool_e32m8(const float* input, float* output,
                       int batch, int channels,
                       int in_h, int in_w,
                       int k_h, int k_w,
                       int stride_h, int stride_w) {

        int out_h = (in_h - k_h) / stride_h + 1;
        int out_w = (in_w - k_w) / stride_w + 1;

        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channels; ++c) {
                
                const float* in_ptr_base = input + (b * channels + c) * in_h * in_w;
                float* out_ptr_base = output + (b * channels + c) * out_h * out_w;

                for (int oh = 0; oh < out_h; ++oh) {
                    
                    int h_start = oh * stride_h;
                    float* out_row = out_ptr_base + oh * out_w;

                    // Vectorize Output Width
                    for (int ow = 0; ow < out_w; ) {
                        // Calculate how many output pixels we can compute at once
                        size_t vl = __riscv_vsetvl_e32m8(out_w - ow);
                        
                        // Init accumulator with very small number
                        vfloat32m8_t v_max = __riscv_vfmv_v_f_f32m8(-FLT_MAX, vl);

                        int w_start_base = ow * stride_w;

                        // Iterate over the Kernel Window
                        for (int kh = 0; kh < k_h; ++kh) {
                            int cur_h = h_start + kh;
                            if (cur_h >= in_h) continue;

                            const float* in_row_ptr = in_ptr_base + cur_h * in_w;

                            for (int kw = 0; kw < k_w; ++kw) {
                                // Strided Load: Load pixels (0,0), (0,S), (0,2S)... relative to window
                                ptrdiff_t in_stride = stride_w * sizeof(float);
                                const float* load_addr = in_row_ptr + w_start_base + kw;
                                
                                vfloat32m8_t v_in = __riscv_vlse32_v_f32m8(load_addr, in_stride, vl);
                                
                                // Max Update
                                v_max = __riscv_vfmax_vv_f32m8(v_max, v_in, vl);
                            }
                        }

                        // Store results
                        __riscv_vse32_v_f32m8(out_row + ow, v_max, vl);
                        
                        ow += vl;
                    }
                }
            }
        }
    }
    
    // Scalar reference for debugging/comparison
    void maxpool_scalar(const float* input, float* output,
                        int batch, int channels,
                        int in_h, int in_w,
                        int k_h, int k_w,
                        int stride_h, int stride_w) {

        int out_h = (in_h - k_h) / stride_h + 1;
        int out_w = (in_w - k_w) / stride_w + 1;

        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        int h_start = oh * stride_h;
                        int w_start = ow * stride_w;
                        int h_end = MIN(h_start + k_h, in_h);
                        int w_end = MIN(w_start + k_w, in_w);
                        float max_val = -FLT_MAX;
                        for (int h = h_start; h < h_end; ++h) {
                            for (int w = w_start; w < w_end; ++w) {
                                float val = input[b * channels * in_h * in_w + c * in_h * in_w + h * in_w + w];
                                if (val > max_val) max_val = val;
                            }
                        }
                        output[b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = max_val;
                    }
                }
            }
        }
    }
}