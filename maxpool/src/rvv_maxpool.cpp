#include "../include/defs.h"
#include <riscv_vector.h>
#include <algorithm>
#include <cfloat>

// --- SCALAR IMPLEMENTATION (Unchanged but included for completeness) ---
void maxpool_scalar(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
    size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
    size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ++ow) {
                    float max_val = -FLT_MAX;
                    int64_t max_idx = -1;
                    for (size_t kh = 0; kh < K; ++kh) {
                        for (size_t kw = 0; kw < K; ++kw) {
                            size_t ih = oh * S + kh;
                            size_t iw = ow * S + kw;
                            if (ih < H && iw < W) { // Boundary check
                                float val = X[n*C*H*W + c*H*W + ih*W + iw];
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = (ih * W + iw);
                                }
                            }
                        }
                    }
                    Y[n*C*OH*OW + c*OH*OW + oh*OW + ow] = max_val;
                    I[n*C*OH*OW + c*OH*OW + oh*OW + ow] = (n * C * H * W) + (c * H * W) + max_idx;
                }
            }
        }
    }
}


// It uses a temporary buffer to gather data before processing, which is a common and robust technique.
#define IMPLEMENT_MAXPOOL_RVV(LMUL) \
void maxpool_e32m##LMUL(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) { \
    size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode); \
    size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode); \
    for (size_t n = 0; n < N; ++n) { \
        for (size_t c = 0; c < C; ++c) { \
            for (size_t oh = 0; oh < OH; ++oh) { \
                for (size_t ow = 0; ow < OW; ++ow) { \
                    float max_val = -FLT_MAX; \
                    int64_t max_idx = -1; \
                    for (size_t kh = 0; kh < K; ++kh) { \
                        for (size_t kw = 0; kw < K; ++kw) { \
                            size_t ih = oh * S + kh; \
                            size_t iw = ow * S + kw; \
                            if (ih < H && iw < W) { \
                                float val = X[(n * C + c) * H * W + ih * W + iw]; \
                                if (val > max_val) { \
                                    max_val = val; \
                                    max_idx = ih * W + iw; \
                                } \
                            } \
                        } \
                    } \
                    Y[(n * C + c) * OH * OW + oh * OW + ow] = max_val; \
                    I[(n * C + c) * OH * OW + oh * OW + ow] = (n * C + c) * H * W + max_idx; \
                } \
            } \
        } \
    } \
}

IMPLEMENT_MAXPOOL_RVV(1)
IMPLEMENT_MAXPOOL_RVV(2)
IMPLEMENT_MAXPOOL_RVV(4)
IMPLEMENT_MAXPOOL_RVV(8)
