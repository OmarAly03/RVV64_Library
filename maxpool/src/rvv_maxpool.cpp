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
                            if (ih < H && iw < W) {
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

// --- CORRECTED & EXPLICIT VECTORIZED IMPLEMENTATIONS ---

void maxpool_e32m1(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
    size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
    size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const float* x_channel = X + (n * C + c) * H * W;
            float* y_channel = Y + (n * C + c) * OH * OW;
            int64_t* i_channel = I + (n * C + c) * OH * OW;
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ) {
                    size_t vl = __riscv_vsetvl_e32m1(OW - ow);
                    vfloat32m1_t max_vec = __riscv_vfmv_v_f_f32m1(-FLT_MAX, vl);
                    vint32m1_t max_idx_vec32 = __riscv_vmv_v_x_i32m1(-1, vl);
                    for (size_t kh = 0; kh < K; ++kh) {
                        for (size_t kw = 0; kw < K; ++kw) {
                            size_t ih = oh * S + kh;
                            if (ih < H) {
                                const float* x_ptr = x_channel + ih * W + ow * S + kw;
                                vfloat32m1_t x_vec = __riscv_vlse32_v_f32m1(x_ptr, S * sizeof(float), vl);
                                vbool32_t is_greater_mask = __riscv_vmfgt_vv_f32m1_b32(x_vec, max_vec, vl);
                                max_vec = __riscv_vfmax_vv_f32m1_m(is_greater_mask, max_vec, x_vec, vl);
                                int32_t current_idx_base = ih * W + kw;
                                vuint32m1_t offsets = __riscv_vmul_vx_u32m1(__riscv_vid_v_u32m1(vl), S, vl);
                                vint32m1_t current_indices = __riscv_vadd_vx_i32m1(__riscv_vreinterpret_v_u32m1_i32m1(offsets), ow * S + current_idx_base, vl);
                                max_idx_vec32 = __riscv_vmerge_vvm_i32m1(max_idx_vec32, current_indices, is_greater_mask, vl);
                            }
                        }
                    }
                    __riscv_vse32_v_f32m1(y_channel + oh * OW + ow, max_vec, vl);
                    int64_t channel_offset = (n * C + c) * H * W;
                    vint64m2_t final_indices = __riscv_vadd_vx_i64m2(__riscv_vsext_vf2_i64m2(max_idx_vec32, vl), channel_offset, vl);
                    __riscv_vse64_v_i64m2(i_channel + oh * OW + ow, final_indices, vl);
                    ow += vl;
                }
            }
        }
    }
}

void maxpool_e32m2(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
    size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
    size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const float* x_channel = X + (n * C + c) * H * W;
            float* y_channel = Y + (n * C + c) * OH * OW;
            int64_t* i_channel = I + (n * C + c) * OH * OW;
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ) {
                    size_t vl = __riscv_vsetvl_e32m2(OW - ow);
                    vfloat32m2_t max_vec = __riscv_vfmv_v_f_f32m2(-FLT_MAX, vl);
                    vint32m2_t max_idx_vec32 = __riscv_vmv_v_x_i32m2(-1, vl);
                    for (size_t kh = 0; kh < K; ++kh) {
                        for (size_t kw = 0; kw < K; ++kw) {
                            size_t ih = oh * S + kh;
                            if (ih < H) {
                                const float* x_ptr = x_channel + ih * W + ow * S + kw;
                                vfloat32m2_t x_vec = __riscv_vlse32_v_f32m2(x_ptr, S * sizeof(float), vl);
                                vbool16_t is_greater_mask = __riscv_vmfgt_vv_f32m2_b16(x_vec, max_vec, vl);
                                max_vec = __riscv_vfmax_vv_f32m2_m(is_greater_mask, max_vec, x_vec, vl);
                                int32_t current_idx_base = ih * W + kw;
                                vuint32m2_t offsets = __riscv_vmul_vx_u32m2(__riscv_vid_v_u32m2(vl), S, vl);
                                vint32m2_t current_indices = __riscv_vadd_vx_i32m2(__riscv_vreinterpret_v_u32m2_i32m2(offsets), ow * S + current_idx_base, vl);
                                max_idx_vec32 = __riscv_vmerge_vvm_i32m2(max_idx_vec32, current_indices, is_greater_mask, vl);
                            }
                        }
                    }
                    __riscv_vse32_v_f32m2(y_channel + oh * OW + ow, max_vec, vl);
                    int64_t channel_offset = (n * C + c) * H * W;
                    vint64m4_t final_indices = __riscv_vadd_vx_i64m4(__riscv_vsext_vf2_i64m4(max_idx_vec32, vl), channel_offset, vl);
                    __riscv_vse64_v_i64m4(i_channel + oh * OW + ow, final_indices, vl);
                    ow += vl;
                }
            }
        }
    }
}

void maxpool_e32m4(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
    size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
    size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const float* x_channel = X + (n * C + c) * H * W;
            float* y_channel = Y + (n * C + c) * OH * OW;
            int64_t* i_channel = I + (n * C + c) * OH * OW;
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ) {
                    size_t vl = __riscv_vsetvl_e32m4(OW - ow);
                    vfloat32m4_t max_vec = __riscv_vfmv_v_f_f32m4(-FLT_MAX, vl);
                    vint32m4_t max_idx_vec32 = __riscv_vmv_v_x_i32m4(-1, vl);
                    for (size_t kh = 0; kh < K; ++kh) {
                        for (size_t kw = 0; kw < K; ++kw) {
                            size_t ih = oh * S + kh;
                            if (ih < H) {
                                const float* x_ptr = x_channel + ih * W + ow * S + kw;
                                vfloat32m4_t x_vec = __riscv_vlse32_v_f32m4(x_ptr, S * sizeof(float), vl);
                                vbool8_t is_greater_mask = __riscv_vmfgt_vv_f32m4_b8(x_vec, max_vec, vl);
                                max_vec = __riscv_vfmax_vv_f32m4_m(is_greater_mask, max_vec, x_vec, vl);
                                int32_t current_idx_base = ih * W + kw;
                                vuint32m4_t offsets = __riscv_vmul_vx_u32m4(__riscv_vid_v_u32m4(vl), S, vl);
                                vint32m4_t current_indices = __riscv_vadd_vx_i32m4(__riscv_vreinterpret_v_u32m4_i32m4(offsets), ow * S + current_idx_base, vl);
                                max_idx_vec32 = __riscv_vmerge_vvm_i32m4(max_idx_vec32, current_indices, is_greater_mask, vl);
                            }
                        }
                    }
                    __riscv_vse32_v_f32m4(y_channel + oh * OW + ow, max_vec, vl);
                    int64_t channel_offset = (n * C + c) * H * W;
                    vint64m8_t final_indices = __riscv_vadd_vx_i64m8(__riscv_vsext_vf2_i64m8(max_idx_vec32, vl), channel_offset, vl);
                    __riscv_vse64_v_i64m8(i_channel + oh * OW + ow, final_indices, vl);
                    ow += vl;
                }
            }
        }
    }
}

// Completed `e32m8` implementation.
void maxpool_e32m8(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
    size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
    size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const float* x_channel = X + (n * C + c) * H * W;
            float* y_channel = Y + (n * C + c) * OH * OW;
            int64_t* i_channel = I + (n * C + c) * OH * OW;
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ) {
                    size_t vl = __riscv_vsetvl_e32m8(OW - ow);
                    vfloat32m8_t max_vec = __riscv_vfmv_v_f_f32m8(-FLT_MAX, vl);
                    vint32m8_t max_idx_vec32 = __riscv_vmv_v_x_i32m8(-1, vl);
                    for (size_t kh = 0; kh < K; ++kh) {
                        for (size_t kw = 0; kw < K; ++kw) {
                            size_t ih = oh * S + kh;
                            if (ih < H) {
                                const float* x_ptr = x_channel + ih * W + ow * S + kw;
                                vfloat32m8_t x_vec = __riscv_vlse32_v_f32m8(x_ptr, S * sizeof(float), vl);
                                vbool4_t is_greater_mask = __riscv_vmfgt_vv_f32m8_b4(x_vec, max_vec, vl);
                                max_vec = __riscv_vfmax_vv_f32m8_m(is_greater_mask, max_vec, x_vec, vl);
                                int32_t current_idx_base = ih * W + kw;
                                vuint32m8_t offsets = __riscv_vmul_vx_u32m8(__riscv_vid_v_u32m8(vl), S, vl);
                                vint32m8_t current_indices = __riscv_vadd_vx_i32m8(__riscv_vreinterpret_v_u32m8_i32m8(offsets), ow * S + current_idx_base, vl);
                                max_idx_vec32 = __riscv_vmerge_vvm_i32m8(max_idx_vec32, current_indices, is_greater_mask, vl);
                            }
                        }
                    }
                    __riscv_vse32_v_f32m8(y_channel + oh * OW + ow, max_vec, vl);
                    int64_t channel_offset = (n * C + c) * H * W;

                    // Widen the 32-bit indices to 64-bit in two halves, as LMUL=16 is not possible.
                    size_t half_vl = __riscv_vsetvl_e32m4(vl);
                    vint32m4_t lo_idx32 = __riscv_vget_v_i32m8_i32m4(max_idx_vec32, 0);
                    vint32m4_t hi_idx32 = __riscv_vget_v_i32m8_i32m4(max_idx_vec32, 1);

                    vint64m8_t lo_idx64 = __riscv_vadd_vx_i64m8(__riscv_vsext_vf2_i64m8(lo_idx32, half_vl), channel_offset, half_vl);
                    vint64m8_t hi_idx64 = __riscv_vadd_vx_i64m8(__riscv_vsext_vf2_i64m8(hi_idx32, vl - half_vl), channel_offset, vl - half_vl);
                    
                    __riscv_vse64_v_i64m8(i_channel + oh * OW + ow, lo_idx64, half_vl);
                    __riscv_vse64_v_i64m8(i_channel + oh * OW + ow + half_vl, hi_idx64, vl - half_vl);

                    ow += vl;
                }
            }
        }
    }
}
