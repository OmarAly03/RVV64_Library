#include "../include/defs.h"
#include <riscv_vector.h>
#include <algorithm> // Required for std::min
#include <cfloat>

// --- SCALAR IMPLEMENTATION ---
void maxpool_scalar_tile(
    const float* X, float* Y, int64_t* I,
    size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode,
    size_t OH, size_t OW,
    size_t tile_oh_start, size_t tile_ow_start,
    size_t tile_oh_end, size_t tile_ow_end)
{
    // (Scalar code remains the same)
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t oh = tile_oh_start; oh < tile_oh_end; ++oh) {
                for (size_t ow = tile_ow_start; ow < tile_ow_end; ++ow) {
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
                    I[n*C*OH*OW + c*OH*OW + oh*OW + ow] = (max_idx != -1) ? (n * C * H * W) + (c * H * W) + max_idx : -1;
                }
            }
        }
    }
}

// --- VECTORIZED TILE KERNELS ---

void maxpool_e32m1_tile(
    const float* X, float* Y, int64_t* I,
    size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode,
    size_t OH, size_t OW,
    size_t tile_oh_start, size_t tile_ow_start,
    size_t tile_oh_end, size_t tile_ow_end)
{
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const float* x_channel = X + (n * C + c) * H * W;
            float* y_channel = Y + (n * C + c) * OH * OW;
            int64_t* i_channel = I + (n * C + c) * OH * OW;
            for (size_t oh = tile_oh_start; oh < tile_oh_end; ++oh) {
                for (size_t ow = tile_ow_start; ow < tile_ow_end; ) {
                    size_t current_tile_width = tile_ow_end - ow;
                    size_t vl = __riscv_vsetvl_e32m1(current_tile_width);

                    vfloat32m1_t max_vec = __riscv_vfmv_v_f_f32m1(-FLT_MAX, vl);
                    vint32m1_t max_idx_vec32 = __riscv_vmv_v_x_i32m1(-1, vl);
                    for (size_t kh = 0; kh < K; ++kh) {
                        for (size_t kw = 0; kw < K; ++kw) {
                            size_t ih = oh * S + kh;
                            if (ih < H) {
                                const float* x_ptr = x_channel + ih * W + ow * S + kw;
                                vuint32m1_t element_indices = __riscv_vid_v_u32m1(vl);
                                vuint32m1_t iw_vec = __riscv_vadd_vx_u32m1(__riscv_vmul_vx_u32m1(element_indices, S, vl), ow * S + kw, vl);
                                vbool32_t load_mask = __riscv_vmsltu_vx_u32m1_b32(iw_vec, W, vl);

                                vfloat32m1_t x_vec = __riscv_vlse32_v_f32m1_m(load_mask, x_ptr, S * sizeof(float), vl);
                                vbool32_t is_greater_mask = __riscv_vmfgt_vv_f32m1_b32_m(load_mask, x_vec, max_vec, vl);
                                max_vec = __riscv_vfmax_vv_f32m1_m(is_greater_mask, max_vec, x_vec, vl);
                                int32_t current_idx_base = ih * W + kw;
                                vuint32m1_t offsets = __riscv_vmul_vx_u32m1(element_indices, S, vl);
                                // Corrected typo: vint32m1_t
                                vint32m1_t current_indices = __riscv_vadd_vx_i32m1(__riscv_vreinterpret_v_u32m1_i32m1(offsets), ow * S + current_idx_base, vl);
                                max_idx_vec32 = __riscv_vmerge_vvm_i32m1(max_idx_vec32, current_indices, is_greater_mask, vl);
                            }
                        }
                    }
                    __riscv_vse32_v_f32m1(y_channel + oh * OW + ow, max_vec, vl);

                    int64_t channel_offset = (n * C + c) * H * W;
                    vbool32_t valid_idx_mask = __riscv_vmsne_vx_i32m1_b32(max_idx_vec32, -1, vl);
                    // Corrected offset addition
                    vint64m2_t widened_indices = __riscv_vsext_vf2_i64m2(max_idx_vec32, vl);
                    vint64m2_t final_indices_added = __riscv_vadd_vx_i64m2(widened_indices, (long)channel_offset, vl);
                    vint64m2_t final_indices = __riscv_vmerge_vvm_i64m2(__riscv_vmv_v_x_i64m2(-1, vl), final_indices_added, valid_idx_mask, vl);

                    __riscv_vse64_v_i64m2(i_channel + oh * OW + ow, final_indices, vl);
                    ow += vl;
                }
            }
        }
    }
}


void maxpool_e32m2_tile(
    const float* X, float* Y, int64_t* I,
    size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode,
    size_t OH, size_t OW,
    size_t tile_oh_start, size_t tile_ow_start,
    size_t tile_oh_end, size_t tile_ow_end)
{
     for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const float* x_channel = X + (n * C + c) * H * W;
            float* y_channel = Y + (n * C + c) * OH * OW;
            int64_t* i_channel = I + (n * C + c) * OH * OW;
            for (size_t oh = tile_oh_start; oh < tile_oh_end; ++oh) {
                for (size_t ow = tile_ow_start; ow < tile_ow_end; ) {
                    size_t current_tile_width = tile_ow_end - ow;
                    size_t vl = __riscv_vsetvl_e32m2(current_tile_width);

                    vfloat32m2_t max_vec = __riscv_vfmv_v_f_f32m2(-FLT_MAX, vl);
                    vint32m2_t max_idx_vec32 = __riscv_vmv_v_x_i32m2(-1, vl);
                    for (size_t kh = 0; kh < K; ++kh) {
                        for (size_t kw = 0; kw < K; ++kw) {
                            size_t ih = oh * S + kh;
                            if (ih < H) {
                                const float* x_ptr = x_channel + ih * W + ow * S + kw;
                                vuint32m2_t element_indices = __riscv_vid_v_u32m2(vl);
                                vuint32m2_t iw_vec = __riscv_vadd_vx_u32m2(__riscv_vmul_vx_u32m2(element_indices, S, vl), ow * S + kw, vl);
                                vbool16_t load_mask = __riscv_vmsltu_vx_u32m2_b16(iw_vec, W, vl);

                                vfloat32m2_t x_vec = __riscv_vlse32_v_f32m2_m(load_mask, x_ptr, S * sizeof(float), vl);
                                vbool16_t is_greater_mask = __riscv_vmfgt_vv_f32m2_b16_m(load_mask, x_vec, max_vec, vl);
                                max_vec = __riscv_vfmax_vv_f32m2_m(is_greater_mask, max_vec, x_vec, vl);
                                int32_t current_idx_base = ih * W + kw;
                                vuint32m2_t offsets = __riscv_vmul_vx_u32m2(element_indices, S, vl);
                                vint32m2_t current_indices = __riscv_vadd_vx_i32m2(__riscv_vreinterpret_v_u32m2_i32m2(offsets), ow * S + current_idx_base, vl);
                                max_idx_vec32 = __riscv_vmerge_vvm_i32m2(max_idx_vec32, current_indices, is_greater_mask, vl);
                            }
                        }
                    }
                    __riscv_vse32_v_f32m2(y_channel + oh * OW + ow, max_vec, vl);
                    int64_t channel_offset = (n * C + c) * H * W;
                    vbool16_t valid_idx_mask = __riscv_vmsne_vx_i32m2_b16(max_idx_vec32, -1, vl);
                    // Corrected offset addition
                    vint64m4_t widened_indices = __riscv_vsext_vf2_i64m4(max_idx_vec32, vl);
                    vint64m4_t final_indices_added = __riscv_vadd_vx_i64m4(widened_indices, (long)channel_offset, vl);
                    vint64m4_t final_indices = __riscv_vmerge_vvm_i64m4(__riscv_vmv_v_x_i64m4(-1, vl), final_indices_added, valid_idx_mask, vl);
                    __riscv_vse64_v_i64m4(i_channel + oh * OW + ow, final_indices, vl);
                    ow += vl;
                }
            }
        }
    }
}


void maxpool_e32m4_tile(
    const float* X, float* Y, int64_t* I,
    size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode,
    size_t OH, size_t OW,
    size_t tile_oh_start, size_t tile_ow_start,
    size_t tile_oh_end, size_t tile_ow_end)
{
      for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const float* x_channel = X + (n * C + c) * H * W;
            float* y_channel = Y + (n * C + c) * OH * OW;
            int64_t* i_channel = I + (n * C + c) * OH * OW;
            for (size_t oh = tile_oh_start; oh < tile_oh_end; ++oh) {
                for (size_t ow = tile_ow_start; ow < tile_ow_end; ) {
                    size_t current_tile_width = tile_ow_end - ow;
                    size_t vl = __riscv_vsetvl_e32m4(current_tile_width);

                    vfloat32m4_t max_vec = __riscv_vfmv_v_f_f32m4(-FLT_MAX, vl);
                    vint32m4_t max_idx_vec32 = __riscv_vmv_v_x_i32m4(-1, vl);
                    for (size_t kh = 0; kh < K; ++kh) {
                        for (size_t kw = 0; kw < K; ++kw) {
                            size_t ih = oh * S + kh;
                            if (ih < H) {
                                const float* x_ptr = x_channel + ih * W + ow * S + kw;
                                vuint32m4_t element_indices = __riscv_vid_v_u32m4(vl);
                                vuint32m4_t iw_vec = __riscv_vadd_vx_u32m4(__riscv_vmul_vx_u32m4(element_indices, S, vl), ow * S + kw, vl);
                                vbool8_t load_mask = __riscv_vmsltu_vx_u32m4_b8(iw_vec, W, vl);

                                vfloat32m4_t x_vec = __riscv_vlse32_v_f32m4_m(load_mask, x_ptr, S * sizeof(float), vl);
                                vbool8_t is_greater_mask = __riscv_vmfgt_vv_f32m4_b8_m(load_mask, x_vec, max_vec, vl);
                                max_vec = __riscv_vfmax_vv_f32m4_m(is_greater_mask, max_vec, x_vec, vl);
                                int32_t current_idx_base = ih * W + kw;
                                vuint32m4_t offsets = __riscv_vmul_vx_u32m4(element_indices, S, vl);
                                vint32m4_t current_indices = __riscv_vadd_vx_i32m4(__riscv_vreinterpret_v_u32m4_i32m4(offsets), ow * S + current_idx_base, vl);
                                max_idx_vec32 = __riscv_vmerge_vvm_i32m4(max_idx_vec32, current_indices, is_greater_mask, vl);
                            }
                        }
                    }
                    __riscv_vse32_v_f32m4(y_channel + oh * OW + ow, max_vec, vl);
                    int64_t channel_offset = (n * C + c) * H * W;
                    vbool8_t valid_idx_mask = __riscv_vmsne_vx_i32m4_b8(max_idx_vec32, -1, vl);
                    // Corrected offset addition
                    vint64m8_t widened_indices = __riscv_vsext_vf2_i64m8(max_idx_vec32, vl);
                    vint64m8_t final_indices_added = __riscv_vadd_vx_i64m8(widened_indices, (long)channel_offset, vl);
                    vint64m8_t final_indices = __riscv_vmerge_vvm_i64m8(__riscv_vmv_v_x_i64m8(-1, vl), final_indices_added, valid_idx_mask, vl);
                    __riscv_vse64_v_i64m8(i_channel + oh * OW + ow, final_indices, vl);
                    ow += vl;
                }
            }
        }
    }
}


void maxpool_e32m8_tile(
    const float* X, float* Y, int64_t* I,
    size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode,
    size_t OH, size_t OW,
    size_t tile_oh_start, size_t tile_ow_start,
    size_t tile_oh_end, size_t tile_ow_end)
{
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const float* x_channel = X + (n * C + c) * H * W;
            float* y_channel = Y + (n * C + c) * OH * OW;
            int64_t* i_channel = I + (n * C + c) * OH * OW;
            for (size_t oh = tile_oh_start; oh < tile_oh_end; ++oh) {
                for (size_t ow = tile_ow_start; ow < tile_ow_end; ) {
                    size_t current_tile_width = tile_ow_end - ow;
                    size_t vl = __riscv_vsetvl_e32m8(current_tile_width);

                    vfloat32m8_t max_vec = __riscv_vfmv_v_f_f32m8(-FLT_MAX, vl);
                    vint32m8_t max_idx_vec32 = __riscv_vmv_v_x_i32m8(-1, vl);
                    for (size_t kh = 0; kh < K; ++kh) {
                        for (size_t kw = 0; kw < K; ++kw) {
                            size_t ih = oh * S + kh;
                            if (ih < H) {
                                const float* x_ptr = x_channel + ih * W + ow * S + kw;
                                vuint32m8_t element_indices = __riscv_vid_v_u32m8(vl);
                                vuint32m8_t iw_vec = __riscv_vadd_vx_u32m8(__riscv_vmul_vx_u32m8(element_indices, S, vl), ow * S + kw, vl);
                                vbool4_t load_mask = __riscv_vmsltu_vx_u32m8_b4(iw_vec, W, vl);

                                vfloat32m8_t x_vec = __riscv_vlse32_v_f32m8_m(load_mask, x_ptr, S * sizeof(float), vl);
                                vbool4_t is_greater_mask = __riscv_vmfgt_vv_f32m8_b4_m(load_mask, x_vec, max_vec, vl);
                                max_vec = __riscv_vfmax_vv_f32m8_m(is_greater_mask, max_vec, x_vec, vl);
                                int32_t current_idx_base = ih * W + kw;
                                vuint32m8_t offsets = __riscv_vmul_vx_u32m8(element_indices, S, vl);
                                vint32m8_t current_indices = __riscv_vadd_vx_i32m8(__riscv_vreinterpret_v_u32m8_i32m8(offsets), ow * S + current_idx_base, vl);
                                max_idx_vec32 = __riscv_vmerge_vvm_i32m8(max_idx_vec32, current_indices, is_greater_mask, vl);
                            }
                        }
                    }
                    __riscv_vse32_v_f32m8(y_channel + oh * OW + ow, max_vec, vl);
                    int64_t channel_offset = (n * C + c) * H * W;

                    // Widen the 32-bit indices to 64-bit in two halves
                    size_t current_vl_m8 = vl; // Save original vl
                    size_t half_vl_m4 = __riscv_vsetvl_e32m4(current_vl_m8); // Use m4's vl for splitting m8

                    vint32m4_t lo_idx32 = __riscv_vget_v_i32m8_i32m4(max_idx_vec32, 0);
                    vint32m4_t hi_idx32 = __riscv_vget_v_i32m8_i32m4(max_idx_vec32, 1);

                    vbool8_t lo_valid_mask = __riscv_vmsne_vx_i32m4_b8(lo_idx32, -1, half_vl_m4);
                    vbool8_t hi_valid_mask = __riscv_vmsne_vx_i32m4_b8(hi_idx32, -1, current_vl_m8 - half_vl_m4);

                    // Corrected offset addition for halves
                    vint64m8_t widened_lo = __riscv_vsext_vf2_i64m8(lo_idx32, half_vl_m4);
                    vint64m8_t lo_idx64_added = __riscv_vadd_vx_i64m8(widened_lo, (long)channel_offset, half_vl_m4);
                    vint64m8_t lo_idx64 = __riscv_vmerge_vvm_i64m8(__riscv_vmv_v_x_i64m8(-1, half_vl_m4), lo_idx64_added, lo_valid_mask, half_vl_m4);


                    vint64m8_t widened_hi = __riscv_vsext_vf2_i64m8(hi_idx32, current_vl_m8 - half_vl_m4);
                    vint64m8_t hi_idx64_added = __riscv_vadd_vx_i64m8(widened_hi, (long)channel_offset, current_vl_m8 - half_vl_m4);
                    vint64m8_t hi_idx64 = __riscv_vmerge_vvm_i64m8(__riscv_vmv_v_x_i64m8(-1, current_vl_m8 - half_vl_m4), hi_idx64_added, hi_valid_mask, current_vl_m8 - half_vl_m4);


                    vl = __riscv_vsetvl_e32m8(current_tile_width); // Restore original vl for storage
                    __riscv_vse64_v_i64m8(i_channel + oh * OW + ow, lo_idx64, half_vl_m4);
                    __riscv_vse64_v_i64m8(i_channel + oh * OW + ow + half_vl_m4, hi_idx64, vl - half_vl_m4);

                    ow += vl;
                }
            }
        }
    }
}


// --- HIGH-LEVEL TILING FUNCTIONS (Unchanged) ---
void maxpool_scalar_tiled(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
    size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
    size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
    for (size_t oh_base = 0; oh_base < OH; oh_base += TILE_H) {
        for (size_t ow_base = 0; ow_base < OW; ow_base += TILE_W) {
            size_t oh_end = std::min(oh_base + TILE_H, OH);
            size_t ow_end = std::min(ow_base + TILE_W, OW);
            maxpool_scalar_tile(X, Y, I, N, C, H, W, K, S, ceil_mode, OH, OW, oh_base, ow_base, oh_end, ow_end);
        }
    }
}
void maxpool_e32m1_tiled(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
    size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
    size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
    for (size_t oh_base = 0; oh_base < OH; oh_base += TILE_H) {
        for (size_t ow_base = 0; ow_base < OW; ow_base += TILE_W) {
            size_t oh_end = std::min(oh_base + TILE_H, OH);
            size_t ow_end = std::min(ow_base + TILE_W, OW);
            maxpool_e32m1_tile(X, Y, I, N, C, H, W, K, S, ceil_mode, OH, OW, oh_base, ow_base, oh_end, ow_end);
        }
    }
}
void maxpool_e32m2_tiled(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
    size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
    size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
    for (size_t oh_base = 0; oh_base < OH; oh_base += TILE_H) {
        for (size_t ow_base = 0; ow_base < OW; ow_base += TILE_W) {
            size_t oh_end = std::min(oh_base + TILE_H, OH);
            size_t ow_end = std::min(ow_base + TILE_W, OW);
            maxpool_e32m2_tile(X, Y, I, N, C, H, W, K, S, ceil_mode, OH, OW, oh_base, ow_base, oh_end, ow_end);
        }
    }
}
void maxpool_e32m4_tiled(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
    size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
    size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
    for (size_t oh_base = 0; oh_base < OH; oh_base += TILE_H) {
        for (size_t ow_base = 0; ow_base < OW; ow_base += TILE_W) {
            size_t oh_end = std::min(oh_base + TILE_H, OH);
            size_t ow_end = std::min(ow_base + TILE_W, OW);
            maxpool_e32m4_tile(X, Y, I, N, C, H, W, K, S, ceil_mode, OH, OW, oh_base, ow_base, oh_end, ow_end);
        }
    }
}
void maxpool_e32m8_tiled(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
    size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
    size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
    for (size_t oh_base = 0; oh_base < OH; oh_base += TILE_H) {
        for (size_t ow_base = 0; ow_base < OW; ow_base += TILE_W) {
            size_t oh_end = std::min(oh_base + TILE_H, OH);
            size_t ow_end = std::min(ow_base + TILE_W, OW);
            maxpool_e32m8_tile(X, Y, I, N, C, H, W, K, S, ceil_mode, OH, OW, oh_base, ow_base, oh_end, ow_end);
        }
    }
}
