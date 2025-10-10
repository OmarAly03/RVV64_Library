#include "../include/defs_maxpool.h"
#include <riscv_vector.h>
#include <algorithm>
#include <climits>  // Changed from cfloat
#include "rvv_defs.hpp"

// --- SCALAR IMPLEMENTATION ---
void maxpool_scalar(const int32_t* X, int32_t* Y, int32_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
    size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
    size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ++ow) {
                    int32_t max_val = INT32_MIN;  // Changed from -FLT_MAX
                    int32_t max_idx = -1;         // Changed from int64_t
                    for (size_t kh = 0; kh < K; ++kh) {
                        for (size_t kw = 0; kw < K; ++kw) {
                            size_t ih = oh * S + kh;
                            size_t iw = ow * S + kw;
                            if (ih < H && iw < W) {
                                int32_t val = X[n*C*H*W + c*H*W + ih*W + iw];  // Changed from float
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

void maxpool_e32m1(const int32_t* X, int32_t* Y, int32_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
    size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
    size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const int32_t* x_channel = X + (n * C + c) * H * W;
            int32_t* y_channel = Y + (n * C + c) * OH * OW;
            int32_t* i_channel = I + (n * C + c) * OH * OW;
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ) {
                    size_t vl = SET_VECTOR_LENGTH<int32_t, M1>(OW - ow);
                    auto max_vec = VECTOR_MOVE<int32_t, M1>(INT32_MIN, vl);  // Changed from vfloat32m1_t and -FLT_MAX
                    auto max_idx_vec32 = VECTOR_MOVE<int32_t, M1>(-1, vl);
                    for (size_t kh = 0; kh < K; ++kh) {
                        for (size_t kw = 0; kw < K; ++kw) {
                            size_t ih = oh * S + kh;
                            if (ih < H) {
                                const int32_t* x_ptr = x_channel + ih * W + ow * S + kw;
                                auto x_vec = VECTOR_STRIDED_LOAD<int32_t, M1>(x_ptr, S * sizeof(int32_t), vl);  // Changed from vfloat32m1_t
                                auto is_greater_mask = VECTOR_GREATER_THAN<int32_t, M1>(x_vec, max_vec, vl);  // Changed from vmfgt
                                max_vec = VECTOR_MAX_MASKED<int32_t, M1>(is_greater_mask, max_vec, x_vec, vl);  // Changed from vfmax
                                int32_t current_idx_base = ih * W + kw;
                                auto offsets = VECTOR_MULTIPLY<uint32_t, M1>(__riscv_vid_v_u32m1(vl), S, vl);
                                auto current_indices = VECTOR_ADD<int32_t, M1>(__riscv_vreinterpret_v_u32m1_i32m1(offsets), ow * S + current_idx_base, vl);
                                max_idx_vec32 = __riscv_vmerge_vvm_i32m1(max_idx_vec32, current_indices, is_greater_mask, vl);
                            }
                        }
                    }
                    VECTOR_STORE<int32_t, M1>(y_channel + oh * OW + ow, max_vec, vl);  // Changed from f32
                    int32_t channel_offset = (n * C + c) * H * W;  // Changed from int64_t
                    auto final_indices = VECTOR_ADD<int32_t, M1>(max_idx_vec32, channel_offset, vl);  // Removed vsext
                    VECTOR_STORE<int32_t, M1>(i_channel + oh * OW + ow, final_indices, vl);  // Changed from i64m2
                    ow += vl;
                }
            }
        }
    }
}

void maxpool_e32m2(const int32_t* X, int32_t* Y, int32_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
	size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
	size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
	for (size_t n = 0; n < N; ++n) {
		for (size_t c = 0; c < C; ++c) {
			const int32_t* x_channel = X + (n * C + c) * H * W;
			int32_t* y_channel = Y + (n * C + c) * OH * OW;
			int32_t* i_channel = I + (n * C + c) * OH * OW;
			for (size_t oh = 0; oh < OH; ++oh) {
				for (size_t ow = 0; ow < OW; ) {
					size_t vl = SET_VECTOR_LENGTH<int32_t, M2>(OW - ow);
					auto max_vec = VECTOR_MOVE<int32_t, M2>(INT32_MIN, vl);
					auto max_idx_vec32 = VECTOR_MOVE<int32_t, M2>(-1, vl);
					for (size_t kh = 0; kh < K; ++kh) {
						for (size_t kw = 0; kw < K; ++kw) {
							size_t ih = oh * S + kh;
							if (ih < H) {
								const int32_t* x_ptr = x_channel + ih * W + ow * S + kw;
								auto x_vec = VECTOR_STRIDED_LOAD<int32_t, M2>(x_ptr, S * sizeof(int32_t), vl);
								auto is_greater_mask = VECTOR_GREATER_THAN<int32_t, M2>(x_vec, max_vec, vl);
								max_vec = VECTOR_MAX_MASKED<int32_t, M2>(is_greater_mask, max_vec, x_vec, vl);
								int32_t current_idx_base = ih * W + kw;
								auto offsets = VECTOR_MULTIPLY<uint32_t, M2>(__riscv_vid_v_u32m2(vl), S, vl);
								auto current_indices = VECTOR_ADD<int32_t, M2>(__riscv_vreinterpret_v_u32m2_i32m2(offsets), ow * S + current_idx_base, vl);
								max_idx_vec32 = __riscv_vmerge_vvm_i32m2(max_idx_vec32, current_indices, is_greater_mask, vl);
							}
						}
					}
					VECTOR_STORE<int32_t, M2>(y_channel + oh * OW + ow, max_vec, vl);
					int32_t channel_offset = (n * C + c) * H * W;
					auto final_indices = VECTOR_ADD<int32_t, M2>(max_idx_vec32, channel_offset, vl);
					VECTOR_STORE<int32_t, M2>(i_channel + oh * OW + ow, final_indices, vl);
					ow += vl;
				}
			}
		}
	}
}

void maxpool_e32m4(const int32_t* X, int32_t* Y, int32_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
	size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
	size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
	for (size_t n = 0; n < N; ++n) {
		for (size_t c = 0; c < C; ++c) {
			const int32_t* x_channel = X + (n * C + c) * H * W;
			int32_t* y_channel = Y + (n * C + c) * OH * OW;
			int32_t* i_channel = I + (n * C + c) * OH * OW;
			for (size_t oh = 0; oh < OH; ++oh) {
				for (size_t ow = 0; ow < OW; ) {
					size_t vl = SET_VECTOR_LENGTH<int32_t, M4>(OW - ow);
					auto max_vec = VECTOR_MOVE<int32_t, M4>(INT32_MIN, vl);
					auto max_idx_vec32 = VECTOR_MOVE<int32_t, M4>(-1, vl);
					for (size_t kh = 0; kh < K; ++kh) {
						for (size_t kw = 0; kw < K; ++kw) {
							size_t ih = oh * S + kh;
							if (ih < H) {
								const int32_t* x_ptr = x_channel + ih * W + ow * S + kw;
								auto x_vec = VECTOR_STRIDED_LOAD<int32_t, M4>(x_ptr, S * sizeof(int32_t), vl);
								auto is_greater_mask = VECTOR_GREATER_THAN<int32_t, M4>(x_vec, max_vec, vl);
								max_vec = VECTOR_MAX_MASKED<int32_t, M4>(is_greater_mask, max_vec, x_vec, vl);
								int32_t current_idx_base = ih * W + kw;
								auto offsets = VECTOR_MULTIPLY<uint32_t, M4>(__riscv_vid_v_u32m4(vl), S, vl);
								auto current_indices = VECTOR_ADD<int32_t, M4>(__riscv_vreinterpret_v_u32m4_i32m4(offsets), ow * S + current_idx_base, vl);
								max_idx_vec32 = __riscv_vmerge_vvm_i32m4(max_idx_vec32, current_indices, is_greater_mask, vl);
							}
						}
					}
					VECTOR_STORE<int32_t, M4>(y_channel + oh * OW + ow, max_vec, vl);
					int32_t channel_offset = (n * C + c) * H * W;
					auto final_indices = VECTOR_ADD<int32_t, M4>(max_idx_vec32, channel_offset, vl);
					VECTOR_STORE<int32_t, M4>(i_channel + oh * OW + ow, final_indices, vl);
					ow += vl;
				}
			}
		}
	}
}

void maxpool_e32m8(const int32_t* X, int32_t* Y, int32_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
	size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
	size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
	for (size_t n = 0; n < N; ++n) {
		for (size_t c = 0; c < C; ++c) {
			const int32_t* x_channel = X + (n * C + c) * H * W;
			int32_t* y_channel = Y + (n * C + c) * OH * OW;
			int32_t* i_channel = I + (n * C + c) * OH * OW;
			for (size_t oh = 0; oh < OH; ++oh) {
				for (size_t ow = 0; ow < OW; ) {
					size_t vl = SET_VECTOR_LENGTH<int32_t, M8>(OW - ow);
					auto max_vec = VECTOR_MOVE<int32_t, M8>(INT32_MIN, vl);
					auto max_idx_vec32 = VECTOR_MOVE<int32_t, M8>(-1, vl);
					for (size_t kh = 0; kh < K; ++kh) {
						for (size_t kw = 0; kw < K; ++kw) {
							size_t ih = oh * S + kh;
							if (ih < H) {
								const int32_t* x_ptr = x_channel + ih * W + ow * S + kw;
								auto x_vec = VECTOR_STRIDED_LOAD<int32_t, M8>(x_ptr, S * sizeof(int32_t), vl);
								auto is_greater_mask = VECTOR_GREATER_THAN<int32_t, M8>(x_vec, max_vec, vl);
								max_vec = VECTOR_MAX_MASKED<int32_t, M8>(is_greater_mask, max_vec, x_vec, vl);
								int32_t current_idx_base = ih * W + kw;
								auto offsets = VECTOR_MULTIPLY<uint32_t, M8>(__riscv_vid_v_u32m8(vl), S, vl);
								auto current_indices = VECTOR_ADD<int32_t, M8>(__riscv_vreinterpret_v_u32m8_i32m8(offsets), ow * S + current_idx_base, vl);
								max_idx_vec32 = __riscv_vmerge_vvm_i32m8(max_idx_vec32, current_indices, is_greater_mask, vl);
							}
						}
					}
					VECTOR_STORE<int32_t, M8>(y_channel + oh * OW + ow, max_vec, vl);
					int32_t channel_offset = (n * C + c) * H * W;
					auto final_indices = VECTOR_ADD<int32_t, M8>(max_idx_vec32, channel_offset, vl);
					VECTOR_STORE<int32_t, M8>(i_channel + oh * OW + ow, final_indices, vl);
					ow += vl;
				}
			}
		}
	}
}
