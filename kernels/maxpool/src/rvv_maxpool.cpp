#include "../include/defs_maxpool.h"
#include <riscv_vector.h>
#include <algorithm>
#include <climits>  // Changed from cfloat
#include "rvv_defs.hpp"
extern "C" {
    #include <uart.h>
}

// --- SCALAR IMPLEMENTATION ---
void maxpool_scalar(const int32_t* X, int32_t* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
    size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
    size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
    
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ++ow) {
                    int32_t max_val = INT32_MIN;
                    int64_t max_idx = -1;  // Changed to int64_t
                    
                    for (size_t kh = 0; kh < K; ++kh) {
                        for (size_t kw = 0; kw < K; ++kw) {
                            size_t ih = oh * S + kh;
                            size_t iw = ow * S + kw;
                            
                            if (ih < H && iw < W) {
                                size_t input_idx = n*C*H*W + c*H*W + ih*W + iw;
                                int32_t val = X[input_idx];
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = input_idx;  // Store the absolute index
                                }
                            }
                        }
                    }
                    
                    size_t output_idx = n*C*OH*OW + c*OH*OW + oh*OW + ow;
                    Y[output_idx] = max_val;
                    I[output_idx] = max_idx;
                }
            }
        }
    }
}

void maxpool_e32m1(const int32_t* X, int32_t* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
    size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
    size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
    
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const int32_t* x_channel = X + (n * C + c) * H * W;
            int32_t* y_channel = Y + (n * C + c) * OH * OW;
            int64_t* i_channel = I + (n * C + c) * OH * OW;
            
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ) {
                    size_t vl = SET_VECTOR_LENGTH<int32_t, M1>(OW - ow);
                    auto max_vec = VECTOR_MOVE<int32_t, M1>(INT32_MIN, vl);
                    auto max_idx_vec32 = VECTOR_MOVE<int32_t, M1>(-1, vl);
                    
                    for (size_t kh = 0; kh < K; ++kh) {
                        for (size_t kw = 0; kw < K; ++kw) {
                            size_t ih = oh * S + kh;
                            if (ih < H) {
                                // Check if ALL elements in the vector will be valid
                                bool all_valid = true;
                                for (size_t v = 0; v < vl; ++v) {
                                    size_t iw = (ow + v) * S + kw;
                                    if (iw >= W) {
                                        all_valid = false;
                                        break;
                                    }
                                }
                                
                                if (all_valid) {
                                    // Safe to use vectorized approach
                                    const int32_t* x_ptr = x_channel + ih * W + ow * S + kw;
                                    auto x_vec = VECTOR_STRIDED_LOAD<int32_t, M1>(x_ptr, S * sizeof(int32_t), vl);
                                    auto is_greater_mask = VECTOR_GREATER_THAN<int32_t, M1>(x_vec, max_vec, vl);
                                    max_vec = VECTOR_MAX_MASKED<int32_t, M1>(is_greater_mask, max_vec, x_vec, vl);
                                    
                                    // Calculate indices
                                    int32_t current_idx_base = ih * W + kw;
                                    auto offsets = VECTOR_MULTIPLY<uint32_t, M1>(__riscv_vid_v_u32m1(vl), S, vl);
                                    auto current_indices = VECTOR_ADD<int32_t, M1>(__riscv_vreinterpret_v_u32m1_i32m1(offsets), ow * S + current_idx_base, vl);
                                    max_idx_vec32 = __riscv_vmerge_vvm_i32m1(max_idx_vec32, current_indices, is_greater_mask, vl);
                                } else {
                                    // Handle boundary elements one by one using scalar operations
                                    // Extract current max values and indices
                                    int32_t temp_max[32];
                                    int32_t temp_idx[32];
                                    VECTOR_STORE<int32_t, M1>(temp_max, max_vec, vl);
                                    VECTOR_STORE<int32_t, M1>(temp_idx, max_idx_vec32, vl);
                                    
                                    for (size_t v = 0; v < vl; ++v) {
                                        size_t iw = (ow + v) * S + kw;
                                        if (iw < W) {
                                            int32_t val = x_channel[ih * W + iw];
                                            if (val > temp_max[v]) {
                                                temp_max[v] = val;
                                                temp_idx[v] = ih * W + iw;
                                            }
                                        }
                                    }
                                    
                                    // Load back into vectors
                                    max_vec = VECTOR_LOAD<int32_t, M1>(temp_max, vl);
                                    max_idx_vec32 = VECTOR_LOAD<int32_t, M1>(temp_idx, vl);
                                }
                            }
                        }
                    }
                    
                    VECTOR_STORE<int32_t, M1>(y_channel + oh * OW + ow, max_vec, vl);
                    
                    // Store 32-bit indices and convert to 64-bit in scalar code (Vicuna compatible)
                    int32_t temp_indices[32];
                    VECTOR_STORE<int32_t, M1>(temp_indices, max_idx_vec32, vl);
                    
                    // Add channel offset and convert to 64-bit using scalar operations
                    int64_t channel_offset = (n * C + c) * H * W;
                    for (size_t i = 0; i < vl; ++i) {
                        i_channel[oh * OW + ow + i] = channel_offset + temp_indices[i];
                    }
                    
                    ow += vl;
                }
            }
        }
    }
}

void maxpool_e32m2(const int32_t* X, int32_t* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
    size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
    size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
    
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const int32_t* x_channel = X + (n * C + c) * H * W;
            int32_t* y_channel = Y + (n * C + c) * OH * OW;
            int64_t* i_channel = I + (n * C + c) * OH * OW;
            
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ) {
                    size_t vl = SET_VECTOR_LENGTH<int32_t, M2>(OW - ow);
                    auto max_vec = VECTOR_MOVE<int32_t, M2>(INT32_MIN, vl);
                    auto max_idx_vec32 = VECTOR_MOVE<int32_t, M2>(-1, vl);
                    
                    for (size_t kh = 0; kh < K; ++kh) {
                        for (size_t kw = 0; kw < K; ++kw) {
                            size_t ih = oh * S + kh;
                            if (ih < H) {
                                // Always use the safe scalar approach for index calculations
                                // Extract current max values and indices
                                int32_t temp_max[32];
                                int32_t temp_idx[32];
                                VECTOR_STORE<int32_t, M2>(temp_max, max_vec, vl);
                                VECTOR_STORE<int32_t, M2>(temp_idx, max_idx_vec32, vl);
                                
                                for (size_t v = 0; v < vl; ++v) {
                                    size_t iw = (ow + v) * S + kw;
                                    if (iw < W) {
                                        int32_t val = x_channel[ih * W + iw];
                                        if (val > temp_max[v]) {
                                            temp_max[v] = val;
                                            temp_idx[v] = ih * W + iw;
                                        }
                                    }
                                }
                                
                                // Load back into vectors
                                max_vec = VECTOR_LOAD<int32_t, M2>(temp_max, vl);
                                max_idx_vec32 = VECTOR_LOAD<int32_t, M2>(temp_idx, vl);
                            }
                        }
                    }
                    
                    VECTOR_STORE<int32_t, M2>(y_channel + oh * OW + ow, max_vec, vl);
                    
                    // Store 32-bit indices and convert to 64-bit in scalar code (Vicuna compatible)
                    int32_t temp_indices[32];
                    VECTOR_STORE<int32_t, M2>(temp_indices, max_idx_vec32, vl);
                    
                    // Add channel offset and convert to 64-bit using scalar operations
                    int64_t channel_offset = (n * C + c) * H * W;
                    for (size_t i = 0; i < vl; ++i) {
                        i_channel[oh * OW + ow + i] = channel_offset + temp_indices[i];
                    }
                    
                    ow += vl;
                }
            }
        }
    }
}

void maxpool_e32m4(const int32_t* X, int32_t* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
    size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
    size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
    
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const int32_t* x_channel = X + (n * C + c) * H * W;
            int32_t* y_channel = Y + (n * C + c) * OH * OW;
            int64_t* i_channel = I + (n * C + c) * OH * OW;
            
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ) {
                    size_t vl = SET_VECTOR_LENGTH<int32_t, M4>(OW - ow);
                    auto max_vec = VECTOR_MOVE<int32_t, M4>(INT32_MIN, vl);
                    auto max_idx_vec32 = VECTOR_MOVE<int32_t, M4>(-1, vl);
                    
                    for (size_t kh = 0; kh < K; ++kh) {
                        for (size_t kw = 0; kw < K; ++kw) {
                            size_t ih = oh * S + kh;
                            if (ih < H) {
                                // Always use the safe scalar approach for index calculations
                                // Extract current max values and indices
                                int32_t temp_max[32];
                                int32_t temp_idx[32];
                                VECTOR_STORE<int32_t, M4>(temp_max, max_vec, vl);
                                VECTOR_STORE<int32_t, M4>(temp_idx, max_idx_vec32, vl);
                                
                                for (size_t v = 0; v < vl; ++v) {
                                    size_t iw = (ow + v) * S + kw;
                                    if (iw < W) {
                                        int32_t val = x_channel[ih * W + iw];
                                        if (val > temp_max[v]) {
                                            temp_max[v] = val;
                                            temp_idx[v] = ih * W + iw;
                                        }
                                    }
                                }
                                
                                // Load back into vectors
                                max_vec = VECTOR_LOAD<int32_t, M4>(temp_max, vl);
                                max_idx_vec32 = VECTOR_LOAD<int32_t, M4>(temp_idx, vl);
                            }
                        }
                    }
                    
                    VECTOR_STORE<int32_t, M4>(y_channel + oh * OW + ow, max_vec, vl);
                    
                    // Store 32-bit indices and convert to 64-bit in scalar code (Vicuna compatible)
                    int32_t temp_indices[32];
                    VECTOR_STORE<int32_t, M4>(temp_indices, max_idx_vec32, vl);
                    
                    // Add channel offset and convert to 64-bit using scalar operations
                    int64_t channel_offset = (n * C + c) * H * W;
                    for (size_t i = 0; i < vl; ++i) {
                        i_channel[oh * OW + ow + i] = channel_offset + temp_indices[i];
                    }
                    
                    ow += vl;
                }
            }
        }
    }
}

void maxpool_e32m8(const int32_t* X, int32_t* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode) {
    size_t OH = CALC_OUT_DIM(H, K, S, ceil_mode);
    size_t OW = CALC_OUT_DIM(W, K, S, ceil_mode);
    
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const int32_t* x_channel = X + (n * C + c) * H * W;
            int32_t* y_channel = Y + (n * C + c) * OH * OW;
            int64_t* i_channel = I + (n * C + c) * OH * OW;
            
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ) {
                    size_t vl = SET_VECTOR_LENGTH<int32_t, M8>(OW - ow);
                    auto max_vec = VECTOR_MOVE<int32_t, M8>(INT32_MIN, vl);
                    auto max_idx_vec32 = VECTOR_MOVE<int32_t, M8>(-1, vl);
                    
                    for (size_t kh = 0; kh < K; ++kh) {
                        for (size_t kw = 0; kw < K; ++kw) {
                            size_t ih = oh * S + kh;
                            if (ih < H) {
                                // Always use the safe scalar approach for index calculations
                                // Extract current max values and indices
                                int32_t temp_max[64]; // Larger buffer for M8
                                int32_t temp_idx[64];
                                VECTOR_STORE<int32_t, M8>(temp_max, max_vec, vl);
                                VECTOR_STORE<int32_t, M8>(temp_idx, max_idx_vec32, vl);
                                
                                for (size_t v = 0; v < vl; ++v) {
                                    size_t iw = (ow + v) * S + kw;
                                    if (iw < W) {
                                        int32_t val = x_channel[ih * W + iw];
                                        if (val > temp_max[v]) {
                                            temp_max[v] = val;
                                            temp_idx[v] = ih * W + iw;
                                        }
                                    }
                                }
                                
                                // Load back into vectors
                                max_vec = VECTOR_LOAD<int32_t, M8>(temp_max, vl);
                                max_idx_vec32 = VECTOR_LOAD<int32_t, M8>(temp_idx, vl);
                            }
                        }
                    }
                    
                    VECTOR_STORE<int32_t, M8>(y_channel + oh * OW + ow, max_vec, vl);
                    
                    // Store 32-bit indices and convert to 64-bit in scalar code (Vicuna compatible)
                    int32_t temp_indices[64]; // Larger buffer for M8
                    VECTOR_STORE<int32_t, M8>(temp_indices, max_idx_vec32, vl);
                    
                    // Add channel offset and convert to 64-bit using scalar operations
                    int64_t channel_offset = (n * C + c) * H * W;
                    for (size_t i = 0; i < vl; ++i) {
                        i_channel[oh * OW + ow + i] = channel_offset + temp_indices[i];
                    }
                    
                    ow += vl;
                }
            }
        }
    }
}
