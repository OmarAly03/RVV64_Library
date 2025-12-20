#include "../include/defs.h"
#include <cstring>
#include <riscv_vector.h>

// Scalar implementation
void scatter_elements_scalar(
    const float* data,
    const int64_t* indices,
    const float* updates,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis
) {
    // Copy data to output first
    memcpy(output, data, data_rows * data_cols * sizeof(float));
    
    if (axis == 0) {
        // Scatter along rows
        for (size_t i = 0; i < indices_rows; i++) {
            for (size_t j = 0; j < indices_cols; j++) {
                int64_t target_row = indices[i * indices_cols + j];
                if (target_row >= 0 && target_row < (int64_t)data_rows) {
                    output[target_row * data_cols + j] = updates[i * indices_cols + j];
                }
            }
        }
    } else if (axis == 1) {
        // Scatter along columns
        for (size_t i = 0; i < indices_rows; i++) {
            for (size_t j = 0; j < indices_cols; j++) {
                int64_t target_col = indices[i * indices_cols + j];
                if (target_col >= 0 && target_col < (int64_t)data_cols) {
                    output[i * data_cols + target_col] = updates[i * indices_cols + j];
                }
            }
        }
    }
}

// Vectorized implementation using indexed stores (LMUL=1)
void scatter_elements_e32m1(
    const float* data,
    const int64_t* indices,
    const float* updates,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis
) {
    // Copy data to output first
    memcpy(output, data, data_rows * data_cols * sizeof(float));
    
    if (axis == 0) {
        // Scatter along rows - vectorize across columns
        for (size_t i = 0; i < indices_rows; i++) {
            size_t j = 0;
            while (j < indices_cols) {
                size_t vl = __riscv_vsetvl_e32m1(indices_cols - j);
                
                // Load indices and updates
                vint64m2_t v_indices_64 = __riscv_vle64_v_i64m2(&indices[i * indices_cols + j], vl);
                vfloat32m1_t v_updates = __riscv_vle32_v_f32m1(&updates[i * indices_cols + j], vl);
                
                // Convert 64-bit indices to 32-bit and scale by stride
                vint32m1_t v_indices_32 = __riscv_vnsra_wx_i32m1(v_indices_64, 0, vl);
                vint32m1_t v_byte_offsets = __riscv_vmul_vx_i32m1(v_indices_32, data_cols * sizeof(float), vl);
                
                // Add column offset (j + vid to get absolute column position)
                vuint32m1_t v_vid = __riscv_vid_v_u32m1(vl);
                vuint32m1_t v_col_idx = __riscv_vadd_vx_u32m1(v_vid, j, vl);
                vuint32m1_t v_col_offsets_u = __riscv_vmul_vx_u32m1(v_col_idx, sizeof(float), vl);
                vint32m1_t v_col_offsets = __riscv_vreinterpret_v_u32m1_i32m1(v_col_offsets_u);
                vint32m1_t v_final_offsets = __riscv_vadd_vv_i32m1(v_byte_offsets, v_col_offsets, vl);
                
                // Indexed store
                vuint32m1_t v_offsets_u = __riscv_vreinterpret_v_i32m1_u32m1(v_final_offsets);
                __riscv_vsuxei32_v_f32m1(output, v_offsets_u, v_updates, vl);
                
                j += vl;
            }
        }
    } else if (axis == 1) {
        // Scatter along columns
        for (size_t i = 0; i < indices_rows; i++) {
            float* row_output = &output[i * data_cols];
            size_t j = 0;
            while (j < indices_cols) {
                size_t vl = __riscv_vsetvl_e32m1(indices_cols - j);
                
                // Load indices and updates
                vint64m2_t v_indices_64 = __riscv_vle64_v_i64m2(&indices[i * indices_cols + j], vl);
                vfloat32m1_t v_updates = __riscv_vle32_v_f32m1(&updates[i * indices_cols + j], vl);
                
                // Convert to byte offsets
                vint32m1_t v_indices_32 = __riscv_vnsra_wx_i32m1(v_indices_64, 0, vl);
                vint32m1_t v_byte_offsets = __riscv_vsll_vx_i32m1(v_indices_32, 2, vl); // multiply by 4
                
                // Indexed store
                vuint32m1_t v_offsets_u = __riscv_vreinterpret_v_i32m1_u32m1(v_byte_offsets);
                __riscv_vsuxei32_v_f32m1(row_output, v_offsets_u, v_updates, vl);
                
                j += vl;
            }
        }
    }
}

// LMUL=2 version
void scatter_elements_e32m2(
    const float* data,
    const int64_t* indices,
    const float* updates,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis
) {
    memcpy(output, data, data_rows * data_cols * sizeof(float));
    
    if (axis == 0) {
        for (size_t i = 0; i < indices_rows; i++) {
            size_t j = 0;
            while (j < indices_cols) {
                size_t vl = __riscv_vsetvl_e32m2(indices_cols - j);
                
                vint64m4_t v_indices_64 = __riscv_vle64_v_i64m4(&indices[i * indices_cols + j], vl);
                vfloat32m2_t v_updates = __riscv_vle32_v_f32m2(&updates[i * indices_cols + j], vl);
                
                vint32m2_t v_indices_32 = __riscv_vnsra_wx_i32m2(v_indices_64, 0, vl);
                vint32m2_t v_byte_offsets = __riscv_vmul_vx_i32m2(v_indices_32, data_cols * sizeof(float), vl);
                vuint32m2_t v_vid = __riscv_vid_v_u32m2(vl);
                vuint32m2_t v_col_idx = __riscv_vadd_vx_u32m2(v_vid, j, vl);
                vuint32m2_t v_col_offsets_u = __riscv_vmul_vx_u32m2(v_col_idx, sizeof(float), vl);
                vint32m2_t v_col_offsets = __riscv_vreinterpret_v_u32m2_i32m2(v_col_offsets_u);
                vint32m2_t v_final_offsets = __riscv_vadd_vv_i32m2(v_byte_offsets, v_col_offsets, vl);
                
                vuint32m2_t v_offsets_u = __riscv_vreinterpret_v_i32m2_u32m2(v_final_offsets);
                __riscv_vsuxei32_v_f32m2(output, v_offsets_u, v_updates, vl);
                
                j += vl;
            }
        }
    } else if (axis == 1) {
        for (size_t i = 0; i < indices_rows; i++) {
            float* row_output = &output[i * data_cols];
            size_t j = 0;
            while (j < indices_cols) {
                size_t vl = __riscv_vsetvl_e32m2(indices_cols - j);
                
                vint64m4_t v_indices_64 = __riscv_vle64_v_i64m4(&indices[i * indices_cols + j], vl);
                vfloat32m2_t v_updates = __riscv_vle32_v_f32m2(&updates[i * indices_cols + j], vl);
                
                vint32m2_t v_indices_32 = __riscv_vnsra_wx_i32m2(v_indices_64, 0, vl);
                vint32m2_t v_byte_offsets = __riscv_vsll_vx_i32m2(v_indices_32, 2, vl);
                
                vuint32m2_t v_offsets_u = __riscv_vreinterpret_v_i32m2_u32m2(v_byte_offsets);
                __riscv_vsuxei32_v_f32m2(row_output, v_offsets_u, v_updates, vl);
                
                j += vl;
            }
        }
    }
}

// LMUL=4 version
void scatter_elements_e32m4(
    const float* data,
    const int64_t* indices,
    const float* updates,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis
) {
    memcpy(output, data, data_rows * data_cols * sizeof(float));
    
    if (axis == 0) {
        for (size_t i = 0; i < indices_rows; i++) {
            size_t j = 0;
            while (j < indices_cols) {
                size_t vl = __riscv_vsetvl_e32m4(indices_cols - j);
                
                vint64m8_t v_indices_64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl);
                vfloat32m4_t v_updates = __riscv_vle32_v_f32m4(&updates[i * indices_cols + j], vl);
                
                vint32m4_t v_indices_32 = __riscv_vnsra_wx_i32m4(v_indices_64, 0, vl);
                vint32m4_t v_byte_offsets = __riscv_vmul_vx_i32m4(v_indices_32, data_cols * sizeof(float), vl);
                vuint32m4_t v_vid = __riscv_vid_v_u32m4(vl);
                vuint32m4_t v_col_idx = __riscv_vadd_vx_u32m4(v_vid, j, vl);
                vuint32m4_t v_col_offsets_u = __riscv_vmul_vx_u32m4(v_col_idx, sizeof(float), vl);
                vint32m4_t v_col_offsets = __riscv_vreinterpret_v_u32m4_i32m4(v_col_offsets_u);
                vint32m4_t v_final_offsets = __riscv_vadd_vv_i32m4(v_byte_offsets, v_col_offsets, vl);
                
                vuint32m4_t v_offsets_u = __riscv_vreinterpret_v_i32m4_u32m4(v_final_offsets);
                __riscv_vsuxei32_v_f32m4(output, v_offsets_u, v_updates, vl);
                
                j += vl;
            }
        }
    } else if (axis == 1) {
        for (size_t i = 0; i < indices_rows; i++) {
            float* row_output = &output[i * data_cols];
            size_t j = 0;
            while (j < indices_cols) {
                size_t vl = __riscv_vsetvl_e32m4(indices_cols - j);
                
                vint64m8_t v_indices_64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl);
                vfloat32m4_t v_updates = __riscv_vle32_v_f32m4(&updates[i * indices_cols + j], vl);
                
                vint32m4_t v_indices_32 = __riscv_vnsra_wx_i32m4(v_indices_64, 0, vl);
                vint32m4_t v_byte_offsets = __riscv_vsll_vx_i32m4(v_indices_32, 2, vl);
                
                vuint32m4_t v_offsets_u = __riscv_vreinterpret_v_i32m4_u32m4(v_byte_offsets);
                __riscv_vsuxei32_v_f32m4(row_output, v_offsets_u, v_updates, vl);
                
                j += vl;
            }
        }
    }
}

// LMUL=8 version
void scatter_elements_e32m8(
    const float* data,
    const int64_t* indices,
    const float* updates,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis
) {
    memcpy(output, data, data_rows * data_cols * sizeof(float));
    
    if (axis == 0) {
        for (size_t i = 0; i < indices_rows; i++) {
            size_t j = 0;
            while (j < indices_cols) {
                size_t vl = __riscv_vsetvl_e32m8(indices_cols - j);
                
                // For m8, we need to process in smaller chunks for indices
                // since we can't have m16 for int64
                size_t vl_half = vl / 2;
                if (vl_half == 0) vl_half = 1;
                
                // Process first half
                vint64m8_t v_indices_64_1 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl_half);
                vfloat32m4_t v_updates_1 = __riscv_vle32_v_f32m4(&updates[i * indices_cols + j], vl_half);
                
                vint32m4_t v_indices_32_1 = __riscv_vnsra_wx_i32m4(v_indices_64_1, 0, vl_half);
                vint32m4_t v_byte_offsets_1 = __riscv_vmul_vx_i32m4(v_indices_32_1, data_cols * sizeof(float), vl_half);
                vuint32m4_t v_vid = __riscv_vid_v_u32m4(vl_half);
                vuint32m4_t v_col_idx = __riscv_vadd_vx_u32m4(v_vid, j, vl_half);
                vuint32m4_t v_col_offsets_u_1 = __riscv_vmul_vx_u32m4(v_col_idx, sizeof(float), vl_half);
                vint32m4_t v_col_offsets_1 = __riscv_vreinterpret_v_u32m4_i32m4(v_col_offsets_u_1);
                vint32m4_t v_final_offsets_1 = __riscv_vadd_vv_i32m4(v_byte_offsets_1, v_col_offsets_1, vl_half);
                
                vuint32m4_t v_offsets_u_1 = __riscv_vreinterpret_v_i32m4_u32m4(v_final_offsets_1);
                __riscv_vsuxei32_v_f32m4(output, v_offsets_u_1, v_updates_1, vl_half);
                
                j += vl_half;
            }
        }
    } else if (axis == 1) {
        for (size_t i = 0; i < indices_rows; i++) {
            float* row_output = &output[i * data_cols];
            size_t j = 0;
            while (j < indices_cols) {
                size_t vl = __riscv_vsetvl_e32m8(indices_cols - j);
                size_t vl_half = vl / 2;
                if (vl_half == 0) vl_half = 1;
                
                vint64m8_t v_indices_64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl_half);
                vfloat32m4_t v_updates = __riscv_vle32_v_f32m4(&updates[i * indices_cols + j], vl_half);
                
                vint32m4_t v_indices_32 = __riscv_vnsra_wx_i32m4(v_indices_64, 0, vl_half);
                vint32m4_t v_byte_offsets = __riscv_vsll_vx_i32m4(v_indices_32, 2, vl_half);
                
                vuint32m4_t v_offsets_u = __riscv_vreinterpret_v_i32m4_u32m4(v_byte_offsets);
                __riscv_vsuxei32_v_f32m4(row_output, v_offsets_u, v_updates, vl_half);
                
                j += vl_half;
            }
        }
    }
}


// ============================================================================
// TILED IMPLEMENTATIONS
// ============================================================================

// Tiled scalar implementation
void scatter_elements_tiled_scalar(
    const float* data,
    const int64_t* indices,
    const float* updates,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis,
    size_t tile_size
) {
    // Copy data to output first
    memcpy(output, data, data_rows * data_cols * sizeof(float));
    
    if (axis == 0) {
        // Process indices in tiles for better cache locality
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            
            for (size_t j_tile = 0; j_tile < indices_cols; j_tile += tile_size) {
                size_t j_end = (j_tile + tile_size < indices_cols) ? j_tile + tile_size : indices_cols;
                
                // Process tile
                for (size_t i = i_tile; i < i_end; i++) {
                    for (size_t j = j_tile; j < j_end; j++) {
                        int64_t target_row = indices[i * indices_cols + j];
                        if (target_row >= 0 && target_row < (int64_t)data_rows) {
                            output[target_row * data_cols + j] = updates[i * indices_cols + j];
                        }
                    }
                }
            }
        }
    } else if (axis == 1) {
        // Process rows in tiles
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            
            for (size_t j_tile = 0; j_tile < indices_cols; j_tile += tile_size) {
                size_t j_end = (j_tile + tile_size < indices_cols) ? j_tile + tile_size : indices_cols;
                
                // Process tile
                for (size_t i = i_tile; i < i_end; i++) {
                    for (size_t j = j_tile; j < j_end; j++) {
                        int64_t target_col = indices[i * indices_cols + j];
                        if (target_col >= 0 && target_col < (int64_t)data_cols) {
                            output[i * data_cols + target_col] = updates[i * indices_cols + j];
                        }
                    }
                }
            }
        }
    }
}

// Tiled vectorized implementation (LMUL=1)
void scatter_elements_tiled_e32m1(
    const float* data,
    const int64_t* indices,
    const float* updates,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis,
    size_t tile_size
) {
    memcpy(output, data, data_rows * data_cols * sizeof(float));
    
    if (axis == 0) {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            
            for (size_t i = i_tile; i < i_end; i++) {
                size_t j = 0;
                while (j < indices_cols) {
                    size_t vl = __riscv_vsetvl_e32m1(indices_cols - j);
                    
                    vint64m2_t v_indices_64 = __riscv_vle64_v_i64m2(&indices[i * indices_cols + j], vl);
                    vfloat32m1_t v_updates = __riscv_vle32_v_f32m1(&updates[i * indices_cols + j], vl);
                    
                    vint32m1_t v_indices_32 = __riscv_vnsra_wx_i32m1(v_indices_64, 0, vl);
                    vint32m1_t v_byte_offsets = __riscv_vmul_vx_i32m1(v_indices_32, data_cols * sizeof(float), vl);
                    vuint32m1_t v_vid = __riscv_vid_v_u32m1(vl);
                    vuint32m1_t v_col_idx = __riscv_vadd_vx_u32m1(v_vid, j, vl);
                    vuint32m1_t v_col_offsets_u = __riscv_vmul_vx_u32m1(v_col_idx, sizeof(float), vl);
                    vint32m1_t v_col_offsets = __riscv_vreinterpret_v_u32m1_i32m1(v_col_offsets_u);
                    vint32m1_t v_final_offsets = __riscv_vadd_vv_i32m1(v_byte_offsets, v_col_offsets, vl);
                    
                    vuint32m1_t v_offsets_u = __riscv_vreinterpret_v_i32m1_u32m1(v_final_offsets);
                    __riscv_vsuxei32_v_f32m1(output, v_offsets_u, v_updates, vl);
                    
                    j += vl;
                }
            }
        }
    } else if (axis == 1) {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            
            for (size_t i = i_tile; i < i_end; i++) {
                float* row_output = &output[i * data_cols];
                size_t j = 0;
                while (j < indices_cols) {
                    size_t vl = __riscv_vsetvl_e32m1(indices_cols - j);
                    
                    vint64m2_t v_indices_64 = __riscv_vle64_v_i64m2(&indices[i * indices_cols + j], vl);
                    vfloat32m1_t v_updates = __riscv_vle32_v_f32m1(&updates[i * indices_cols + j], vl);
                    
                    vint32m1_t v_indices_32 = __riscv_vnsra_wx_i32m1(v_indices_64, 0, vl);
                    vint32m1_t v_byte_offsets = __riscv_vsll_vx_i32m1(v_indices_32, 2, vl);
                    
                    vuint32m1_t v_offsets_u = __riscv_vreinterpret_v_i32m1_u32m1(v_byte_offsets);
                    __riscv_vsuxei32_v_f32m1(row_output, v_offsets_u, v_updates, vl);
                    
                    j += vl;
                }
            }
        }
    }
}

// Tiled vectorized implementation (LMUL=2)
void scatter_elements_tiled_e32m2(
    const float* data,
    const int64_t* indices,
    const float* updates,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis,
    size_t tile_size
) {
    memcpy(output, data, data_rows * data_cols * sizeof(float));
    
    if (axis == 0) {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            
            for (size_t i = i_tile; i < i_end; i++) {
                size_t j = 0;
                while (j < indices_cols) {
                    size_t vl = __riscv_vsetvl_e32m2(indices_cols - j);
                    
                    vint64m4_t v_indices_64 = __riscv_vle64_v_i64m4(&indices[i * indices_cols + j], vl);
                    vfloat32m2_t v_updates = __riscv_vle32_v_f32m2(&updates[i * indices_cols + j], vl);
                    
                    vint32m2_t v_indices_32 = __riscv_vnsra_wx_i32m2(v_indices_64, 0, vl);
                    vint32m2_t v_byte_offsets = __riscv_vmul_vx_i32m2(v_indices_32, data_cols * sizeof(float), vl);
                    vuint32m2_t v_vid = __riscv_vid_v_u32m2(vl);
                    vuint32m2_t v_col_idx = __riscv_vadd_vx_u32m2(v_vid, j, vl);
                    vuint32m2_t v_col_offsets_u = __riscv_vmul_vx_u32m2(v_col_idx, sizeof(float), vl);
                    vint32m2_t v_col_offsets = __riscv_vreinterpret_v_u32m2_i32m2(v_col_offsets_u);
                    vint32m2_t v_final_offsets = __riscv_vadd_vv_i32m2(v_byte_offsets, v_col_offsets, vl);
                    
                    vuint32m2_t v_offsets_u = __riscv_vreinterpret_v_i32m2_u32m2(v_final_offsets);
                    __riscv_vsuxei32_v_f32m2(output, v_offsets_u, v_updates, vl);
                    
                    j += vl;
                }
            }
        }
    } else if (axis == 1) {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            
            for (size_t i = i_tile; i < i_end; i++) {
                float* row_output = &output[i * data_cols];
                size_t j = 0;
                while (j < indices_cols) {
                    size_t vl = __riscv_vsetvl_e32m2(indices_cols - j);
                    
                    vint64m4_t v_indices_64 = __riscv_vle64_v_i64m4(&indices[i * indices_cols + j], vl);
                    vfloat32m2_t v_updates = __riscv_vle32_v_f32m2(&updates[i * indices_cols + j], vl);
                    
                    vint32m2_t v_indices_32 = __riscv_vnsra_wx_i32m2(v_indices_64, 0, vl);
                    vint32m2_t v_byte_offsets = __riscv_vsll_vx_i32m2(v_indices_32, 2, vl);
                    
                    vuint32m2_t v_offsets_u = __riscv_vreinterpret_v_i32m2_u32m2(v_byte_offsets);
                    __riscv_vsuxei32_v_f32m2(row_output, v_offsets_u, v_updates, vl);
                    
                    j += vl;
                }
            }
        }
    }
}

// Tiled vectorized implementation (LMUL=4)
void scatter_elements_tiled_e32m4(
    const float* data,
    const int64_t* indices,
    const float* updates,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis,
    size_t tile_size
) {
    memcpy(output, data, data_rows * data_cols * sizeof(float));
    
    if (axis == 0) {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            
            for (size_t i = i_tile; i < i_end; i++) {
                size_t j = 0;
                while (j < indices_cols) {
                    size_t vl = __riscv_vsetvl_e32m4(indices_cols - j);
                    
                    vint64m8_t v_indices_64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl);
                    vfloat32m4_t v_updates = __riscv_vle32_v_f32m4(&updates[i * indices_cols + j], vl);
                    
                    vint32m4_t v_indices_32 = __riscv_vnsra_wx_i32m4(v_indices_64, 0, vl);
                    vint32m4_t v_byte_offsets = __riscv_vmul_vx_i32m4(v_indices_32, data_cols * sizeof(float), vl);
                    vuint32m4_t v_vid = __riscv_vid_v_u32m4(vl);
                    vuint32m4_t v_col_idx = __riscv_vadd_vx_u32m4(v_vid, j, vl);
                    vuint32m4_t v_col_offsets_u = __riscv_vmul_vx_u32m4(v_col_idx, sizeof(float), vl);
                    vint32m4_t v_col_offsets = __riscv_vreinterpret_v_u32m4_i32m4(v_col_offsets_u);
                    vint32m4_t v_final_offsets = __riscv_vadd_vv_i32m4(v_byte_offsets, v_col_offsets, vl);
                    
                    vuint32m4_t v_offsets_u = __riscv_vreinterpret_v_i32m4_u32m4(v_final_offsets);
                    __riscv_vsuxei32_v_f32m4(output, v_offsets_u, v_updates, vl);
                    
                    j += vl;
                }
            }
        }
    } else if (axis == 1) {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            
            for (size_t i = i_tile; i < i_end; i++) {
                float* row_output = &output[i * data_cols];
                size_t j = 0;
                while (j < indices_cols) {
                    size_t vl = __riscv_vsetvl_e32m4(indices_cols - j);
                    
                    vint64m8_t v_indices_64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl);
                    vfloat32m4_t v_updates = __riscv_vle32_v_f32m4(&updates[i * indices_cols + j], vl);
                    
                    vint32m4_t v_indices_32 = __riscv_vnsra_wx_i32m4(v_indices_64, 0, vl);
                    vint32m4_t v_byte_offsets = __riscv_vsll_vx_i32m4(v_indices_32, 2, vl);
                    
                    vuint32m4_t v_offsets_u = __riscv_vreinterpret_v_i32m4_u32m4(v_byte_offsets);
                    __riscv_vsuxei32_v_f32m4(row_output, v_offsets_u, v_updates, vl);
                    
                    j += vl;
                }
            }
        }
    }
}

// Tiled vectorized implementation (LMUL=8)
void scatter_elements_tiled_e32m8(
    const float* data,
    const int64_t* indices,
    const float* updates,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis,
    size_t tile_size
) {
    memcpy(output, data, data_rows * data_cols * sizeof(float));
    
    if (axis == 0) {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            
            for (size_t i = i_tile; i < i_end; i++) {
                size_t j = 0;
                while (j < indices_cols) {
                    size_t vl = __riscv_vsetvl_e32m8(indices_cols - j);
                    size_t vl_half = vl / 2;
                    if (vl_half == 0) vl_half = 1;
                    
                    vint64m8_t v_indices_64_1 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl_half);
                    vfloat32m4_t v_updates_1 = __riscv_vle32_v_f32m4(&updates[i * indices_cols + j], vl_half);
                    
                    vint32m4_t v_indices_32_1 = __riscv_vnsra_wx_i32m4(v_indices_64_1, 0, vl_half);
                    vint32m4_t v_byte_offsets_1 = __riscv_vmul_vx_i32m4(v_indices_32_1, data_cols * sizeof(float), vl_half);
                    vuint32m4_t v_vid = __riscv_vid_v_u32m4(vl_half);
                    vuint32m4_t v_col_idx = __riscv_vadd_vx_u32m4(v_vid, j, vl_half);
                    vuint32m4_t v_col_offsets_u_1 = __riscv_vmul_vx_u32m4(v_col_idx, sizeof(float), vl_half);
                    vint32m4_t v_col_offsets_1 = __riscv_vreinterpret_v_u32m4_i32m4(v_col_offsets_u_1);
                    vint32m4_t v_final_offsets_1 = __riscv_vadd_vv_i32m4(v_byte_offsets_1, v_col_offsets_1, vl_half);
                    
                    vuint32m4_t v_offsets_u_1 = __riscv_vreinterpret_v_i32m4_u32m4(v_final_offsets_1);
                    __riscv_vsuxei32_v_f32m4(output, v_offsets_u_1, v_updates_1, vl_half);
                    
                    j += vl_half;
                }
            }
        }
    } else if (axis == 1) {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            
            for (size_t i = i_tile; i < i_end; i++) {
                float* row_output = &output[i * data_cols];
                size_t j = 0;
                while (j < indices_cols) {
                    size_t vl = __riscv_vsetvl_e32m8(indices_cols - j);
                    size_t vl_half = vl / 2;
                    if (vl_half == 0) vl_half = 1;
                    
                    vint64m8_t v_indices_64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl_half);
                    vfloat32m4_t v_updates = __riscv_vle32_v_f32m4(&updates[i * indices_cols + j], vl_half);
                    
                    vint32m4_t v_indices_32 = __riscv_vnsra_wx_i32m4(v_indices_64, 0, vl_half);
                    vint32m4_t v_byte_offsets = __riscv_vsll_vx_i32m4(v_indices_32, 2, vl_half);
                    
                    vuint32m4_t v_offsets_u = __riscv_vreinterpret_v_i32m4_u32m4(v_byte_offsets);
                    __riscv_vsuxei32_v_f32m4(row_output, v_offsets_u, v_updates, vl_half);
                    
                    j += vl_half;
                }
            }
        }
    }
}
