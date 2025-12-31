#include "../include/defs.h"
#include <cstring>
#include <riscv_vector.h>

/*********************************** Scalar ************************************/
void gather_elements_scalar(
    const float* data,
    const int64_t* indices,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis
) {
    if (axis == 0) {
        // Gather along rows
        for (size_t i = 0; i < indices_rows; i++) {
            for (size_t j = 0; j < indices_cols; j++) {
                int64_t source_row = indices[i * indices_cols + j];
                if (source_row >= 0 && source_row < (int64_t)data_rows) {
                    output[i * indices_cols + j] = data[source_row * data_cols + j];
                } else {
                    output[i * indices_cols + j] = 0.0f;
                }
            }
        }
    } else if (axis == 1) {
        // Gather along columns
        for (size_t i = 0; i < indices_rows; i++) {
            for (size_t j = 0; j < indices_cols; j++) {
                int64_t source_col = indices[i * indices_cols + j];
                if (source_col >= 0 && source_col < (int64_t)data_cols) {
                    output[i * indices_cols + j] = data[i * data_cols + source_col];
                } else {
                    output[i * indices_cols + j] = 0.0f;
                }
            }
        }
    }
}

/****************************** Vectorized ******************************/
void gather_elements_e32m1(
    const float* data,
    const int64_t* indices,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis
) {
    if (axis == 0) {
        // Gather along rows - vectorize across columns
        for (size_t i = 0; i < indices_rows; i++) {
            size_t j = 0;
            while (j < indices_cols) {
                size_t vl = __riscv_vsetvl_e32m1(indices_cols - j);
                
                // Load indices
                vint64m2_t v_indices_64 = __riscv_vle64_v_i64m2(&indices[i * indices_cols + j], vl);
                
                // Convert 64-bit indices to 32-bit and scale by stride
                vint32m1_t v_indices_32 = __riscv_vnsra_wx_i32m1(v_indices_64, 0, vl);
                vint32m1_t v_byte_offsets = __riscv_vmul_vx_i32m1(v_indices_32, data_cols * sizeof(float), vl);
                
                // Add column offset
                vuint32m1_t v_vid = __riscv_vid_v_u32m1(vl);
                vuint32m1_t v_col_idx = __riscv_vadd_vx_u32m1(v_vid, j, vl);
                vuint32m1_t v_col_offsets_u = __riscv_vmul_vx_u32m1(v_col_idx, sizeof(float), vl);
                vint32m1_t v_col_offsets = __riscv_vreinterpret_v_u32m1_i32m1(v_col_offsets_u);
                vint32m1_t v_final_offsets = __riscv_vadd_vv_i32m1(v_byte_offsets, v_col_offsets, vl);
                
                // Indexed load
                vuint32m1_t v_offsets_u = __riscv_vreinterpret_v_i32m1_u32m1(v_final_offsets);
                vfloat32m1_t v_gathered = __riscv_vluxei32_v_f32m1(data, v_offsets_u, vl);
                
                // Store result
                __riscv_vse32_v_f32m1(&output[i * indices_cols + j], v_gathered, vl);
                
                j += vl;
            }
        }
    } else if (axis == 1) {
        // Gather along columns
        for (size_t i = 0; i < indices_rows; i++) {
            const float* row_data = &data[i * data_cols];
            size_t j = 0;
            while (j < indices_cols) {
                size_t vl = __riscv_vsetvl_e32m1(indices_cols - j);
                
                // Load indices
                vint64m2_t v_indices_64 = __riscv_vle64_v_i64m2(&indices[i * indices_cols + j], vl);
                
                // Convert to byte offsets
                vint32m1_t v_indices_32 = __riscv_vnsra_wx_i32m1(v_indices_64, 0, vl);
                vint32m1_t v_byte_offsets = __riscv_vmul_vx_i32m1(v_indices_32, sizeof(float), vl);
                vuint32m1_t v_offsets_u = __riscv_vreinterpret_v_i32m1_u32m1(v_byte_offsets);
                
                // Indexed load
                vfloat32m1_t v_gathered = __riscv_vluxei32_v_f32m1(row_data, v_offsets_u, vl);
                
                // Store result
                __riscv_vse32_v_f32m1(&output[i * indices_cols + j], v_gathered, vl);
                
                j += vl;
            }
        }
    }
}

// LMUL=2 version
void gather_elements_e32m2(
    const float* data,
    const int64_t* indices,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis
) {
    if (axis == 0) {
        for (size_t i = 0; i < indices_rows; i++) {
            size_t j = 0;
            while (j < indices_cols) {
                size_t vl = __riscv_vsetvl_e32m2(indices_cols - j);
                
                vint64m4_t v_indices_64 = __riscv_vle64_v_i64m4(&indices[i * indices_cols + j], vl);
                vint32m2_t v_indices_32 = __riscv_vnsra_wx_i32m2(v_indices_64, 0, vl);
                vint32m2_t v_byte_offsets = __riscv_vmul_vx_i32m2(v_indices_32, data_cols * sizeof(float), vl);
                
                vuint32m2_t v_vid = __riscv_vid_v_u32m2(vl);
                vuint32m2_t v_col_idx = __riscv_vadd_vx_u32m2(v_vid, j, vl);
                vuint32m2_t v_col_offsets_u = __riscv_vmul_vx_u32m2(v_col_idx, sizeof(float), vl);
                vint32m2_t v_col_offsets = __riscv_vreinterpret_v_u32m2_i32m2(v_col_offsets_u);
                vint32m2_t v_final_offsets = __riscv_vadd_vv_i32m2(v_byte_offsets, v_col_offsets, vl);
                
                vuint32m2_t v_offsets_u = __riscv_vreinterpret_v_i32m2_u32m2(v_final_offsets);
                vfloat32m2_t v_gathered = __riscv_vluxei32_v_f32m2(data, v_offsets_u, vl);
                __riscv_vse32_v_f32m2(&output[i * indices_cols + j], v_gathered, vl);
                
                j += vl;
            }
        }
    } else if (axis == 1) {
        for (size_t i = 0; i < indices_rows; i++) {
            const float* row_data = &data[i * data_cols];
            size_t j = 0;
            while (j < indices_cols) {
                size_t vl = __riscv_vsetvl_e32m2(indices_cols - j);
                
                vint64m4_t v_indices_64 = __riscv_vle64_v_i64m4(&indices[i * indices_cols + j], vl);
                vint32m2_t v_indices_32 = __riscv_vnsra_wx_i32m2(v_indices_64, 0, vl);
                vint32m2_t v_byte_offsets = __riscv_vmul_vx_i32m2(v_indices_32, sizeof(float), vl);
                vuint32m2_t v_offsets_u = __riscv_vreinterpret_v_i32m2_u32m2(v_byte_offsets);
                
                vfloat32m2_t v_gathered = __riscv_vluxei32_v_f32m2(row_data, v_offsets_u, vl);
                __riscv_vse32_v_f32m2(&output[i * indices_cols + j], v_gathered, vl);
                
                j += vl;
            }
        }
    }
}

// LMUL=4 version
void gather_elements_e32m4(
    const float* data,
    const int64_t* indices,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis
) {
    if (axis == 0) {
        for (size_t i = 0; i < indices_rows; i++) {
            size_t j = 0;
            while (j < indices_cols) {
                size_t vl = __riscv_vsetvl_e32m4(indices_cols - j);
                
                vint64m8_t v_indices_64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl);
                vint32m4_t v_indices_32 = __riscv_vnsra_wx_i32m4(v_indices_64, 0, vl);
                vint32m4_t v_byte_offsets = __riscv_vmul_vx_i32m4(v_indices_32, data_cols * sizeof(float), vl);
                
                vuint32m4_t v_vid = __riscv_vid_v_u32m4(vl);
                vuint32m4_t v_col_idx = __riscv_vadd_vx_u32m4(v_vid, j, vl);
                vuint32m4_t v_col_offsets_u = __riscv_vmul_vx_u32m4(v_col_idx, sizeof(float), vl);
                vint32m4_t v_col_offsets = __riscv_vreinterpret_v_u32m4_i32m4(v_col_offsets_u);
                vint32m4_t v_final_offsets = __riscv_vadd_vv_i32m4(v_byte_offsets, v_col_offsets, vl);
                
                vuint32m4_t v_offsets_u = __riscv_vreinterpret_v_i32m4_u32m4(v_final_offsets);
                vfloat32m4_t v_gathered = __riscv_vluxei32_v_f32m4(data, v_offsets_u, vl);
                __riscv_vse32_v_f32m4(&output[i * indices_cols + j], v_gathered, vl);
                
                j += vl;
            }
        }
    } else if (axis == 1) {
        for (size_t i = 0; i < indices_rows; i++) {
            const float* row_data = &data[i * data_cols];
            size_t j = 0;
            while (j < indices_cols) {
                size_t vl = __riscv_vsetvl_e32m4(indices_cols - j);
                
                vint64m8_t v_indices_64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl);
                vint32m4_t v_indices_32 = __riscv_vnsra_wx_i32m4(v_indices_64, 0, vl);
                vint32m4_t v_byte_offsets = __riscv_vmul_vx_i32m4(v_indices_32, sizeof(float), vl);
                vuint32m4_t v_offsets_u = __riscv_vreinterpret_v_i32m4_u32m4(v_byte_offsets);
                
                vfloat32m4_t v_gathered = __riscv_vluxei32_v_f32m4(row_data, v_offsets_u, vl);
                __riscv_vse32_v_f32m4(&output[i * indices_cols + j], v_gathered, vl);
                
                j += vl;
            }
        }
    }
}

// LMUL=8 version
void gather_elements_e32m8(
    const float* data,
    const int64_t* indices,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis
) {
    if (axis == 0) {
        for (size_t i = 0; i < indices_rows; i++) {
            size_t j = 0;
            while (j < indices_cols) {
                size_t vl = __riscv_vsetvl_e32m8(indices_cols - j);
                
                // For e32m8, we need to process indices in chunks due to LMUL limits
                // Load indices as i64 (would need m16 which doesn't exist)
                // So we fall back to smaller chunks
                size_t vl_idx = __riscv_vsetvl_e64m8(indices_cols - j);
                vint64m8_t v_indices_64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl_idx);
                vint32m4_t v_indices_32 = __riscv_vnsra_wx_i32m4(v_indices_64, 0, vl_idx);
                vint32m4_t v_byte_offsets = __riscv_vmul_vx_i32m4(v_indices_32, data_cols * sizeof(float), vl_idx);
                
                vuint32m4_t v_vid = __riscv_vid_v_u32m4(vl_idx);
                vuint32m4_t v_col_idx = __riscv_vadd_vx_u32m4(v_vid, j, vl_idx);
                vuint32m4_t v_col_offsets_u = __riscv_vmul_vx_u32m4(v_col_idx, sizeof(float), vl_idx);
                vint32m4_t v_col_offsets = __riscv_vreinterpret_v_u32m4_i32m4(v_col_offsets_u);
                vint32m4_t v_final_offsets = __riscv_vadd_vv_i32m4(v_byte_offsets, v_col_offsets, vl_idx);
                
                vuint32m4_t v_offsets_u = __riscv_vreinterpret_v_i32m4_u32m4(v_final_offsets);
                vfloat32m4_t v_gathered = __riscv_vluxei32_v_f32m4(data, v_offsets_u, vl_idx);
                __riscv_vse32_v_f32m4(&output[i * indices_cols + j], v_gathered, vl_idx);
                
                j += vl_idx;
            }
        }
    } else if (axis == 1) {
        for (size_t i = 0; i < indices_rows; i++) {
            const float* row_data = &data[i * data_cols];
            size_t j = 0;
            while (j < indices_cols) {
                size_t vl_idx = __riscv_vsetvl_e64m8(indices_cols - j);
                
                vint64m8_t v_indices_64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl_idx);
                vint32m4_t v_indices_32 = __riscv_vnsra_wx_i32m4(v_indices_64, 0, vl_idx);
                vint32m4_t v_byte_offsets = __riscv_vmul_vx_i32m4(v_indices_32, sizeof(float), vl_idx);
                vuint32m4_t v_offsets_u = __riscv_vreinterpret_v_i32m4_u32m4(v_byte_offsets);
                
                vfloat32m4_t v_gathered = __riscv_vluxei32_v_f32m4(row_data, v_offsets_u, vl_idx);
                __riscv_vse32_v_f32m4(&output[i * indices_cols + j], v_gathered, vl_idx);
                
                j += vl_idx;
            }
        }
    }
}

/******************************** Tiled Scalar *********************************/

void gather_elements_tiled_scalar(
    const float* data,
    const int64_t* indices,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis,
    size_t tile_size
) {
    if (axis == 0) {
        // Process in tiles for better cache locality
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            for (size_t j_tile = 0; j_tile < indices_cols; j_tile += tile_size) {
                size_t j_end = (j_tile + tile_size < indices_cols) ? j_tile + tile_size : indices_cols;
                
                for (size_t i = i_tile; i < i_end; i++) {
                    for (size_t j = j_tile; j < j_end; j++) {
                        int64_t source_row = indices[i * indices_cols + j];
                        if (source_row >= 0 && source_row < (int64_t)data_rows) {
                            output[i * indices_cols + j] = data[source_row * data_cols + j];
                        } else {
                            output[i * indices_cols + j] = 0.0f;
                        }
                    }
                }
            }
        }
    } else if (axis == 1) {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            for (size_t j_tile = 0; j_tile < indices_cols; j_tile += tile_size) {
                size_t j_end = (j_tile + tile_size < indices_cols) ? j_tile + tile_size : indices_cols;
                
                for (size_t i = i_tile; i < i_end; i++) {
                    for (size_t j = j_tile; j < j_end; j++) {
                        int64_t source_col = indices[i * indices_cols + j];
                        if (source_col >= 0 && source_col < (int64_t)data_cols) {
                            output[i * indices_cols + j] = data[i * data_cols + source_col];
                        } else {
                            output[i * indices_cols + j] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

/****************************** Tiled Vectorized ******************************/

void gather_elements_tiled_e32m1(
    const float* data,
    const int64_t* indices,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis,
    size_t tile_size
) {
    if (axis == 0) {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            
            for (size_t i = i_tile; i < i_end; i++) {
                size_t j = 0;
                while (j < indices_cols) {
                    size_t vl = __riscv_vsetvl_e32m1(indices_cols - j);
                    
                    vint64m2_t v_indices_64 = __riscv_vle64_v_i64m2(&indices[i * indices_cols + j], vl);
                    vint32m1_t v_indices_32 = __riscv_vnsra_wx_i32m1(v_indices_64, 0, vl);
                    vint32m1_t v_byte_offsets = __riscv_vmul_vx_i32m1(v_indices_32, data_cols * sizeof(float), vl);
                    
                    vuint32m1_t v_vid = __riscv_vid_v_u32m1(vl);
                    vuint32m1_t v_col_idx = __riscv_vadd_vx_u32m1(v_vid, j, vl);
                    vuint32m1_t v_col_offsets_u = __riscv_vmul_vx_u32m1(v_col_idx, sizeof(float), vl);
                    vint32m1_t v_col_offsets = __riscv_vreinterpret_v_u32m1_i32m1(v_col_offsets_u);
                    vint32m1_t v_final_offsets = __riscv_vadd_vv_i32m1(v_byte_offsets, v_col_offsets, vl);
                    
                    vuint32m1_t v_offsets_u = __riscv_vreinterpret_v_i32m1_u32m1(v_final_offsets);
                    vfloat32m1_t v_gathered = __riscv_vluxei32_v_f32m1(data, v_offsets_u, vl);
                    __riscv_vse32_v_f32m1(&output[i * indices_cols + j], v_gathered, vl);
                    
                    j += vl;
                }
            }
        }
    } else if (axis == 1) {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            
            for (size_t i = i_tile; i < i_end; i++) {
                const float* row_data = &data[i * data_cols];
                size_t j = 0;
                while (j < indices_cols) {
                    size_t vl = __riscv_vsetvl_e32m1(indices_cols - j);
                    
                    vint64m2_t v_indices_64 = __riscv_vle64_v_i64m2(&indices[i * indices_cols + j], vl);
                    vint32m1_t v_indices_32 = __riscv_vnsra_wx_i32m1(v_indices_64, 0, vl);
                    vint32m1_t v_byte_offsets = __riscv_vmul_vx_i32m1(v_indices_32, sizeof(float), vl);
                    vuint32m1_t v_offsets_u = __riscv_vreinterpret_v_i32m1_u32m1(v_byte_offsets);
                    
                    vfloat32m1_t v_gathered = __riscv_vluxei32_v_f32m1(row_data, v_offsets_u, vl);
                    __riscv_vse32_v_f32m1(&output[i * indices_cols + j], v_gathered, vl);
                    
                    j += vl;
                }
            }
        }
    }
}

// Tiled vectorized implementation (LMUL=2)
void gather_elements_tiled_e32m2(
    const float* data,
    const int64_t* indices,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis,
    size_t tile_size
) {
    if (axis == 0) {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            
            for (size_t i = i_tile; i < i_end; i++) {
                size_t j = 0;
                while (j < indices_cols) {
                    size_t vl = __riscv_vsetvl_e32m2(indices_cols - j);
                    
                    vint64m4_t v_indices_64 = __riscv_vle64_v_i64m4(&indices[i * indices_cols + j], vl);
                    vint32m2_t v_indices_32 = __riscv_vnsra_wx_i32m2(v_indices_64, 0, vl);
                    vint32m2_t v_byte_offsets = __riscv_vmul_vx_i32m2(v_indices_32, data_cols * sizeof(float), vl);
                    
                    vuint32m2_t v_vid = __riscv_vid_v_u32m2(vl);
                    vuint32m2_t v_col_idx = __riscv_vadd_vx_u32m2(v_vid, j, vl);
                    vuint32m2_t v_col_offsets_u = __riscv_vmul_vx_u32m2(v_col_idx, sizeof(float), vl);
                    vint32m2_t v_col_offsets = __riscv_vreinterpret_v_u32m2_i32m2(v_col_offsets_u);
                    vint32m2_t v_final_offsets = __riscv_vadd_vv_i32m2(v_byte_offsets, v_col_offsets, vl);
                    
                    vuint32m2_t v_offsets_u = __riscv_vreinterpret_v_i32m2_u32m2(v_final_offsets);
                    vfloat32m2_t v_gathered = __riscv_vluxei32_v_f32m2(data, v_offsets_u, vl);
                    __riscv_vse32_v_f32m2(&output[i * indices_cols + j], v_gathered, vl);
                    
                    j += vl;
                }
            }
        }
    } else if (axis == 1) {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            
            for (size_t i = i_tile; i < i_end; i++) {
                const float* row_data = &data[i * data_cols];
                size_t j = 0;
                while (j < indices_cols) {
                    size_t vl = __riscv_vsetvl_e32m2(indices_cols - j);
                    
                    vint64m4_t v_indices_64 = __riscv_vle64_v_i64m4(&indices[i * indices_cols + j], vl);
                    vint32m2_t v_indices_32 = __riscv_vnsra_wx_i32m2(v_indices_64, 0, vl);
                    vint32m2_t v_byte_offsets = __riscv_vmul_vx_i32m2(v_indices_32, sizeof(float), vl);
                    vuint32m2_t v_offsets_u = __riscv_vreinterpret_v_i32m2_u32m2(v_byte_offsets);
                    
                    vfloat32m2_t v_gathered = __riscv_vluxei32_v_f32m2(row_data, v_offsets_u, vl);
                    __riscv_vse32_v_f32m2(&output[i * indices_cols + j], v_gathered, vl);
                    
                    j += vl;
                }
            }
        }
    }
}

// Tiled vectorized implementation (LMUL=4)
void gather_elements_tiled_e32m4(
    const float* data,
    const int64_t* indices,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis,
    size_t tile_size
) {
    if (axis == 0) {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            
            for (size_t i = i_tile; i < i_end; i++) {
                size_t j = 0;
                while (j < indices_cols) {
                    size_t vl = __riscv_vsetvl_e32m4(indices_cols - j);
                    
                    vint64m8_t v_indices_64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl);
                    vint32m4_t v_indices_32 = __riscv_vnsra_wx_i32m4(v_indices_64, 0, vl);
                    vint32m4_t v_byte_offsets = __riscv_vmul_vx_i32m4(v_indices_32, data_cols * sizeof(float), vl);
                    
                    vuint32m4_t v_vid = __riscv_vid_v_u32m4(vl);
                    vuint32m4_t v_col_idx = __riscv_vadd_vx_u32m4(v_vid, j, vl);
                    vuint32m4_t v_col_offsets_u = __riscv_vmul_vx_u32m4(v_col_idx, sizeof(float), vl);
                    vint32m4_t v_col_offsets = __riscv_vreinterpret_v_u32m4_i32m4(v_col_offsets_u);
                    vint32m4_t v_final_offsets = __riscv_vadd_vv_i32m4(v_byte_offsets, v_col_offsets, vl);
                    
                    vuint32m4_t v_offsets_u = __riscv_vreinterpret_v_i32m4_u32m4(v_final_offsets);
                    vfloat32m4_t v_gathered = __riscv_vluxei32_v_f32m4(data, v_offsets_u, vl);
                    __riscv_vse32_v_f32m4(&output[i * indices_cols + j], v_gathered, vl);
                    
                    j += vl;
                }
            }
        }
    } else if (axis == 1) {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            
            for (size_t i = i_tile; i < i_end; i++) {
                const float* row_data = &data[i * data_cols];
                size_t j = 0;
                while (j < indices_cols) {
                    size_t vl = __riscv_vsetvl_e32m4(indices_cols - j);
                    
                    vint64m8_t v_indices_64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl);
                    vint32m4_t v_indices_32 = __riscv_vnsra_wx_i32m4(v_indices_64, 0, vl);
                    vint32m4_t v_byte_offsets = __riscv_vmul_vx_i32m4(v_indices_32, sizeof(float), vl);
                    vuint32m4_t v_offsets_u = __riscv_vreinterpret_v_i32m4_u32m4(v_byte_offsets);
                    
                    vfloat32m4_t v_gathered = __riscv_vluxei32_v_f32m4(row_data, v_offsets_u, vl);
                    __riscv_vse32_v_f32m4(&output[i * indices_cols + j], v_gathered, vl);
                    
                    j += vl;
                }
            }
        }
    }
}

// Tiled vectorized implementation (LMUL=8)
void gather_elements_tiled_e32m8(
    const float* data,
    const int64_t* indices,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis,
    size_t tile_size
) {
    if (axis == 0) {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            
            for (size_t i = i_tile; i < i_end; i++) {
                size_t j = 0;
                while (j < indices_cols) {
                    size_t vl_idx = __riscv_vsetvl_e64m8(indices_cols - j);
                    
                    vint64m8_t v_indices_64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl_idx);
                    vint32m4_t v_indices_32 = __riscv_vnsra_wx_i32m4(v_indices_64, 0, vl_idx);
                    vint32m4_t v_byte_offsets = __riscv_vmul_vx_i32m4(v_indices_32, data_cols * sizeof(float), vl_idx);
                    
                    vuint32m4_t v_vid = __riscv_vid_v_u32m4(vl_idx);
                    vuint32m4_t v_col_idx = __riscv_vadd_vx_u32m4(v_vid, j, vl_idx);
                    vuint32m4_t v_col_offsets_u = __riscv_vmul_vx_u32m4(v_col_idx, sizeof(float), vl_idx);
                    vint32m4_t v_col_offsets = __riscv_vreinterpret_v_u32m4_i32m4(v_col_offsets_u);
                    vint32m4_t v_final_offsets = __riscv_vadd_vv_i32m4(v_byte_offsets, v_col_offsets, vl_idx);
                    
                    vuint32m4_t v_offsets_u = __riscv_vreinterpret_v_i32m4_u32m4(v_final_offsets);
                    vfloat32m4_t v_gathered = __riscv_vluxei32_v_f32m4(data, v_offsets_u, vl_idx);
                    __riscv_vse32_v_f32m4(&output[i * indices_cols + j], v_gathered, vl_idx);
                    
                    j += vl_idx;
                }
            }
        }
    } else if (axis == 1) {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            
            for (size_t i = i_tile; i < i_end; i++) {
                const float* row_data = &data[i * data_cols];
                size_t j = 0;
                while (j < indices_cols) {
                    size_t vl_idx = __riscv_vsetvl_e64m8(indices_cols - j);
                    
                    vint64m8_t v_indices_64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl_idx);
                    vint32m4_t v_indices_32 = __riscv_vnsra_wx_i32m4(v_indices_64, 0, vl_idx);
                    vint32m4_t v_byte_offsets = __riscv_vmul_vx_i32m4(v_indices_32, sizeof(float), vl_idx);
                    vuint32m4_t v_offsets_u = __riscv_vreinterpret_v_i32m4_u32m4(v_byte_offsets);
                    
                    vfloat32m4_t v_gathered = __riscv_vluxei32_v_f32m4(row_data, v_offsets_u, vl_idx);
                    __riscv_vse32_v_f32m4(&output[i * indices_cols + j], v_gathered, vl_idx);
                    
                    j += vl_idx;
                }
            }
        }
    }
}
