#include "../include/defs.h"
#include <riscv_vector.h>
#include <cstring>

// Gather: output has shape of indices (indices_rows x indices_cols)
// axis == 0: output[i, j] = data[ indices[i,j], j ]
// axis == 1: output[i, j] = data[ i, indices[i,j] ]

void gather_scalar(
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
            for (size_t j = 0; j < indices_cols; j++) {
                int64_t src_row = indices[i * indices_cols + j];
                if (src_row >= 0 && src_row < (int64_t)data_rows) {
                    output[i * indices_cols + j] = data[src_row * data_cols + j];
                } else {
                    output[i * indices_cols + j] = 0.0f;
                }
            }
        }
    } else { // axis == 1
        for (size_t i = 0; i < indices_rows; i++) {
            for (size_t j = 0; j < indices_cols; j++) {
                int64_t src_col = indices[i * indices_cols + j];
                if (src_col >= 0 && src_col < (int64_t)data_cols) {
                    output[i * indices_cols + j] = data[i * data_cols + src_col];
                } else {
                    output[i * indices_cols + j] = 0.0f;
                }
            }
        }
    }
}

// Vectorized gather using indexed loads (LMUL=1)
void gather_e32m1(
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
                size_t vl = __riscv_vsetvl_e32m1(indices_cols - j);

                // load 64-bit indices for this row segment
                vint64m2_t v_idx64 = __riscv_vle64_v_i64m2(&indices[i * indices_cols + j], vl);
                // narrow to 32-bit
                vint32m1_t v_idx32 = __riscv_vnsra_wx_i32m1(v_idx64, 0, vl);

                // row strides in bytes: index * data_cols * sizeof(float)
                vint32m1_t v_row_byte = __riscv_vmul_vx_i32m1(v_idx32, (int32_t)(data_cols * sizeof(float)), vl);

                // add column offsets: (j + vid) * sizeof(float)
                vuint32m1_t v_vid = __riscv_vid_v_u32m1(vl);
                vuint32m1_t v_col = __riscv_vadd_vx_u32m1(v_vid, (uint32_t)j, vl);
                vuint32m1_t v_col_byte_u = __riscv_vmul_vx_u32m1(v_col, (int32_t)sizeof(float), vl);
                vint32m1_t v_col_byte = __riscv_vreinterpret_v_u32m1_i32m1(v_col_byte_u);

                // final byte offsets relative to base data pointer
                vint32m1_t v_off = __riscv_vadd_vv_i32m1(v_row_byte, v_col_byte, vl);
                vuint32m1_t v_off_u = __riscv_vreinterpret_v_i32m1_u32m1(v_off);

                // indexed load and contiguous store to output row
                vfloat32m1_t v_vals = __riscv_vluxei32_v_f32m1(data, v_off_u, vl);
                __riscv_vse32_v_f32m1(&output[i * indices_cols + j], v_vals, vl);

                j += vl;
            }
        }
    } else { // axis == 1
        for (size_t i = 0; i < indices_rows; i++) {
            const float* row_data = &data[i * data_cols];
            size_t j = 0;
            while (j < indices_cols) {
                size_t vl = __riscv_vsetvl_e32m1(indices_cols - j);

                vint64m2_t v_idx64 = __riscv_vle64_v_i64m2(&indices[i * indices_cols + j], vl);
                vint32m1_t v_idx32 = __riscv_vnsra_wx_i32m1(v_idx64, 0, vl);

                // byte offsets within the row: idx * sizeof(float)
                vint32m1_t v_byte = __riscv_vsll_vx_i32m1(v_idx32, 2, vl);
                vuint32m1_t v_off_u = __riscv_vreinterpret_v_i32m1_u32m1(v_byte);

                vfloat32m1_t v_vals = __riscv_vluxei32_v_f32m1(row_data, v_off_u, vl);
                __riscv_vse32_v_f32m1(&output[i * indices_cols + j], v_vals, vl);

                j += vl;
            }
        }
    }
}

void gather_tiled_scalar(
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
            for (size_t j_tile = 0; j_tile < indices_cols; j_tile += tile_size) {
                size_t j_end = (j_tile + tile_size < indices_cols) ? j_tile + tile_size : indices_cols;
                for (size_t i = i_tile; i < i_end; i++) {
                    for (size_t j = j_tile; j < j_end; j++) {
                        int64_t src_row = indices[i * indices_cols + j];
                        if (src_row >= 0 && src_row < (int64_t)data_rows) {
                            output[i * indices_cols + j] = data[src_row * data_cols + j];
                        } else {
                            output[i * indices_cols + j] = 0.0f;
                        }
                    }
                }
            }
        }
    } else {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            for (size_t j_tile = 0; j_tile < indices_cols; j_tile += tile_size) {
                size_t j_end = (j_tile + tile_size < indices_cols) ? j_tile + tile_size : indices_cols;
                for (size_t i = i_tile; i < i_end; i++) {
                    for (size_t j = j_tile; j < j_end; j++) {
                        int64_t src_col = indices[i * indices_cols + j];
                        if (src_col >= 0 && src_col < (int64_t)data_cols) {
                            output[i * indices_cols + j] = data[i * data_cols + src_col];
                        } else {
                            output[i * indices_cols + j] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

void gather_tiled_e32m1(
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
                    vint64m2_t v_idx64 = __riscv_vle64_v_i64m2(&indices[i * indices_cols + j], vl);
                    vint32m1_t v_idx32 = __riscv_vnsra_wx_i32m1(v_idx64, 0, vl);
                    vint32m1_t v_row_byte = __riscv_vmul_vx_i32m1(v_idx32, (int32_t)(data_cols * sizeof(float)), vl);
                    vuint32m1_t v_vid = __riscv_vid_v_u32m1(vl);
                    vuint32m1_t v_col = __riscv_vadd_vx_u32m1(v_vid, (uint32_t)j, vl);
                    vuint32m1_t v_col_byte_u = __riscv_vmul_vx_u32m1(v_col, (int32_t)sizeof(float), vl);
                    vint32m1_t v_col_byte = __riscv_vreinterpret_v_u32m1_i32m1(v_col_byte_u);
                    vint32m1_t v_off = __riscv_vadd_vv_i32m1(v_row_byte, v_col_byte, vl);
                    vuint32m1_t v_off_u = __riscv_vreinterpret_v_i32m1_u32m1(v_off);
                    vfloat32m1_t v_vals = __riscv_vluxei32_v_f32m1(data, v_off_u, vl);
                    __riscv_vse32_v_f32m1(&output[i * indices_cols + j], v_vals, vl);
                    j += vl;
                }
            }
        }
    } else {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            for (size_t i = i_tile; i < i_end; i++) {
                const float* row_data = &data[i * data_cols];
                size_t j = 0;
                while (j < indices_cols) {
                    size_t vl = __riscv_vsetvl_e32m1(indices_cols - j);
                    vint64m2_t v_idx64 = __riscv_vle64_v_i64m2(&indices[i * indices_cols + j], vl);
                    vint32m1_t v_idx32 = __riscv_vnsra_wx_i32m1(v_idx64, 0, vl);
                    vint32m1_t v_byte = __riscv_vsll_vx_i32m1(v_idx32, 2, vl);
                    vuint32m1_t v_off_u = __riscv_vreinterpret_v_i32m1_u32m1(v_byte);
                    vfloat32m1_t v_vals = __riscv_vluxei32_v_f32m1(row_data, v_off_u, vl);
                    __riscv_vse32_v_f32m1(&output[i * indices_cols + j], v_vals, vl);
                    j += vl;
                }
            }
        }
    }
}

// LMUL=2
void gather_e32m2(
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
                vint64m4_t v_idx64 = __riscv_vle64_v_i64m4(&indices[i * indices_cols + j], vl);
                vint32m2_t v_idx32 = __riscv_vnsra_wx_i32m2(v_idx64, 0, vl);
                vint32m2_t v_row_byte = __riscv_vmul_vx_i32m2(v_idx32, (int32_t)(data_cols * sizeof(float)), vl);
                vuint32m2_t v_vid = __riscv_vid_v_u32m2(vl);
                vuint32m2_t v_col = __riscv_vadd_vx_u32m2(v_vid, (uint32_t)j, vl);
                vuint32m2_t v_col_byte_u = __riscv_vmul_vx_u32m2(v_col, (int32_t)sizeof(float), vl);
                vint32m2_t v_col_byte = __riscv_vreinterpret_v_u32m2_i32m2(v_col_byte_u);
                vint32m2_t v_off = __riscv_vadd_vv_i32m2(v_row_byte, v_col_byte, vl);
                vuint32m2_t v_off_u = __riscv_vreinterpret_v_i32m2_u32m2(v_off);
                vfloat32m2_t v_vals = __riscv_vluxei32_v_f32m2(data, v_off_u, vl);
                __riscv_vse32_v_f32m2(&output[i * indices_cols + j], v_vals, vl);
                j += vl;
            }
        }
    } else {
        for (size_t i = 0; i < indices_rows; i++) {
            const float* row_data = &data[i * data_cols];
            size_t j = 0;
            while (j < indices_cols) {
                size_t vl = __riscv_vsetvl_e32m2(indices_cols - j);
                vint64m4_t v_idx64 = __riscv_vle64_v_i64m4(&indices[i * indices_cols + j], vl);
                vint32m2_t v_idx32 = __riscv_vnsra_wx_i32m2(v_idx64, 0, vl);
                vint32m2_t v_byte = __riscv_vsll_vx_i32m2(v_idx32, 2, vl);
                vuint32m2_t v_off_u = __riscv_vreinterpret_v_i32m2_u32m2(v_byte);
                vfloat32m2_t v_vals = __riscv_vluxei32_v_f32m2(row_data, v_off_u, vl);
                __riscv_vse32_v_f32m2(&output[i * indices_cols + j], v_vals, vl);
                j += vl;
            }
        }
    }
}

void gather_e32m4(
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
                vint64m8_t v_idx64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl);
                vint32m4_t v_idx32 = __riscv_vnsra_wx_i32m4(v_idx64, 0, vl);
                vint32m4_t v_row_byte = __riscv_vmul_vx_i32m4(v_idx32, (int32_t)(data_cols * sizeof(float)), vl);
                vuint32m4_t v_vid = __riscv_vid_v_u32m4(vl);
                vuint32m4_t v_col = __riscv_vadd_vx_u32m4(v_vid, (uint32_t)j, vl);
                vuint32m4_t v_col_byte_u = __riscv_vmul_vx_u32m4(v_col, (int32_t)sizeof(float), vl);
                vint32m4_t v_col_byte = __riscv_vreinterpret_v_u32m4_i32m4(v_col_byte_u);
                vint32m4_t v_off = __riscv_vadd_vv_i32m4(v_row_byte, v_col_byte, vl);
                vuint32m4_t v_off_u = __riscv_vreinterpret_v_i32m4_u32m4(v_off);
                vfloat32m4_t v_vals = __riscv_vluxei32_v_f32m4(data, v_off_u, vl);
                __riscv_vse32_v_f32m4(&output[i * indices_cols + j], v_vals, vl);
                j += vl;
            }
        }
    } else {
        for (size_t i = 0; i < indices_rows; i++) {
            const float* row_data = &data[i * data_cols];
            size_t j = 0;
            while (j < indices_cols) {
                size_t vl = __riscv_vsetvl_e32m4(indices_cols - j);
                vint64m8_t v_idx64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl);
                vint32m4_t v_idx32 = __riscv_vnsra_wx_i32m4(v_idx64, 0, vl);
                vint32m4_t v_byte = __riscv_vsll_vx_i32m4(v_idx32, 2, vl);
                vuint32m4_t v_off_u = __riscv_vreinterpret_v_i32m4_u32m4(v_byte);
                vfloat32m4_t v_vals = __riscv_vluxei32_v_f32m4(row_data, v_off_u, vl);
                __riscv_vse32_v_f32m4(&output[i * indices_cols + j], v_vals, vl);
                j += vl;
            }
        }
    }
}

void gather_e32m8(
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
                size_t vl_half = vl / 2;
                if (vl_half == 0) vl_half = 1;
                vint64m8_t v_idx64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl_half);
                vfloat32m4_t v_vals;
                vint32m4_t v_idx32 = __riscv_vnsra_wx_i32m4(v_idx64, 0, vl_half);
                vint32m4_t v_row_byte = __riscv_vmul_vx_i32m4(v_idx32, (int32_t)(data_cols * sizeof(float)), vl_half);
                vuint32m4_t v_vid = __riscv_vid_v_u32m4(vl_half);
                vuint32m4_t v_col = __riscv_vadd_vx_u32m4(v_vid, (uint32_t)j, vl_half);
                vuint32m4_t v_col_byte_u = __riscv_vmul_vx_u32m4(v_col, (int32_t)sizeof(float), vl_half);
                vint32m4_t v_col_byte = __riscv_vreinterpret_v_u32m4_i32m4(v_col_byte_u);
                vint32m4_t v_off = __riscv_vadd_vv_i32m4(v_row_byte, v_col_byte, vl_half);
                vuint32m4_t v_off_u = __riscv_vreinterpret_v_i32m4_u32m4(v_off);
                v_vals = __riscv_vluxei32_v_f32m4(data, v_off_u, vl_half);
                __riscv_vse32_v_f32m4(&output[i * indices_cols + j], v_vals, vl_half);
                j += vl_half;
            }
        }
    } else {
        for (size_t i = 0; i < indices_rows; i++) {
            const float* row_data = &data[i * data_cols];
            size_t j = 0;
            while (j < indices_cols) {
                size_t vl = __riscv_vsetvl_e32m8(indices_cols - j);
                size_t vl_half = vl / 2;
                if (vl_half == 0) vl_half = 1;
                vint64m8_t v_idx64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl_half);
                vfloat32m4_t v_vals;
                vint32m4_t v_idx32 = __riscv_vnsra_wx_i32m4(v_idx64, 0, vl_half);
                vint32m4_t v_byte = __riscv_vsll_vx_i32m4(v_idx32, 2, vl_half);
                vuint32m4_t v_off_u = __riscv_vreinterpret_v_i32m4_u32m4(v_byte);
                v_vals = __riscv_vluxei32_v_f32m4(row_data, v_off_u, vl_half);
                __riscv_vse32_v_f32m4(&output[i * indices_cols + j], v_vals, vl_half);
                j += vl_half;
            }
        }
    }
}

void gather_tiled_e32m2(
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
                    vint64m4_t v_idx64 = __riscv_vle64_v_i64m4(&indices[i * indices_cols + j], vl);
                    vint32m2_t v_idx32 = __riscv_vnsra_wx_i32m2(v_idx64, 0, vl);
                    vint32m2_t v_row_byte = __riscv_vmul_vx_i32m2(v_idx32, (int32_t)(data_cols * sizeof(float)), vl);
                    vuint32m2_t v_vid = __riscv_vid_v_u32m2(vl);
                    vuint32m2_t v_col = __riscv_vadd_vx_u32m2(v_vid, (uint32_t)j, vl);
                    vuint32m2_t v_col_byte_u = __riscv_vmul_vx_u32m2(v_col, (int32_t)sizeof(float), vl);
                    vint32m2_t v_col_byte = __riscv_vreinterpret_v_u32m2_i32m2(v_col_byte_u);
                    vint32m2_t v_off = __riscv_vadd_vv_i32m2(v_row_byte, v_col_byte, vl);
                    vuint32m2_t v_off_u = __riscv_vreinterpret_v_i32m2_u32m2(v_off);
                    vfloat32m2_t v_vals = __riscv_vluxei32_v_f32m2(data, v_off_u, vl);
                    __riscv_vse32_v_f32m2(&output[i * indices_cols + j], v_vals, vl);
                    j += vl;
                }
            }
        }
    } else {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            for (size_t i = i_tile; i < i_end; i++) {
                const float* row_data = &data[i * data_cols];
                size_t j = 0;
                while (j < indices_cols) {
                    size_t vl = __riscv_vsetvl_e32m2(indices_cols - j);
                    vint64m4_t v_idx64 = __riscv_vle64_v_i64m4(&indices[i * indices_cols + j], vl);
                    vint32m2_t v_idx32 = __riscv_vnsra_wx_i32m2(v_idx64, 0, vl);
                    vint32m2_t v_byte = __riscv_vsll_vx_i32m2(v_idx32, 2, vl);
                    vuint32m2_t v_off_u = __riscv_vreinterpret_v_i32m2_u32m2(v_byte);
                    vfloat32m2_t v_vals = __riscv_vluxei32_v_f32m2(row_data, v_off_u, vl);
                    __riscv_vse32_v_f32m2(&output[i * indices_cols + j], v_vals, vl);
                    j += vl;
                }
            }
        }
    }
}

void gather_tiled_e32m4(
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
                    vint64m8_t v_idx64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl);
                    vint32m4_t v_idx32 = __riscv_vnsra_wx_i32m4(v_idx64, 0, vl);
                    vint32m4_t v_row_byte = __riscv_vmul_vx_i32m4(v_idx32, (int32_t)(data_cols * sizeof(float)), vl);
                    vuint32m4_t v_vid = __riscv_vid_v_u32m4(vl);
                    vuint32m4_t v_col = __riscv_vadd_vx_u32m4(v_vid, (uint32_t)j, vl);
                    vuint32m4_t v_col_byte_u = __riscv_vmul_vx_u32m4(v_col, (int32_t)sizeof(float), vl);
                    vint32m4_t v_col_byte = __riscv_vreinterpret_v_u32m4_i32m4(v_col_byte_u);
                    vint32m4_t v_off = __riscv_vadd_vv_i32m4(v_row_byte, v_col_byte, vl);
                    vuint32m4_t v_off_u = __riscv_vreinterpret_v_i32m4_u32m4(v_off);
                    vfloat32m4_t v_vals = __riscv_vluxei32_v_f32m4(data, v_off_u, vl);
                    __riscv_vse32_v_f32m4(&output[i * indices_cols + j], v_vals, vl);
                    j += vl;
                }
            }
        }
    } else {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            for (size_t i = i_tile; i < i_end; i++) {
                const float* row_data = &data[i * data_cols];
                size_t j = 0;
                while (j < indices_cols) {
                    size_t vl = __riscv_vsetvl_e32m4(indices_cols - j);
                    vint64m8_t v_idx64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl);
                    vint32m4_t v_idx32 = __riscv_vnsra_wx_i32m4(v_idx64, 0, vl);
                    vint32m4_t v_byte = __riscv_vsll_vx_i32m4(v_idx32, 2, vl);
                    vuint32m4_t v_off_u = __riscv_vreinterpret_v_i32m4_u32m4(v_byte);
                    vfloat32m4_t v_vals = __riscv_vluxei32_v_f32m4(row_data, v_off_u, vl);
                    __riscv_vse32_v_f32m4(&output[i * indices_cols + j], v_vals, vl);
                    j += vl;
                }
            }
        }
    }
}

void gather_tiled_e32m8(
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
                    size_t vl = __riscv_vsetvl_e32m8(indices_cols - j);
                    size_t vl_half = vl / 2;
                    if (vl_half == 0) vl_half = 1;
                    vint64m8_t v_idx64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl_half);
                    vint32m4_t v_idx32 = __riscv_vnsra_wx_i32m4(v_idx64, 0, vl_half);
                    vint32m4_t v_row_byte = __riscv_vmul_vx_i32m4(v_idx32, (int32_t)(data_cols * sizeof(float)), vl_half);
                    vuint32m4_t v_vid = __riscv_vid_v_u32m4(vl_half);
                    vuint32m4_t v_col = __riscv_vadd_vx_u32m4(v_vid, (uint32_t)j, vl_half);
                    vuint32m4_t v_col_byte_u = __riscv_vmul_vx_u32m4(v_col, (int32_t)sizeof(float), vl_half);
                    vint32m4_t v_col_byte = __riscv_vreinterpret_v_u32m4_i32m4(v_col_byte_u);
                    vint32m4_t v_off = __riscv_vadd_vv_i32m4(v_row_byte, v_col_byte, vl_half);
                    vuint32m4_t v_off_u = __riscv_vreinterpret_v_i32m4_u32m4(v_off);
                    vfloat32m4_t v_vals = __riscv_vluxei32_v_f32m4(data, v_off_u, vl_half);
                    __riscv_vse32_v_f32m4(&output[i * indices_cols + j], v_vals, vl_half);
                    j += vl_half;
                }
            }
        }
    } else {
        for (size_t i_tile = 0; i_tile < indices_rows; i_tile += tile_size) {
            size_t i_end = (i_tile + tile_size < indices_rows) ? i_tile + tile_size : indices_rows;
            for (size_t i = i_tile; i < i_end; i++) {
                const float* row_data = &data[i * data_cols];
                size_t j = 0;
                while (j < indices_cols) {
                    size_t vl = __riscv_vsetvl_e32m8(indices_cols - j);
                    size_t vl_half = vl / 2;
                    if (vl_half == 0) vl_half = 1;
                    vint64m8_t v_idx64 = __riscv_vle64_v_i64m8(&indices[i * indices_cols + j], vl_half);
                    vint32m4_t v_idx32 = __riscv_vnsra_wx_i32m4(v_idx64, 0, vl_half);
                    vint32m4_t v_byte = __riscv_vsll_vx_i32m4(v_idx32, 2, vl_half);
                    vuint32m4_t v_off_u = __riscv_vreinterpret_v_i32m4_u32m4(v_byte);
                    vfloat32m4_t v_vals = __riscv_vluxei32_v_f32m4(row_data, v_off_u, vl_half);
                    __riscv_vse32_v_f32m4(&output[i * indices_cols + j], v_vals, vl_half);
                    j += vl_half;
                }
            }
        }
    }
}


