#include "../include/defs.h"
#include "rvv_defs.hpp"
#include <riscv_vector.h>
#include <cstring>

// Gather: output has shape of indices (indices_rows x indices_cols)
// axis == 0: output[i, j] = data[ indices[i,j], j ]
// axis == 1: output[i, j] = data[ i, indices[i,j] ]

/****************************** Scalar ******************************/
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
/****************************** Vectorized ******************************/
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
				size_t vl = SET_VECTOR_LENGTH<float, M1>(indices_cols - j);

				// load 64-bit indices for this row segment
				auto v_idx64 = VECTOR_LOAD<int64_t, M2>(&indices[i * indices_cols + j], vl);
				// narrow to 32-bit
				auto v_idx32 = VECTOR_NARROW_SRA<int32_t, M1>(v_idx64, 0, vl);

				// row strides in bytes: index * data_cols * sizeof(float)
				auto v_row_byte = VECTOR_MUL<int32_t, M1>(v_idx32, (int32_t)(data_cols * sizeof(float)), vl);

				// add column offsets: (j + vid) * sizeof(float)
				auto v_vid = VECTOR_VID<uint32_t, M1>(vl);
				auto v_col = VECTOR_ADD<uint32_t, M1>(v_vid, (uint32_t)j, vl);
				auto v_col_byte_u = VECTOR_MUL<uint32_t, M1>(v_col, (int32_t)sizeof(float), vl);
				auto v_col_byte = VECTOR_REINTERPRET<uint32_t, int32_t, M1>(v_col_byte_u);

				// final byte offsets relative to base data pointer
				auto v_off = VECTOR_ADD<int32_t, M1>(v_row_byte, v_col_byte, vl);
				auto v_off_u = VECTOR_REINTERPRET<int32_t, uint32_t, M1>(v_off);

				// indexed load and contiguous store to output row
				auto v_vals = VECTOR_INDEXED_LOAD<float, M1>(data, v_off_u, vl);
				VECTOR_STORE<float, M1>(&output[i * indices_cols + j], v_vals, vl);

				j += vl;
			}
		}
	} else { // axis == 1
		for (size_t i = 0; i < indices_rows; i++) {
			const float* row_data = &data[i * data_cols];
			size_t j = 0;
			while (j < indices_cols) {
				size_t vl = SET_VECTOR_LENGTH<float, M1>(indices_cols - j);

				auto v_idx64 = VECTOR_LOAD<int64_t, M2>(const_cast<int64_t*>(&indices[i * indices_cols + j]), vl);
				auto v_idx32 = VECTOR_NARROW_SRA<int32_t, M1>(v_idx64, 0, vl);
				
				// byte offsets within the row: idx * sizeof(float)
				auto v_byte = VECTOR_SLL<int32_t, M1>(v_idx32, 2, vl);
				auto v_off_u = VECTOR_REINTERPRET<int32_t, uint32_t, M1>(v_byte);
				
				auto v_vals = VECTOR_INDEXED_LOAD<float, M1>(row_data, v_off_u, vl);
				VECTOR_STORE<float, M1>(&output[i * indices_cols + j], v_vals, vl);

				j += vl;
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
				size_t vl = SET_VECTOR_LENGTH<float, M2>(indices_cols - j);
				auto v_idx64 = VECTOR_LOAD<int64_t, M4>(&indices[i * indices_cols + j], vl);
				auto v_idx32 = VECTOR_NARROW_SRA<int32_t, M2>(v_idx64, 0, vl);
				auto v_row_byte = VECTOR_MUL<int32_t, M2>(v_idx32, (int32_t)(data_cols * sizeof(float)), vl);
				auto v_vid = VECTOR_VID<uint32_t, M2>(vl);
				auto v_col = VECTOR_ADD<uint32_t, M2>(v_vid, (uint32_t)j, vl);
				auto v_col_byte_u = VECTOR_MUL<uint32_t, M2>(v_col, (int32_t)sizeof(float), vl);
				auto v_col_byte = VECTOR_REINTERPRET<uint32_t, int32_t, M2>(v_col_byte_u);
				auto v_off = VECTOR_ADD<int32_t, M2>(v_row_byte, v_col_byte, vl);
				auto v_off_u = VECTOR_REINTERPRET<int32_t, uint32_t, M2>(v_off);
				auto v_vals = VECTOR_INDEXED_LOAD<float, M2>(data, v_off_u, vl);
				VECTOR_STORE<float, M2>(&output[i * indices_cols + j], v_vals, vl);
				j += vl;
			}
		}
	} else {
		for (size_t i = 0; i < indices_rows; i++) {
			const float* row_data = &data[i * data_cols];
			size_t j = 0;
			while (j < indices_cols) {
				size_t vl = SET_VECTOR_LENGTH<float, M2>(indices_cols - j);
				auto v_idx64 = VECTOR_LOAD<int64_t, M4>(&indices[i * indices_cols + j], vl);
				auto v_idx32 = VECTOR_NARROW_SRA<int32_t, M2>(v_idx64, 0, vl);
				auto v_byte = VECTOR_SLL<int32_t, M2>(v_idx32, 2, vl);
				auto v_off_u = VECTOR_REINTERPRET<int32_t, uint32_t, M2>(v_byte);
				auto v_vals = VECTOR_INDEXED_LOAD<float, M2>(row_data, v_off_u, vl);
				VECTOR_STORE<float, M2>(&output[i * indices_cols + j], v_vals, vl);
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
				size_t vl = SET_VECTOR_LENGTH<float, M4>(indices_cols - j);
				auto v_idx64 = VECTOR_LOAD<int64_t, M8>(&indices[i * indices_cols + j], vl);
				auto v_idx32 = VECTOR_NARROW_SRA<int32_t, M4>(v_idx64, 0, vl);
				auto v_row_byte = VECTOR_MUL<int32_t, M4>(v_idx32, (int32_t)(data_cols * sizeof(float)), vl);
				auto v_vid = VECTOR_VID<uint32_t, M4>(vl);
				auto v_col = VECTOR_ADD<uint32_t, M4>(v_vid, (uint32_t)j, vl);
				auto v_col_byte_u = VECTOR_MUL<uint32_t, M4>(v_col, (int32_t)sizeof(float), vl);
				auto v_col_byte = VECTOR_REINTERPRET<uint32_t, int32_t, M4>(v_col_byte_u);
				auto v_off = VECTOR_ADD<int32_t, M4>(v_row_byte, v_col_byte, vl);
				auto v_off_u = VECTOR_REINTERPRET<int32_t, uint32_t, M4>(v_off);
				auto v_vals = VECTOR_INDEXED_LOAD<float, M4>(data, v_off_u, vl);
				VECTOR_STORE<float, M4>(&output[i * indices_cols + j], v_vals, vl);
				j += vl;
			}
		}
	} else {
		for (size_t i = 0; i < indices_rows; i++) {
			const float* row_data = &data[i * data_cols];
			size_t j = 0;
			while (j < indices_cols) {
				size_t vl = SET_VECTOR_LENGTH<float, M4>(indices_cols - j);
				auto v_idx64 = VECTOR_LOAD<int64_t, M8>(&indices[i * indices_cols + j], vl);
				auto v_idx32 = VECTOR_NARROW_SRA<int32_t, M4>(v_idx64, 0, vl);
				auto v_byte = VECTOR_SLL<int32_t, M4>(v_idx32, 2, vl);
				auto v_off_u = VECTOR_REINTERPRET<int32_t, uint32_t, M4>(v_byte);
				auto v_vals = VECTOR_INDEXED_LOAD<float, M4>(row_data, v_off_u, vl);
				VECTOR_STORE<float, M4>(&output[i * indices_cols + j], v_vals, vl);
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
				size_t vl = SET_VECTOR_LENGTH<float, M8>(indices_cols - j);
				size_t vl_half = vl / 2;
				if (vl_half == 0) vl_half = 1;
				auto v_idx64 = VECTOR_LOAD<int64_t, M8>(&indices[i * indices_cols + j], vl_half);
				auto v_idx32 = VECTOR_NARROW_SRA<int32_t, M4>(v_idx64, 0, vl_half);
				auto v_row_byte = VECTOR_MUL<int32_t, M4>(v_idx32, (int32_t)(data_cols * sizeof(float)), vl_half);
				auto v_vid = VECTOR_VID<uint32_t, M4>(vl_half);
				auto v_col = VECTOR_ADD<uint32_t, M4>(v_vid, (uint32_t)j, vl_half);
				auto v_col_byte_u = VECTOR_MUL<uint32_t, M4>(v_col, (int32_t)sizeof(float), vl_half);
				auto v_col_byte = VECTOR_REINTERPRET<uint32_t, int32_t, M4>(v_col_byte_u);
				auto v_off = VECTOR_ADD<int32_t, M4>(v_row_byte, v_col_byte, vl_half);
				auto v_off_u = VECTOR_REINTERPRET<int32_t, uint32_t, M4>(v_off);
				auto v_vals = VECTOR_INDEXED_LOAD<float, M4>(data, v_off_u, vl_half);
				VECTOR_STORE<float, M4>(&output[i * indices_cols + j], v_vals, vl_half);
				j += vl_half;
			}
		}
	} else {
		for (size_t i = 0; i < indices_rows; i++) {
			const float* row_data = &data[i * data_cols];
			size_t j = 0;
			while (j < indices_cols) {
				size_t vl = SET_VECTOR_LENGTH<float, M8>(indices_cols - j);
				size_t vl_half = vl / 2;
				if (vl_half == 0) vl_half = 1;
				auto v_idx64 = VECTOR_LOAD<int64_t, M8>(&indices[i * indices_cols + j], vl_half);
				auto v_idx32 = VECTOR_NARROW_SRA<int32_t, M4>(v_idx64, 0, vl_half);
				auto v_byte = VECTOR_SLL<int32_t, M4>(v_idx32, 2, vl_half);
				auto v_off_u = VECTOR_REINTERPRET<int32_t, uint32_t, M4>(v_byte);
				auto v_vals = VECTOR_INDEXED_LOAD<float, M4>(row_data, v_off_u, vl_half);
				VECTOR_STORE<float, M4>(&output[i * indices_cols + j], v_vals, vl_half);
				j += vl_half;
			}
		}
	}
}
/****************************** Tiled Scalar ******************************/
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

/****************************** Tiled Vectorized ******************************/

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
					size_t vl = SET_VECTOR_LENGTH<float, M1>(indices_cols - j);
					auto v_idx64 = VECTOR_LOAD<int64_t, M2>(&indices[i * indices_cols + j], vl);
					auto v_idx32 = VECTOR_NARROW_SRA<int32_t, M1>(v_idx64, 0, vl);
					auto v_row_byte = VECTOR_MUL<int32_t, M1>(v_idx32, (int32_t)(data_cols * sizeof(float)), vl);
					auto v_vid = VECTOR_VID<uint32_t, M1>(vl);
					auto v_col = VECTOR_ADD<uint32_t, M1>(v_vid, (uint32_t)j, vl);
					auto v_col_byte_u = VECTOR_MUL<uint32_t, M1>(v_col, (int32_t)sizeof(float), vl);
					auto v_col_byte = VECTOR_REINTERPRET<uint32_t, int32_t, M1>(v_col_byte_u);
					auto v_off = VECTOR_ADD<int32_t, M1>(v_row_byte, v_col_byte, vl);
					auto v_off_u = VECTOR_REINTERPRET<int32_t, uint32_t, M1>(v_off);
					auto v_vals = VECTOR_INDEXED_LOAD<float, M1>(data, v_off_u, vl);
					VECTOR_STORE<float, M1>(&output[i * indices_cols + j], v_vals, vl);
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
					size_t vl = SET_VECTOR_LENGTH<float, M1>(indices_cols - j);
					auto v_idx64 = VECTOR_LOAD<int64_t, M2>(&indices[i * indices_cols + j], vl);
					auto v_idx32 = VECTOR_NARROW_SRA<int32_t, M1>(v_idx64, 0, vl);
					auto v_byte = VECTOR_SLL<int32_t, M1>(v_idx32, 2, vl);
					auto v_off_u = VECTOR_REINTERPRET<int32_t, uint32_t, M1>(v_byte);
					auto v_vals = VECTOR_INDEXED_LOAD<float, M1>(row_data, v_off_u, vl);
					VECTOR_STORE<float, M1>(&output[i * indices_cols + j], v_vals, vl);
					j += vl;
				}
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
					size_t vl = SET_VECTOR_LENGTH<float, M2>(indices_cols - j);
					auto v_idx64 = VECTOR_LOAD<int64_t, M4>(&indices[i * indices_cols + j], vl);
					auto v_idx32 = VECTOR_NARROW_SRA<int32_t, M2>(v_idx64, 0, vl);
					auto v_row_byte = VECTOR_MUL<int32_t, M2>(v_idx32, (int32_t)(data_cols * sizeof(float)), vl);
					auto v_vid = VECTOR_VID<uint32_t, M2>(vl);
					auto v_col = VECTOR_ADD<uint32_t, M2>(v_vid, (uint32_t)j, vl);
					auto v_col_byte_u = VECTOR_MUL<uint32_t, M2>(v_col, (int32_t)sizeof(float), vl);
					auto v_col_byte = VECTOR_REINTERPRET<uint32_t, int32_t, M2>(v_col_byte_u);
					auto v_off = VECTOR_ADD<int32_t, M2>(v_row_byte, v_col_byte, vl);
					auto v_off_u = VECTOR_REINTERPRET<int32_t, uint32_t, M2>(v_off);
					auto v_vals = VECTOR_INDEXED_LOAD<float, M2>(data, v_off_u, vl);
					VECTOR_STORE<float, M2>(&output[i * indices_cols + j], v_vals, vl);
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
					size_t vl = SET_VECTOR_LENGTH<float, M2>(indices_cols - j);
					auto v_idx64 = VECTOR_LOAD<int64_t, M4>(&indices[i * indices_cols + j], vl);
					auto v_idx32 = VECTOR_NARROW_SRA<int32_t, M2>(v_idx64, 0, vl);
					auto v_byte = VECTOR_SLL<int32_t, M2>(v_idx32, 2, vl);
					auto v_off_u = VECTOR_REINTERPRET<int32_t, uint32_t, M2>(v_byte);
					auto v_vals = VECTOR_INDEXED_LOAD<float, M2>(row_data, v_off_u, vl);
					VECTOR_STORE<float, M2>(&output[i * indices_cols + j], v_vals, vl);
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
					size_t vl = SET_VECTOR_LENGTH<float, M4>(indices_cols - j);
					auto v_idx64 = VECTOR_LOAD<int64_t, M8>(&indices[i * indices_cols + j], vl);
					auto v_idx32 = VECTOR_NARROW_SRA<int32_t, M4>(v_idx64, 0, vl);
					auto v_row_byte = VECTOR_MUL<int32_t, M4>(v_idx32, (int32_t)(data_cols * sizeof(float)), vl);
					auto v_vid = VECTOR_VID<uint32_t, M4>(vl);
					auto v_col = VECTOR_ADD<uint32_t, M4>(v_vid, (uint32_t)j, vl);
					auto v_col_byte_u = VECTOR_MUL<uint32_t, M4>(v_col, (int32_t)sizeof(float), vl);
					auto v_col_byte = VECTOR_REINTERPRET<uint32_t, int32_t, M4>(v_col_byte_u);
					auto v_off = VECTOR_ADD<int32_t, M4>(v_row_byte, v_col_byte, vl);
					auto v_off_u = VECTOR_REINTERPRET<int32_t, uint32_t, M4>(v_off);
					auto v_vals = VECTOR_INDEXED_LOAD<float, M4>(data, v_off_u, vl);
					VECTOR_STORE<float, M4>(&output[i * indices_cols + j], v_vals, vl);
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
					size_t vl = SET_VECTOR_LENGTH<float, M4>(indices_cols - j);
					auto v_idx64 = VECTOR_LOAD<int64_t, M8>(&indices[i * indices_cols + j], vl);
					auto v_idx32 = VECTOR_NARROW_SRA<int32_t, M4>(v_idx64, 0, vl);
					auto v_byte = VECTOR_SLL<int32_t, M4>(v_idx32, 2, vl);
					auto v_off_u = VECTOR_REINTERPRET<int32_t, uint32_t, M4>(v_byte);
					auto v_vals = VECTOR_INDEXED_LOAD<float, M4>(row_data, v_off_u, vl);
					VECTOR_STORE<float, M4>(&output[i * indices_cols + j], v_vals, vl);
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
					size_t vl = SET_VECTOR_LENGTH<float, M8>(indices_cols - j);
					size_t vl_half = vl / 2;
					if (vl_half == 0) vl_half = 1;
					auto v_idx64 = VECTOR_LOAD<int64_t, M8>(&indices[i * indices_cols + j], vl_half);
					auto v_idx32 = VECTOR_NARROW_SRA<int32_t, M4>(v_idx64, 0, vl_half);
					auto v_row_byte = VECTOR_MUL<int32_t, M4>(v_idx32, (int32_t)(data_cols * sizeof(float)), vl_half);
					auto v_vid = VECTOR_VID<uint32_t, M4>(vl_half);
					auto v_col = VECTOR_ADD<uint32_t, M4>(v_vid, (uint32_t)j, vl_half);
					auto v_col_byte_u = VECTOR_MUL<uint32_t, M4>(v_col, (int32_t)sizeof(float), vl_half);
					auto v_col_byte = VECTOR_REINTERPRET<uint32_t, int32_t, M4>(v_col_byte_u);
					auto v_off = VECTOR_ADD<int32_t, M4>(v_row_byte, v_col_byte, vl_half);
					auto v_off_u = VECTOR_REINTERPRET<int32_t, uint32_t, M4>(v_off);
					auto v_vals = VECTOR_INDEXED_LOAD<float, M4>(data, v_off_u, vl_half);
					VECTOR_STORE<float, M4>(&output[i * indices_cols + j], v_vals, vl_half);
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
					size_t vl = SET_VECTOR_LENGTH<float, M8>(indices_cols - j);
					size_t vl_half = vl / 2;
					if (vl_half == 0) vl_half = 1;
					auto v_idx64 = VECTOR_LOAD<int64_t, M8>(&indices[i * indices_cols + j], vl_half);
					auto v_idx32 = VECTOR_NARROW_SRA<int32_t, M4>(v_idx64, 0, vl_half);
					auto v_byte = VECTOR_SLL<int32_t, M4>(v_idx32, 2, vl_half);
					auto v_off_u = VECTOR_REINTERPRET<int32_t, uint32_t, M4>(v_byte);
					auto v_vals = VECTOR_INDEXED_LOAD<float, M4>(row_data, v_off_u, vl_half);
					VECTOR_STORE<float, M4>(&output[i * indices_cols + j], v_vals, vl_half);
					j += vl_half;
				}
			}
		}
	}
}
