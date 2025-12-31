#include "../include/defs.h"
#include "rvv_defs.hpp"
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
				size_t vl = SET_VECTOR_LENGTH<float, M1>(indices_cols - j);
				
				// Load indices
				auto v_indices_64 = VECTOR_LOAD<int64_t, M2>(&indices[i * indices_cols + j], vl);
				
				// Convert 64-bit indices to 32-bit and scale by stride
				auto v_indices_32 = VECTOR_NARROW_SRA<int32_t, M1>(v_indices_64, 0, vl);
				auto v_byte_offsets = VECTOR_MUL<int32_t, M1>(v_indices_32, data_cols * sizeof(float), vl);
				
				// Add column offset
				auto v_vid = VECTOR_VID<uint32_t, M1>(vl);
				auto v_col_idx = VECTOR_ADD<uint32_t, M1>(v_vid, j, vl);
				auto v_col_offsets_u = VECTOR_MUL<uint32_t, M1>(v_col_idx, sizeof(float), vl);
				auto v_col_offsets = VECTOR_REINTERPRET<uint32_t, int32_t, M1>(v_col_offsets_u);
				auto v_final_offsets = VECTOR_ADD<int32_t, M1>(v_byte_offsets, v_col_offsets, vl);
				
				// Indexed load
				auto v_offsets_u = VECTOR_REINTERPRET<int32_t, uint32_t, M1>(v_final_offsets);
				auto v_gathered = VECTOR_INDEXED_LOAD<float, M1>(data, v_offsets_u, vl);
				
				// Store result
				VECTOR_STORE<float, M1>(&output[i * indices_cols + j], v_gathered, vl);
				
				j += vl;
			}
		}
	} else if (axis == 1) {
		// Gather along columns
		for (size_t i = 0; i < indices_rows; i++) {
			const float* row_data = &data[i * data_cols];
			size_t j = 0;
			while (j < indices_cols) {
				size_t vl = SET_VECTOR_LENGTH<float, M1>(indices_cols - j);
				
				// Load indices
				auto v_indices_64 = VECTOR_LOAD<int64_t, M2>(&indices[i * indices_cols + j], vl);
				
				// Convert to byte offsets
				auto v_indices_32 = VECTOR_NARROW_SRA<int32_t, M1>(v_indices_64, 0, vl);
				auto v_byte_offsets = VECTOR_MUL<int32_t, M1>(v_indices_32, sizeof(float), vl);
				auto v_offsets_u = VECTOR_REINTERPRET<int32_t, uint32_t, M1>(v_byte_offsets);
				
				// Indexed load
				auto v_gathered = VECTOR_INDEXED_LOAD<float, M1>(row_data, v_offsets_u, vl);
				
				// Store result
				VECTOR_STORE<float, M1>(&output[i * indices_cols + j], v_gathered, vl);
				
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
				size_t vl = SET_VECTOR_LENGTH<float, M2>(indices_cols - j);
				
				auto v_indices_64 = VECTOR_LOAD<int64_t, M4>(&indices[i * indices_cols + j], vl);
				auto v_indices_32 = VECTOR_NARROW_SRA<int32_t, M2>(v_indices_64, 0, vl);
				auto v_byte_offsets = VECTOR_MUL<int32_t, M2>(v_indices_32, data_cols * sizeof(float), vl);
				
				auto v_vid = VECTOR_VID<uint32_t, M2>(vl);
				auto v_col_idx = VECTOR_ADD<uint32_t, M2>(v_vid, j, vl);
				auto v_col_offsets_u = VECTOR_MUL<uint32_t, M2>(v_col_idx, sizeof(float), vl);
				auto v_col_offsets = VECTOR_REINTERPRET<uint32_t, int32_t, M2>(v_col_offsets_u);
				auto v_final_offsets = VECTOR_ADD<int32_t, M2>(v_byte_offsets, v_col_offsets, vl);
				
				auto v_offsets_u = VECTOR_REINTERPRET<int32_t, uint32_t, M2>(v_final_offsets);
				auto v_gathered = VECTOR_INDEXED_LOAD<float, M2>(data, v_offsets_u, vl);
				VECTOR_STORE<float, M2>(&output[i * indices_cols + j], v_gathered, vl);
				
				j += vl;
			}
		}
	} else if (axis == 1) {
		for (size_t i = 0; i < indices_rows; i++) {
			const float* row_data = &data[i * data_cols];
			size_t j = 0;
			while (j < indices_cols) {
				size_t vl = SET_VECTOR_LENGTH<float, M2>(indices_cols - j);
				
				auto v_indices_64 = VECTOR_LOAD<int64_t, M4>(&indices[i * indices_cols + j], vl);
				auto v_indices_32 = VECTOR_NARROW_SRA<int32_t, M2>(v_indices_64, 0, vl);
				auto v_byte_offsets = VECTOR_MUL<int32_t, M2>(v_indices_32, sizeof(float), vl);
				auto v_offsets_u = VECTOR_REINTERPRET<int32_t, uint32_t, M2>(v_byte_offsets);
				
				auto v_gathered = VECTOR_INDEXED_LOAD<float, M2>(row_data, v_offsets_u, vl);
				VECTOR_STORE<float, M2>(&output[i * indices_cols + j], v_gathered, vl);
				
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
				size_t vl = SET_VECTOR_LENGTH<float, M4>(indices_cols - j);
				
				auto v_indices_64 = VECTOR_LOAD<int64_t, M8>(&indices[i * indices_cols + j], vl);
				auto v_indices_32 = VECTOR_NARROW_SRA<int32_t, M4>(v_indices_64, 0, vl);
				auto v_byte_offsets = VECTOR_MUL<int32_t, M4>(v_indices_32, data_cols * sizeof(float), vl);
				
				auto v_vid = VECTOR_VID<uint32_t, M4>(vl);
				auto v_col_idx = VECTOR_ADD<uint32_t, M4>(v_vid, j, vl);
				auto v_col_offsets_u = VECTOR_MUL<uint32_t, M4>(v_col_idx, sizeof(float), vl);
				auto v_col_offsets = VECTOR_REINTERPRET<uint32_t, int32_t, M4>(v_col_offsets_u);
				auto v_final_offsets = VECTOR_ADD<int32_t, M4>(v_byte_offsets, v_col_offsets, vl);
				
				auto v_offsets_u = VECTOR_REINTERPRET<int32_t, uint32_t, M4>(v_final_offsets);
				auto v_gathered = VECTOR_INDEXED_LOAD<float, M4>(data, v_offsets_u, vl);
				VECTOR_STORE<float, M4>(&output[i * indices_cols + j], v_gathered, vl);
				
				j += vl;
			}
		}
	} else if (axis == 1) {
		for (size_t i = 0; i < indices_rows; i++) {
			const float* row_data = &data[i * data_cols];
			size_t j = 0;
			while (j < indices_cols) {
				size_t vl = SET_VECTOR_LENGTH<float, M4>(indices_cols - j);
				
				auto v_indices_64 = VECTOR_LOAD<int64_t, M8>(&indices[i * indices_cols + j], vl);
				auto v_indices_32 = VECTOR_NARROW_SRA<int32_t, M4>(v_indices_64, 0, vl);
				auto v_byte_offsets = VECTOR_MUL<int32_t, M4>(v_indices_32, sizeof(float), vl);
				auto v_offsets_u = VECTOR_REINTERPRET<int32_t, uint32_t, M4>(v_byte_offsets);
				
				auto v_gathered = VECTOR_INDEXED_LOAD<float, M4>(row_data, v_offsets_u, vl);
				VECTOR_STORE<float, M4>(&output[i * indices_cols + j], v_gathered, vl);
				
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
				// For e32m8, we need to process indices in chunks due to LMUL limits
				size_t vl_idx = SET_VECTOR_LENGTH<int64_t, M8>(indices_cols - j);
				
				auto v_indices_64 = VECTOR_LOAD<int64_t, M8>(&indices[i * indices_cols + j], vl_idx);
				auto v_indices_32 = VECTOR_NARROW_SRA<int32_t, M4>(v_indices_64, 0, vl_idx);
				auto v_byte_offsets = VECTOR_MUL<int32_t, M4>(v_indices_32, data_cols * sizeof(float), vl_idx);
				
				auto v_vid = VECTOR_VID<uint32_t, M4>(vl_idx);
				auto v_col_idx = VECTOR_ADD<uint32_t, M4>(v_vid, j, vl_idx);
				auto v_col_offsets_u = VECTOR_MUL<uint32_t, M4>(v_col_idx, sizeof(float), vl_idx);
				auto v_col_offsets = VECTOR_REINTERPRET<uint32_t, int32_t, M4>(v_col_offsets_u);
				auto v_final_offsets = VECTOR_ADD<int32_t, M4>(v_byte_offsets, v_col_offsets, vl_idx);
				
				auto v_offsets_u = VECTOR_REINTERPRET<int32_t, uint32_t, M4>(v_final_offsets);
				auto v_gathered = VECTOR_INDEXED_LOAD<float, M4>(data, v_offsets_u, vl_idx);
				VECTOR_STORE<float, M4>(&output[i * indices_cols + j], v_gathered, vl_idx);
				
				j += vl_idx;
			}
		}
	} else if (axis == 1) {
		for (size_t i = 0; i < indices_rows; i++) {
			const float* row_data = &data[i * data_cols];
			size_t j = 0;
			while (j < indices_cols) {
				size_t vl_idx = SET_VECTOR_LENGTH<int64_t, M8>(indices_cols - j);
				
				auto v_indices_64 = VECTOR_LOAD<int64_t, M8>(&indices[i * indices_cols + j], vl_idx);
				auto v_indices_32 = VECTOR_NARROW_SRA<int32_t, M4>(v_indices_64, 0, vl_idx);
				auto v_byte_offsets = VECTOR_MUL<int32_t, M4>(v_indices_32, sizeof(float), vl_idx);
				auto v_offsets_u = VECTOR_REINTERPRET<int32_t, uint32_t, M4>(v_byte_offsets);
				
				auto v_gathered = VECTOR_INDEXED_LOAD<float, M4>(row_data, v_offsets_u, vl_idx);
				VECTOR_STORE<float, M4>(&output[i * indices_cols + j], v_gathered, vl_idx);
				
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
					size_t vl = SET_VECTOR_LENGTH<float, M1>(indices_cols - j);
					
					auto v_indices_64 = VECTOR_LOAD<int64_t, M2>(&indices[i * indices_cols + j], vl);
					auto v_indices_32 = VECTOR_NARROW_SRA<int32_t, M1>(v_indices_64, 0, vl);
					auto v_byte_offsets = VECTOR_MUL<int32_t, M1>(v_indices_32, data_cols * sizeof(float), vl);
					
					auto v_vid = VECTOR_VID<uint32_t, M1>(vl);
					auto v_col_idx = VECTOR_ADD<uint32_t, M1>(v_vid, j, vl);
					auto v_col_offsets_u = VECTOR_MUL<uint32_t, M1>(v_col_idx, sizeof(float), vl);
					auto v_col_offsets = VECTOR_REINTERPRET<uint32_t, int32_t, M1>(v_col_offsets_u);
					auto v_final_offsets = VECTOR_ADD<int32_t, M1>(v_byte_offsets, v_col_offsets, vl);
					
					auto v_offsets_u = VECTOR_REINTERPRET<int32_t, uint32_t, M1>(v_final_offsets);
					auto v_gathered = VECTOR_INDEXED_LOAD<float, M1>(data, v_offsets_u, vl);
					VECTOR_STORE<float, M1>(&output[i * indices_cols + j], v_gathered, vl);
					
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
					size_t vl = SET_VECTOR_LENGTH<float, M1>(indices_cols - j);
					
					auto v_indices_64 = VECTOR_LOAD<int64_t, M2>(&indices[i * indices_cols + j], vl);
					auto v_indices_32 = VECTOR_NARROW_SRA<int32_t, M1>(v_indices_64, 0, vl);
					auto v_byte_offsets = VECTOR_MUL<int32_t, M1>(v_indices_32, sizeof(float), vl);
					auto v_offsets_u = VECTOR_REINTERPRET<int32_t, uint32_t, M1>(v_byte_offsets);
					
					auto v_gathered = VECTOR_INDEXED_LOAD<float, M1>(row_data, v_offsets_u, vl);
					VECTOR_STORE<float, M1>(&output[i * indices_cols + j], v_gathered, vl);
					
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
					size_t vl = SET_VECTOR_LENGTH<float, M2>(indices_cols - j);
					
					auto v_indices_64 = VECTOR_LOAD<int64_t, M4>(&indices[i * indices_cols + j], vl);
					auto v_indices_32 = VECTOR_NARROW_SRA<int32_t, M2>(v_indices_64, 0, vl);
					auto v_byte_offsets = VECTOR_MUL<int32_t, M2>(v_indices_32, data_cols * sizeof(float), vl);
					
					auto v_vid = VECTOR_VID<uint32_t, M2>(vl);
					auto v_col_idx = VECTOR_ADD<uint32_t, M2>(v_vid, j, vl);
					auto v_col_offsets_u = VECTOR_MUL<uint32_t, M2>(v_col_idx, sizeof(float), vl);
					auto v_col_offsets = VECTOR_REINTERPRET<uint32_t, int32_t, M2>(v_col_offsets_u);
					auto v_final_offsets = VECTOR_ADD<int32_t, M2>(v_byte_offsets, v_col_offsets, vl);
					
					auto v_offsets_u = VECTOR_REINTERPRET<int32_t, uint32_t, M2>(v_final_offsets);
					auto v_gathered = VECTOR_INDEXED_LOAD<float, M2>(data, v_offsets_u, vl);
					VECTOR_STORE<float, M2>(&output[i * indices_cols + j], v_gathered, vl);
					
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
					size_t vl = SET_VECTOR_LENGTH<float, M2>(indices_cols - j);
					
					auto v_indices_64 = VECTOR_LOAD<int64_t, M4>(&indices[i * indices_cols + j], vl);
					auto v_indices_32 = VECTOR_NARROW_SRA<int32_t, M2>(v_indices_64, 0, vl);
					auto v_byte_offsets = VECTOR_MUL<int32_t, M2>(v_indices_32, sizeof(float), vl);
					auto v_offsets_u = VECTOR_REINTERPRET<int32_t, uint32_t, M2>(v_byte_offsets);
					
					auto v_gathered = VECTOR_INDEXED_LOAD<float, M2>(row_data, v_offsets_u, vl);
					VECTOR_STORE<float, M2>(&output[i * indices_cols + j], v_gathered, vl);
					
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
					size_t vl = SET_VECTOR_LENGTH<float, M4>(indices_cols - j);
					
					auto v_indices_64 = VECTOR_LOAD<int64_t, M8>(&indices[i * indices_cols + j], vl);
					auto v_indices_32 = VECTOR_NARROW_SRA<int32_t, M4>(v_indices_64, 0, vl);
					auto v_byte_offsets = VECTOR_MUL<int32_t, M4>(v_indices_32, data_cols * sizeof(float), vl);
					
					auto v_vid = VECTOR_VID<uint32_t, M4>(vl);
					auto v_col_idx = VECTOR_ADD<uint32_t, M4>(v_vid, j, vl);
					auto v_col_offsets_u = VECTOR_MUL<uint32_t, M4>(v_col_idx, sizeof(float), vl);
					auto v_col_offsets = VECTOR_REINTERPRET<uint32_t, int32_t, M4>(v_col_offsets_u);
					auto v_final_offsets = VECTOR_ADD<int32_t, M4>(v_byte_offsets, v_col_offsets, vl);
					
					auto v_offsets_u = VECTOR_REINTERPRET<int32_t, uint32_t, M4>(v_final_offsets);
					auto v_gathered = VECTOR_INDEXED_LOAD<float, M4>(data, v_offsets_u, vl);
					VECTOR_STORE<float, M4>(&output[i * indices_cols + j], v_gathered, vl);
					
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
					size_t vl = SET_VECTOR_LENGTH<float, M4>(indices_cols - j);
					
					auto v_indices_64 = VECTOR_LOAD<int64_t, M8>(&indices[i * indices_cols + j], vl);
					auto v_indices_32 = VECTOR_NARROW_SRA<int32_t, M4>(v_indices_64, 0, vl);
					auto v_byte_offsets = VECTOR_MUL<int32_t, M4>(v_indices_32, sizeof(float), vl);
					auto v_offsets_u = VECTOR_REINTERPRET<int32_t, uint32_t, M4>(v_byte_offsets);
					
					auto v_gathered = VECTOR_INDEXED_LOAD<float, M4>(row_data, v_offsets_u, vl);
					VECTOR_STORE<float, M4>(&output[i * indices_cols + j], v_gathered, vl);
					
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
					size_t vl_idx = SET_VECTOR_LENGTH<int64_t, M8>(indices_cols - j);
					
					auto v_indices_64 = VECTOR_LOAD<int64_t, M8>(&indices[i * indices_cols + j], vl_idx);
					auto v_indices_32 = VECTOR_NARROW_SRA<int32_t, M4>(v_indices_64, 0, vl_idx);
					auto v_byte_offsets = VECTOR_MUL<int32_t, M4>(v_indices_32, data_cols * sizeof(float), vl_idx);
					
					auto v_vid = VECTOR_VID<uint32_t, M4>(vl_idx);
					auto v_col_idx = VECTOR_ADD<uint32_t, M4>(v_vid, j, vl_idx);
					auto v_col_offsets_u = VECTOR_MUL<uint32_t, M4>(v_col_idx, sizeof(float), vl_idx);
					auto v_col_offsets = VECTOR_REINTERPRET<uint32_t, int32_t, M4>(v_col_offsets_u);
					auto v_final_offsets = VECTOR_ADD<int32_t, M4>(v_byte_offsets, v_col_offsets, vl_idx);
					
					auto v_offsets_u = VECTOR_REINTERPRET<int32_t, uint32_t, M4>(v_final_offsets);
					auto v_gathered = VECTOR_INDEXED_LOAD<float, M4>(data, v_offsets_u, vl_idx);
					VECTOR_STORE<float, M4>(&output[i * indices_cols + j], v_gathered, vl_idx);
					
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
					size_t vl_idx = SET_VECTOR_LENGTH<int64_t, M8>(indices_cols - j);
					
					auto v_indices_64 = VECTOR_LOAD<int64_t, M8>(&indices[i * indices_cols + j], vl_idx);
					auto v_indices_32 = VECTOR_NARROW_SRA<int32_t, M4>(v_indices_64, 0, vl_idx);
					auto v_byte_offsets = VECTOR_MUL<int32_t, M4>(v_indices_32, sizeof(float), vl_idx);
					auto v_offsets_u = VECTOR_REINTERPRET<int32_t, uint32_t, M4>(v_byte_offsets);
					
					auto v_gathered = VECTOR_INDEXED_LOAD<float, M4>(row_data, v_offsets_u, vl_idx);
					VECTOR_STORE<float, M4>(&output[i * indices_cols + j], v_gathered, vl_idx);
					
					j += vl_idx;
				}
			}
		}
	}
}
