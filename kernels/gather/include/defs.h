#ifndef GATHER_DEFS_H
#define GATHER_DEFS_H

#include <cstddef>
#include <cstdint>

// Scalar implementation
void gather_scalar(
    const float* data,
    const int64_t* indices,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis
);

// Vectorized implementations
void gather_e32m1(
    const float* data,
    const int64_t* indices,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis
);

void gather_e32m2(
    const float* data,
    const int64_t* indices,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis
);

void gather_e32m4(
    const float* data,
    const int64_t* indices,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis
);

void gather_e32m8(
    const float* data,
    const int64_t* indices,
    float* output,
    size_t data_rows,
    size_t data_cols,
    size_t indices_rows,
    size_t indices_cols,
    int axis
);

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
);

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
);

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
);

// Tiled implementations
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
);

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
);

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
);

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
);

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
);

// Additional vector tile variants can be added as needed (m2/m4/m8)

#endif // GATHER_DEFS_H
