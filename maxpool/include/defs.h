#ifndef DEFS_H
#define DEFS_H

#include <cstddef>
#include <cstdint>
#include <algorithm> // Needed for std::min

// --- Tiling Configuration ---
#define TILE_H 32
#define TILE_W 32

// Calculate output dimension helper macro
#define CALC_OUT_DIM(in_dim, kernel, stride, ceil_mode) \
    (ceil_mode ? ((in_dim) + (stride) - (kernel) + (stride) - 1) / (stride) : ((in_dim) - (kernel)) / (stride) + 1)

// --- Low-Level Kernel Prototypes (Internal - Process one tile) ---
void maxpool_scalar_tile(
    const float* X, float* Y, int64_t* I,
    size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode,
    size_t OH, size_t OW,
    size_t tile_oh_start, size_t tile_ow_start,
    size_t tile_oh_end, size_t tile_ow_end);

void maxpool_e32m1_tile(
    const float* X, float* Y, int64_t* I,
    size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode,
    size_t OH, size_t OW,
    size_t tile_oh_start, size_t tile_ow_start,
    size_t tile_oh_end, size_t tile_ow_end);

void maxpool_e32m2_tile(
    const float* X, float* Y, int64_t* I,
    size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode,
    size_t OH, size_t OW,
    size_t tile_oh_start, size_t tile_ow_start,
    size_t tile_oh_end, size_t tile_ow_end);

void maxpool_e32m4_tile(
    const float* X, float* Y, int64_t* I,
    size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode,
    size_t OH, size_t OW,
    size_t tile_oh_start, size_t tile_ow_start,
    size_t tile_oh_end, size_t tile_ow_end);

void maxpool_e32m8_tile(
    const float* X, float* Y, int64_t* I,
    size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode,
    size_t OH, size_t OW,
    size_t tile_oh_start, size_t tile_ow_start,
    size_t tile_oh_end, size_t tile_ow_end);

// --- High-Level Tiled Function Prototypes (External API) ---
void maxpool_scalar_tiled(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode);
void maxpool_e32m1_tiled(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode);
void maxpool_e32m2_tiled(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode);
void maxpool_e32m4_tiled(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode);
void maxpool_e32m8_tiled(const float* X, float* Y, int64_t* I, size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode);

// Utility function prototypes
void read_tensor_binary(const char* filename, float* tensor, size_t count);
void write_tensor_binary_float(const char* filename, const float* tensor, size_t count);
void write_tensor_binary_int64(const char* filename, const int64_t* tensor, size_t count);

#endif // DEFS_H
