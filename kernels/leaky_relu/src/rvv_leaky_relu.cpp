#include <cstddef>
#include <riscv_vector.h>
#include "rvv_defs.hpp"

using namespace std;

/*********************************** Scalar ************************************/

void leaky_relu_scalar(const float* src, float* dest, size_t n, float alpha) {
    for (size_t i = 0; i < n; i++) {
        float val = src[i];
        // Standard Leaky ReLU logic: 
        // if val < 0, return val * alpha; else return val.
        dest[i] = (val < 0.0f) ? (val * alpha) : val;
    }
}

/*********************************** Vectorized ************************************/

void leaky_relu_e32m1(const float* src, float* dest, size_t n, float alpha) {
    size_t vl = SET_VECTOR_LENGTH<float, M1>(n);
    while (n >= vl * 8) {
        auto v0 = VECTOR_LOAD<float, M1>(src + vl*0, vl);
        auto v1 = VECTOR_LOAD<float, M1>(src + vl*1, vl);
        auto v2 = VECTOR_LOAD<float, M1>(src + vl*2, vl);
        auto v3 = VECTOR_LOAD<float, M1>(src + vl*3, vl);
        auto v4 = VECTOR_LOAD<float, M1>(src + vl*4, vl);
        auto v5 = VECTOR_LOAD<float, M1>(src + vl*5, vl);
        auto v6 = VECTOR_LOAD<float, M1>(src + vl*6, vl);
        auto v7 = VECTOR_LOAD<float, M1>(src + vl*7, vl);

        VECTOR_STORE<float, M1>(dest + vl*0, __riscv_vfmul_vf_f32m1_m(__riscv_vmflt_vf_f32m1_b32(v0, 0.0f, vl), v0, alpha, vl), vl);
        VECTOR_STORE<float, M1>(dest + vl*1, __riscv_vfmul_vf_f32m1_m(__riscv_vmflt_vf_f32m1_b32(v1, 0.0f, vl), v1, alpha, vl), vl);
        VECTOR_STORE<float, M1>(dest + vl*2, __riscv_vfmul_vf_f32m1_m(__riscv_vmflt_vf_f32m1_b32(v2, 0.0f, vl), v2, alpha, vl), vl);
        VECTOR_STORE<float, M1>(dest + vl*3, __riscv_vfmul_vf_f32m1_m(__riscv_vmflt_vf_f32m1_b32(v3, 0.0f, vl), v3, alpha, vl), vl);
        VECTOR_STORE<float, M1>(dest + vl*4, __riscv_vfmul_vf_f32m1_m(__riscv_vmflt_vf_f32m1_b32(v4, 0.0f, vl), v4, alpha, vl), vl);
        VECTOR_STORE<float, M1>(dest + vl*5, __riscv_vfmul_vf_f32m1_m(__riscv_vmflt_vf_f32m1_b32(v5, 0.0f, vl), v5, alpha, vl), vl);
        VECTOR_STORE<float, M1>(dest + vl*6, __riscv_vfmul_vf_f32m1_m(__riscv_vmflt_vf_f32m1_b32(v6, 0.0f, vl), v6, alpha, vl), vl);
        VECTOR_STORE<float, M1>(dest + vl*7, __riscv_vfmul_vf_f32m1_m(__riscv_vmflt_vf_f32m1_b32(v7, 0.0f, vl), v7, alpha, vl), vl);

        src += vl * 8; dest += vl * 8; n -= vl * 8;
    }
    while (n > 0) {
        vl = SET_VECTOR_LENGTH<float, M1>(n);
        auto v = VECTOR_LOAD<float, M1>(src, vl);
        VECTOR_STORE<float, M1>(dest, __riscv_vfmul_vf_f32m1_m(__riscv_vmflt_vf_f32m1_b32(v, 0.0f, vl), v, alpha, vl), vl);
        src += vl; dest += vl; n -= vl;
    }
}

void leaky_relu_e32m2(const float* src, float* dest, size_t n, float alpha) {
	size_t vl = SET_VECTOR_LENGTH<float, M2>(n);
	while (n >= vl * 4) {
		auto v0 = VECTOR_LOAD<float, M2>(src + vl*0, vl);
		auto v1 = VECTOR_LOAD<float, M2>(src + vl*1, vl);
		auto v2 = VECTOR_LOAD<float, M2>(src + vl*2, vl);
		auto v3 = VECTOR_LOAD<float, M2>(src + vl*3, vl);

		VECTOR_STORE<float, M2>(dest + vl*0, __riscv_vfmul_vf_f32m2_m(__riscv_vmflt_vf_f32m2_b16(v0, 0.0f, vl), v0, alpha, vl), vl);
		VECTOR_STORE<float, M2>(dest + vl*1, __riscv_vfmul_vf_f32m2_m(__riscv_vmflt_vf_f32m2_b16(v1, 0.0f, vl), v1, alpha, vl), vl);
		VECTOR_STORE<float, M2>(dest + vl*2, __riscv_vfmul_vf_f32m2_m(__riscv_vmflt_vf_f32m2_b16(v2, 0.0f, vl), v2, alpha, vl), vl);
		VECTOR_STORE<float, M2>(dest + vl*3, __riscv_vfmul_vf_f32m2_m(__riscv_vmflt_vf_f32m2_b16(v3, 0.0f, vl), v3, alpha, vl), vl);

		src += vl * 4; dest += vl * 4; n -= vl * 4;
	}
	while (n > 0) {
		vl = SET_VECTOR_LENGTH<float, M2>(n);
		auto v = VECTOR_LOAD<float, M2>(src, vl);
		VECTOR_STORE<float, M2>(dest, __riscv_vfmul_vf_f32m2_m(__riscv_vmflt_vf_f32m2_b16(v, 0.0f, vl), v, alpha, vl), vl);
		src += vl; dest += vl; n -= vl;
	}
}

void leaky_relu_e32m4(const float* src, float* dest, size_t n, float alpha) {
	size_t vl = SET_VECTOR_LENGTH<float, M4>(n);
	while (n >= vl * 2) {
		auto v0 = VECTOR_LOAD<float, M4>(src + vl*0, vl);
		auto v1 = VECTOR_LOAD<float, M4>(src + vl*1, vl);

		VECTOR_STORE<float, M4>(dest + vl*0, __riscv_vfmul_vf_f32m4_m(__riscv_vmflt_vf_f32m4_b8(v0, 0.0f, vl), v0, alpha, vl), vl);
		VECTOR_STORE<float, M4>(dest + vl*1, __riscv_vfmul_vf_f32m4_m(__riscv_vmflt_vf_f32m4_b8(v1, 0.0f, vl), v1, alpha, vl), vl);

		src += vl * 2; dest += vl * 2; n -= vl * 2;
	}
	while (n > 0) {
		vl = SET_VECTOR_LENGTH<float, M4>(n);
		auto v = VECTOR_LOAD<float, M4>(src, vl);
		VECTOR_STORE<float, M4>(dest, __riscv_vfmul_vf_f32m4_m(__riscv_vmflt_vf_f32m4_b8(v, 0.0f, vl), v, alpha, vl), vl);
		src += vl; dest += vl; n -= vl;
	}
}

void leaky_relu_e32m8(const float* src, float* dest, size_t n, float alpha) {
	size_t vl = SET_VECTOR_LENGTH<float, M8>(n);
	while (n >= vl * 2) {
		auto v0 = VECTOR_LOAD<float, M8>(src + vl*0, vl);
		auto v1 = VECTOR_LOAD<float, M8>(src + vl*1, vl);

		VECTOR_STORE<float, M8>(dest + vl*0, __riscv_vfmul_vf_f32m8_m(__riscv_vmflt_vf_f32m8_b4(v0, 0.0f, vl), v0, alpha, vl), vl);
		VECTOR_STORE<float, M8>(dest + vl*1, __riscv_vfmul_vf_f32m8_m(__riscv_vmflt_vf_f32m8_b4(v1, 0.0f, vl), v1, alpha, vl), vl);

		src += vl * 2; dest += vl * 2; n -= vl * 2;
	}
	while (n > 0) {
		vl = SET_VECTOR_LENGTH<float, M8>(n);
		auto v = VECTOR_LOAD<float, M8>(src, vl);
		VECTOR_STORE<float, M8>(dest, __riscv_vfmul_vf_f32m8_m(__riscv_vmflt_vf_f32m8_b4(v, 0.0f, vl), v, alpha, vl), vl);
		src += vl; dest += vl; n -= vl;
	}
}

/*********************************** Tiled Scalar ************************************/

void leaky_relu_tiled_scalar(const float* input, float* output, size_t size, float alpha, size_t TILE_SIZE) {
    for (size_t start = 0; start < size; start += TILE_SIZE) {
        // Calculate the actual size of the current tile 
        // (Handles the final partial tile if size is not divisible by TILE_SIZE)
        size_t current_tile_size = (start + TILE_SIZE < size) ? TILE_SIZE : (size - start);
        
        // Call the scalar kernel for this specific memory chunk
        leaky_relu_scalar(input + start, output + start, current_tile_size, alpha);
    }
}

/*********************************** Tiled Vectorized ************************************/

void leaky_relu_tiled_e32m1(const float* input, float* output, size_t size, float alpha, size_t TILE_SIZE) {
    for (size_t i = 0; i < size; i += TILE_SIZE) {
        size_t current_tile_size = (size - i < TILE_SIZE) ? (size - i) : TILE_SIZE;
        leaky_relu_e32m1(input + i, output + i, current_tile_size, alpha);
    }
}

void leaky_relu_tiled_e32m2(const float* input, float* output, size_t size, float alpha, size_t TILE_SIZE) {
    for (size_t i = 0; i < size; i += TILE_SIZE) {
        size_t current_tile_size = (size - i < TILE_SIZE) ? (size - i) : TILE_SIZE;
        leaky_relu_e32m2(input + i, output + i, current_tile_size, alpha);
    }
}

void leaky_relu_tiled_e32m4(const float* input, float* output, size_t size, float alpha, size_t TILE_SIZE) {
    for (size_t i = 0; i < size; i += TILE_SIZE) {
        size_t current_tile_size = (size - i < TILE_SIZE) ? (size - i) : TILE_SIZE;
        leaky_relu_e32m4(input + i, output + i, current_tile_size, alpha);
    }
}

void leaky_relu_tiled_e32m8(const float* input, float* output, size_t size, float alpha, size_t TILE_SIZE) {
    for (size_t i = 0; i < size; i += TILE_SIZE) {
        size_t current_tile_size = (size - i < TILE_SIZE) ? (size - i) : TILE_SIZE;
        leaky_relu_e32m8(input + i, output + i, current_tile_size, alpha);
    }
}