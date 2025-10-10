#ifndef DEFS_H
#define DEFS_H

#include <cstddef>
#include <cstdint>

static inline uint32_t read_mcycle(void) {
    uint32_t cycles;
    asm volatile("csrr %0, mcycle" : "=r"(cycles));
    return cycles;
}

void matmul_e32m1(int32_t *A, int32_t *B, int32_t *C, std::size_t M, std::size_t N, std::size_t K);
void matmul_e32m2(int32_t *A, int32_t *B, int32_t *C, std::size_t M, std::size_t N, std::size_t K);
void matmul_e32m4(int32_t *A, int32_t *B, int32_t *C, std::size_t M, std::size_t N, std::size_t K);
void matmul_e32m8(int32_t *A, int32_t *B, int32_t *C, std::size_t M, std::size_t N, std::size_t K);

void matmul_scalar(int32_t *A, int32_t *B, int32_t *C, std::size_t M, std::size_t N, std::size_t K);

#endif