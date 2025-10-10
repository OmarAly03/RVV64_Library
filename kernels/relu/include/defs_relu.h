#ifndef DEFS_H
#define DEFS_H

#include <cstddef>
#include <cstdint>

static inline uint32_t read_mcycle(void) {
    uint32_t cycles;
    asm volatile("csrr %0, mcycle" : "=r"(cycles));
    return cycles;
}

void relu_e32m1(int32_t* input, int32_t* output, std::size_t size);
void relu_e32m2(int32_t* input, int32_t* output, std::size_t size);
void relu_e32m4(int32_t* input, int32_t* output, std::size_t size);
void relu_e32m8(int32_t* input, int32_t* output, std::size_t size);

void relu_scalar(int32_t* input, int32_t* output, std::size_t size);

#endif