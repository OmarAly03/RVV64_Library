#ifndef RVV_BITWISE_HPP
#define RVV_BITWISE_HPP

#include <cstddef> 
#include <riscv_vector.h>
#include <type_traits>

// Vector-Scalar Bitwise AND
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_AND_VX(VecType op1, T op2, size_t vl) {
    if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vand_vx_u8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vand_vx_u8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vand_vx_u8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vand_vx_u8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vand_vx_u8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vand_vx_u8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vand_vx_u8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vand_vx_u16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vand_vx_u16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vand_vx_u16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vand_vx_u16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vand_vx_u16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vand_vx_u16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vand_vx_u32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vand_vx_u32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vand_vx_u32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vand_vx_u32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vand_vx_u32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vand_vx_u64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vand_vx_u64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vand_vx_u64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vand_vx_u64m8(op1, op2, vl);
    }
}

// Vector-Scalar Logical Left Shift
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_SLL_VX(VecType op1, size_t shift, size_t vl) {
    if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vsll_vx_u8mf8(op1, shift, vl);
        else if constexpr (LMUL == MF4) return __riscv_vsll_vx_u8mf4(op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsll_vx_u8mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vx_u8m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vx_u8m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vx_u8m4(op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vx_u8m8(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vsll_vx_u16mf4(op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsll_vx_u16mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vx_u16m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vx_u16m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vx_u16m4(op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vx_u16m8(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vsll_vx_u32mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vx_u32m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vx_u32m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vx_u32m4(op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vx_u32m8(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vsll_vx_u64m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vx_u64m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vx_u64m4(op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vx_u64m8(op1, shift, vl);
    }
}

// Vector-Scalar Logical Right Shift
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_SRL_VX(VecType op1, size_t shift, size_t vl) {
    if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vsrl_vx_u8mf8(op1, shift, vl);
        else if constexpr (LMUL == MF4) return __riscv_vsrl_vx_u8mf4(op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsrl_vx_u8mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsrl_vx_u8m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsrl_vx_u8m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsrl_vx_u8m4(op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsrl_vx_u8m8(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vsrl_vx_u16mf4(op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsrl_vx_u16mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsrl_vx_u16m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsrl_vx_u16m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsrl_vx_u16m4(op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsrl_vx_u16m8(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vsrl_vx_u32mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsrl_vx_u32m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsrl_vx_u32m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsrl_vx_u32m4(op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsrl_vx_u32m8(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vsrl_vx_u64m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsrl_vx_u64m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsrl_vx_u64m4(op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsrl_vx_u64m8(op1, shift, vl);
    }
}

#endif // RVV_BITWISE_HPP
