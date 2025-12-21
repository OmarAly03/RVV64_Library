#ifndef RVV_ARITHMETIC_HPP
#define RVV_ARITHMETIC_HPP

#include <cstddef> 
#include <riscv_vector.h>
#include <type_traits>

/*************************************************************************************************/

// Vector-Vector Addition Template
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_ADD_VV(VecType op1, VecType op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfadd_vv_f32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfadd_vv_f32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfadd_vv_f32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfadd_vv_f32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfadd_vv_f32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfadd_vv_f64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfadd_vv_f64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfadd_vv_f64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfadd_vv_f64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, _Float16>) {
        if constexpr (LMUL == MF4) return __riscv_vfadd_vv_f16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vfadd_vv_f16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfadd_vv_f16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfadd_vv_f16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfadd_vv_f16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfadd_vv_f16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vadd_vv_i8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vadd_vv_i8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vv_i8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vv_i8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_i8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_i8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_i8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vadd_vv_i16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vv_i16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vv_i16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_i16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_i16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_i16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vadd_vv_i32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vv_i32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_i32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_i32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_i32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vadd_vv_i64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_i64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_i64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_i64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vadd_vv_u8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vadd_vv_u8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vv_u8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vv_u8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_u8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_u8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_u8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vadd_vv_u16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vv_u16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vv_u16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_u16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_u16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_u16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vadd_vv_u32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vv_u32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_u32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_u32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_u32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vadd_vv_u64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_u64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_u64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_u64m8(op1, op2, vl);
    }
}

// Vector-Scalar Addition Template
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_ADD_VX(VecType op1, T op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfadd_vf_f32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfadd_vf_f32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfadd_vf_f32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfadd_vf_f32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfadd_vf_f32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfadd_vf_f64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfadd_vf_f64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfadd_vf_f64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfadd_vf_f64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, _Float16>) {
        if constexpr (LMUL == MF4) return __riscv_vfadd_vf_f16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vfadd_vf_f16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfadd_vf_f16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfadd_vf_f16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfadd_vf_f16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfadd_vf_f16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vadd_vx_i8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vadd_vx_i8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vx_i8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vx_i8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_i8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_i8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_i8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vadd_vx_i16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vx_i16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vx_i16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_i16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_i16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_i16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vadd_vx_i32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vx_i32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_i32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_i32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_i32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vadd_vx_i64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_i64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_i64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_i64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vadd_vx_u8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vadd_vx_u8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vx_u8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vx_u8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_u8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_u8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_u8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vadd_vx_u16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vx_u16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vx_u16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_u16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_u16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_u16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vadd_vx_u32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vx_u32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_u32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_u32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_u32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vadd_vx_u64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_u64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_u64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_u64m8(op1, op2, vl);
    }
}

// Unified VECTOR_ADD template that auto-detects vector vs scalar
template<typename T, int LMUL, typename VecType, typename Op2Type>
inline auto VECTOR_ADD(VecType op1, Op2Type op2, size_t vl) {
    // If op2 is a scalar type (arithmetic type), use vector-scalar addition
    if constexpr (std::is_arithmetic_v<Op2Type>) {
        return VECTOR_ADD_VX<T, LMUL>(op1, static_cast<T>(op2), vl);
    }
    // Otherwise, use vector-vector addition
    else {
        return VECTOR_ADD_VV<T, LMUL>(op1, op2, vl);
    }
}

/*************************************************************************************************/

// Vector-Vector Masked Addition Template
template<typename T, int LMUL, typename VecType, typename MaskType>
inline auto VECTOR_ADD_VV_M(MaskType mask, VecType op1, VecType op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfadd_vv_f32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfadd_vv_f32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfadd_vv_f32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfadd_vv_f32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfadd_vv_f32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfadd_vv_f64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfadd_vv_f64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfadd_vv_f64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfadd_vv_f64m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, _Float16>) {
        if constexpr (LMUL == MF4) return __riscv_vfadd_vv_f16mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vfadd_vv_f16mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfadd_vv_f16m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfadd_vv_f16m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfadd_vv_f16m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfadd_vv_f16m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vadd_vv_i8mf8_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vadd_vv_i8mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vv_i8mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vv_i8m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_i8m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_i8m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_i8m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vadd_vv_i16mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vv_i16mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vv_i16m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_i16m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_i16m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_i16m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vadd_vv_i32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vv_i32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_i32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_i32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_i32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vadd_vv_i64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_i64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_i64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_i64m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vadd_vv_u8mf8_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vadd_vv_u8mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vv_u8mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vv_u8m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_u8m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_u8m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_u8m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vadd_vv_u16mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vv_u16mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vv_u16m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_u16m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_u16m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_u16m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vadd_vv_u32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vv_u32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_u32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_u32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_u32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vadd_vv_u64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_u64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_u64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_u64m8_m(mask, op1, op2, vl);
    }
}

// Vector-Scalar Masked Addition Template
template<typename T, int LMUL, typename VecType, typename MaskType>
inline auto VECTOR_ADD_VX_M(MaskType mask, VecType op1, T op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfadd_vf_f32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfadd_vf_f32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfadd_vf_f32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfadd_vf_f32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfadd_vf_f32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfadd_vf_f64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfadd_vf_f64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfadd_vf_f64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfadd_vf_f64m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, _Float16>) {
        if constexpr (LMUL == MF4) return __riscv_vfadd_vf_f16mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vfadd_vf_f16mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfadd_vf_f16m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfadd_vf_f16m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfadd_vf_f16m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfadd_vf_f16m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vadd_vx_i8mf8_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vadd_vx_i8mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vx_i8mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vx_i8m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_i8m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_i8m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_i8m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vadd_vx_i16mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vx_i16mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vx_i16m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_i16m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_i16m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_i16m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vadd_vx_i32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vx_i32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_i32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_i32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_i32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vadd_vx_i64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_i64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_i64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_i64m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vadd_vx_u8mf8_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vadd_vx_u8mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vx_u8mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vx_u8m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_u8m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_u8m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_u8m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vadd_vx_u16mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vx_u16mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vx_u16m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_u16m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_u16m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_u16m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vadd_vx_u32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vx_u32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_u32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_u32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_u32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vadd_vx_u64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_u64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_u64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_u64m8_m(mask, op1, op2, vl);
    }
}

// Unified Masked VECTOR_ADD template that auto-detects vector vs scalar
template<typename T, int LMUL, typename MaskType, typename VecType, typename Op2Type>
inline auto VECTOR_ADD_MASKED(MaskType mask, VecType op1, Op2Type op2, size_t vl) {
    // If op2 is a scalar type (arithmetic type), use vector-scalar masked addition
    if constexpr (std::is_arithmetic_v<Op2Type>) {
        return VECTOR_ADD_VX_M<T, LMUL>(mask, op1, static_cast<T>(op2), vl);
    }
    // Otherwise, use vector-vector masked addition
    else {
        return VECTOR_ADD_VV_M<T, LMUL>(mask, op1, op2, vl);
    }
}

/*************************************************************************************************/
// Vector-Vector Subtraction Template
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_SUB_VV(VecType op1, VecType op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfsub_vv_f32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfsub_vv_f32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfsub_vv_f32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfsub_vv_f32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfsub_vv_f32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfsub_vv_f64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfsub_vv_f64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfsub_vv_f64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfsub_vv_f64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, _Float16>) {
        if constexpr (LMUL == MF4) return __riscv_vfsub_vv_f16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vfsub_vv_f16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfsub_vv_f16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfsub_vv_f16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfsub_vv_f16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfsub_vv_f16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vsub_vv_i8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vsub_vv_i8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsub_vv_i8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vv_i8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vv_i8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vv_i8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vv_i8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vsub_vv_i16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsub_vv_i16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vv_i16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vv_i16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vv_i16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vv_i16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vsub_vv_i32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vv_i32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vv_i32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vv_i32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vv_i32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vsub_vv_i64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vv_i64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vv_i64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vv_i64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vsub_vv_u8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vsub_vv_u8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsub_vv_u8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vv_u8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vv_u8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vv_u8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vv_u8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vsub_vv_u16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsub_vv_u16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vv_u16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vv_u16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vv_u16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vv_u16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vsub_vv_u32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vv_u32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vv_u32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vv_u32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vv_u32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vsub_vv_u64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vv_u64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vv_u64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vv_u64m8(op1, op2, vl);
    }
}

// Vector-Scalar Subtraction Template
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_SUB_VX(VecType op1, T op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfsub_vf_f32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfsub_vf_f32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfsub_vf_f32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfsub_vf_f32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfsub_vf_f32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfsub_vf_f64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfsub_vf_f64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfsub_vf_f64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfsub_vf_f64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, _Float16>) {
        if constexpr (LMUL == MF4) return __riscv_vfsub_vf_f16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vfsub_vf_f16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfsub_vf_f16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfsub_vf_f16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfsub_vf_f16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfsub_vf_f16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vsub_vx_i8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vsub_vx_i8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsub_vx_i8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vx_i8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vx_i8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vx_i8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vx_i8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vsub_vx_i16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsub_vx_i16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vx_i16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vx_i16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vx_i16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vx_i16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vsub_vx_i32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vx_i32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vx_i32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vx_i32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vx_i32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vsub_vx_i64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vx_i64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vx_i64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vx_i64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vsub_vx_u8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vsub_vx_u8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsub_vx_u8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vx_u8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vx_u8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vx_u8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vx_u8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vsub_vx_u16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsub_vx_u16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vx_u16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vx_u16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vx_u16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vx_u16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vsub_vx_u32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vx_u32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vx_u32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vx_u32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vx_u32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vsub_vx_u64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vx_u64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vx_u64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vx_u64m8(op1, op2, vl);
    }
}

// Updated Unified VECTOR_SUB template that auto-detects vector vs scalar
template<typename T, int LMUL, typename VecType, typename Op2Type>
inline auto VECTOR_SUB(VecType op1, Op2Type op2, size_t vl) {
    // If op2 is a scalar type (arithmetic type), use vector-scalar subtraction
    if constexpr (std::is_arithmetic_v<Op2Type>) {
        return VECTOR_SUB_VX<T, LMUL>(op1, static_cast<T>(op2), vl);
    }
    // Otherwise, use vector-vector subtraction
    else {
        return VECTOR_SUB_VV<T, LMUL>(op1, op2, vl);
    }
}

/*************************************************************************************************/

// Vector-Vector Masked Subtraction Template
template<typename T, int LMUL, typename VecType, typename MaskType>
inline auto VECTOR_SUB_VV_M(MaskType mask, VecType op1, VecType op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfsub_vv_f32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfsub_vv_f32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfsub_vv_f32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfsub_vv_f32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfsub_vv_f32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfsub_vv_f64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfsub_vv_f64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfsub_vv_f64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfsub_vv_f64m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, _Float16>) {
        if constexpr (LMUL == MF4) return __riscv_vfsub_vv_f16mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vfsub_vv_f16mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfsub_vv_f16m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfsub_vv_f16m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfsub_vv_f16m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfsub_vv_f16m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vsub_vv_i8mf8_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vsub_vv_i8mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsub_vv_i8mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vv_i8m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vv_i8m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vv_i8m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vv_i8m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vsub_vv_i16mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsub_vv_i16mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vv_i16m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vv_i16m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vv_i16m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vv_i16m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vsub_vv_i32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vv_i32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vv_i32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vv_i32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vv_i32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vsub_vv_i64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vv_i64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vv_i64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vv_i64m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vsub_vv_u8mf8_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vsub_vv_u8mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsub_vv_u8mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vv_u8m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vv_u8m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vv_u8m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vv_u8m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vsub_vv_u16mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsub_vv_u16mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vv_u16m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vv_u16m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vv_u16m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vv_u16m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vsub_vv_u32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vv_u32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vv_u32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vv_u32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vv_u32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vsub_vv_u64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vv_u64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vv_u64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vv_u64m8_m(mask, op1, op2, vl);
    }
}

// Vector-Scalar Masked Subtraction Template
template<typename T, int LMUL, typename VecType, typename MaskType>
inline auto VECTOR_SUB_VX_M(MaskType mask, VecType op1, T op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfsub_vf_f32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfsub_vf_f32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfsub_vf_f32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfsub_vf_f32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfsub_vf_f32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfsub_vf_f64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfsub_vf_f64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfsub_vf_f64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfsub_vf_f64m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, _Float16>) {
        if constexpr (LMUL == MF4) return __riscv_vfsub_vf_f16mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vfsub_vf_f16mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfsub_vf_f16m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfsub_vf_f16m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfsub_vf_f16m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfsub_vf_f16m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vsub_vx_i8mf8_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vsub_vx_i8mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsub_vx_i8mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vx_i8m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vx_i8m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vx_i8m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vx_i8m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vsub_vx_i16mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsub_vx_i16mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vx_i16m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vx_i16m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vx_i16m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vx_i16m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vsub_vx_i32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vx_i32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vx_i32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vx_i32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vx_i32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vsub_vx_i64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vx_i64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vx_i64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vx_i64m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vsub_vx_u8mf8_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vsub_vx_u8mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsub_vx_u8mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vx_u8m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vx_u8m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vx_u8m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vx_u8m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vsub_vx_u16mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsub_vx_u16mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vx_u16m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vx_u16m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vx_u16m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vx_u16m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vsub_vx_u32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vx_u32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vx_u32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vx_u32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vx_u32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vsub_vx_u64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vx_u64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vx_u64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vx_u64m8_m(mask, op1, op2, vl);
    }
}

// Unified Masked VECTOR_SUB template that auto-detects vector vs scalar
template<typename T, int LMUL, typename MaskType, typename VecType, typename Op2Type>
inline auto VECTOR_SUB_MASKED(MaskType mask, VecType op1, Op2Type op2, size_t vl) {
    // If op2 is a scalar type (arithmetic type), use vector-scalar masked subtraction
    if constexpr (std::is_arithmetic_v<Op2Type>) {
        return VECTOR_SUB_VX_M<T, LMUL>(mask, op1, static_cast<T>(op2), vl);
    }
    // Otherwise, use vector-vector masked subtraction
    else {
        return VECTOR_SUB_VV_M<T, LMUL>(mask, op1, op2, vl);
    }
}

/*************************************************************************************************/

// Vector-Vector Multiply Template
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_MUL_VV(VecType op1, VecType op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfmul_vv_f32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfmul_vv_f32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmul_vv_f32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmul_vv_f32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmul_vv_f32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfmul_vv_f64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmul_vv_f64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmul_vv_f64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmul_vv_f64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, _Float16>) {
        if constexpr (LMUL == MF4) return __riscv_vfmul_vv_f16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vfmul_vv_f16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfmul_vv_f16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmul_vv_f16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmul_vv_f16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmul_vv_f16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vmul_vv_i8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vmul_vv_i8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vv_i8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vv_i8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_i8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_i8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_i8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vmul_vv_i16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vv_i16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vv_i16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_i16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_i16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_i16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vmul_vv_i32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vv_i32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_i32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_i32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_i32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmul_vv_i64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_i64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_i64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_i64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vmul_vv_u8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vmul_vv_u8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vv_u8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vv_u8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_u8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_u8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_u8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vmul_vv_u16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vv_u16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vv_u16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_u16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_u16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_u16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vmul_vv_u32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vv_u32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_u32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_u32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_u32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmul_vv_u64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_u64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_u64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_u64m8(op1, op2, vl);
    }
}

// Vector-Scalar Multiply Template
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_MUL_VX(VecType op1, T op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfmul_vf_f32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfmul_vf_f32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmul_vf_f32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmul_vf_f32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmul_vf_f32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfmul_vf_f64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmul_vf_f64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmul_vf_f64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmul_vf_f64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, _Float16>) {
        if constexpr (LMUL == MF4) return __riscv_vfmul_vf_f16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vfmul_vf_f16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfmul_vf_f16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmul_vf_f16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmul_vf_f16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmul_vf_f16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vmul_vx_i8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vmul_vx_i8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vx_i8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vx_i8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_i8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_i8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_i8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vmul_vx_i16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vx_i16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vx_i16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_i16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_i16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_i16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vmul_vx_i32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vx_i32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_i32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_i32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_i32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmul_vx_i64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_i64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_i64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_i64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vmul_vx_u8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vmul_vx_u8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vx_u8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vx_u8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_u8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_u8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_u8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vmul_vx_u16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vx_u16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vx_u16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_u16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_u16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_u16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vmul_vx_u32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vx_u32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_u32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_u32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_u32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmul_vx_u64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_u64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_u64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_u64m8(op1, op2, vl);
    }
}

// Unified VECTOR_MULTIPLY template that auto-detects vector vs scalar
template<typename T, int LMUL, typename VecType, typename Op2Type>
inline auto VECTOR_MUL(VecType op1, Op2Type op2, size_t vl) {
    // If op2 is a scalar type (arithmetic type), use vector-scalar multiplication
    if constexpr (std::is_arithmetic_v<Op2Type>) {
        return VECTOR_MUL_VX<T, LMUL>(op1, static_cast<T>(op2), vl);
    }
    // Otherwise, use vector-vector multiplication
    else {
        return VECTOR_MUL_VV<T, LMUL>(op1, op2, vl);
    }
}

/*************************************************************************************************/

// Vector-Vector Masked Multiplication Template
template<typename T, int LMUL, typename VecType, typename MaskType>
inline auto VECTOR_MUL_VV_M(MaskType mask, VecType op1, VecType op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfmul_vv_f32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfmul_vv_f32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmul_vv_f32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmul_vv_f32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmul_vv_f32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfmul_vv_f64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmul_vv_f64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmul_vv_f64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmul_vv_f64m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, _Float16>) {
        if constexpr (LMUL == MF4) return __riscv_vfmul_vv_f16mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vfmul_vv_f16mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfmul_vv_f16m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmul_vv_f16m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmul_vv_f16m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmul_vv_f16m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vmul_vv_i8mf8_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vmul_vv_i8mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vv_i8mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vv_i8m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_i8m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_i8m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_i8m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vmul_vv_i16mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vv_i16mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vv_i16m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_i16m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_i16m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_i16m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vmul_vv_i32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vv_i32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_i32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_i32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_i32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmul_vv_i64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_i64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_i64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_i64m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vmul_vv_u8mf8_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vmul_vv_u8mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vv_u8mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vv_u8m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_u8m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_u8m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_u8m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vmul_vv_u16mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vv_u16mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vv_u16m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_u16m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_u16m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_u16m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vmul_vv_u32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vv_u32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_u32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_u32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_u32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmul_vv_u64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_u64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_u64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_u64m8_m(mask, op1, op2, vl);
    }
}

// Vector-Scalar Masked Multiplication Template
template<typename T, int LMUL, typename VecType, typename MaskType>
inline auto VECTOR_MUL_VX_M(MaskType mask, VecType op1, T op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfmul_vf_f32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfmul_vf_f32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmul_vf_f32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmul_vf_f32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmul_vf_f32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfmul_vf_f64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmul_vf_f64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmul_vf_f64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmul_vf_f64m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, _Float16>) {
        if constexpr (LMUL == MF4) return __riscv_vfmul_vf_f16mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vfmul_vf_f16mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfmul_vf_f16m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmul_vf_f16m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmul_vf_f16m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmul_vf_f16m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vmul_vx_i8mf8_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vmul_vx_i8mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vx_i8mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vx_i8m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_i8m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_i8m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_i8m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vmul_vx_i16mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vx_i16mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vx_i16m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_i16m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_i16m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_i16m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vmul_vx_i32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vx_i32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_i32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_i32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_i32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmul_vx_i64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_i64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_i64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_i64m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vmul_vx_u8mf8_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vmul_vx_u8mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vx_u8mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vx_u8m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_u8m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_u8m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_u8m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vmul_vx_u16mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vx_u16mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vx_u16m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_u16m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_u16m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_u16m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vmul_vx_u32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vx_u32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_u32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_u32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_u32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmul_vx_u64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_u64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_u64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_u64m8_m(mask, op1, op2, vl);
    }
}

// Unified Masked VECTOR_MULTIPLY template that auto-detects vector vs scalar
template<typename T, int LMUL, typename MaskType, typename VecType, typename Op2Type>
inline auto VECTOR_MUL_MASKED(MaskType mask, VecType op1, Op2Type op2, size_t vl) {
    // If op2 is a scalar type (arithmetic type), use vector-scalar masked multiplication
    if constexpr (std::is_arithmetic_v<Op2Type>) {
        return VECTOR_MUL_VX_M<T, LMUL>(mask, op1, static_cast<T>(op2), vl);
    }
    // Otherwise, use vector-vector masked multiplication
    else {
        return VECTOR_MUL_VV_M<T, LMUL>(mask, op1, op2, vl);
    }
}

/*************************************************************************************************/

// Vector-Vector Left Shift Template
template<typename T, int LMUL, typename VecType, typename ShiftType>
inline auto VECTOR_SLL_VV(VecType op1, ShiftType shift, size_t vl) {
    if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vsll_vv_i8mf8(op1, shift, vl);
        else if constexpr (LMUL == MF4) return __riscv_vsll_vv_i8mf4(op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsll_vv_i8mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vv_i8m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vv_i8m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vv_i8m4(op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vv_i8m8(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vsll_vv_i16mf4(op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsll_vv_i16mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vv_i16m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vv_i16m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vv_i16m4(op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vv_i16m8(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vsll_vv_i32mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vv_i32m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vv_i32m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vv_i32m4(op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vv_i32m8(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vsll_vv_i64m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vv_i64m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vv_i64m4(op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vv_i64m8(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vsll_vv_u8mf8(op1, shift, vl);
        else if constexpr (LMUL == MF4) return __riscv_vsll_vv_u8mf4(op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsll_vv_u8mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vv_u8m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vv_u8m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vv_u8m4(op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vv_u8m8(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vsll_vv_u16mf4(op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsll_vv_u16mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vv_u16m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vv_u16m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vv_u16m4(op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vv_u16m8(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vsll_vv_u32mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vv_u32m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vv_u32m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vv_u32m4(op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vv_u32m8(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vsll_vv_u64m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vv_u64m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vv_u64m4(op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vv_u64m8(op1, shift, vl);
    }
}

// Vector-Scalar Left Shift Template
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_SLL_VX(VecType op1, size_t shift, size_t vl) {
    if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vsll_vx_i8mf8(op1, shift, vl);
        else if constexpr (LMUL == MF4) return __riscv_vsll_vx_i8mf4(op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsll_vx_i8mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vx_i8m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vx_i8m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vx_i8m4(op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vx_i8m8(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vsll_vx_i16mf4(op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsll_vx_i16mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vx_i16m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vx_i16m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vx_i16m4(op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vx_i16m8(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vsll_vx_i32mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vx_i32m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vx_i32m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vx_i32m4(op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vx_i32m8(op1, shift, vl);
    }
	else if constexpr (std::is_same_v<T, int64_t>) {
    	if constexpr (LMUL == M1) return __riscv_vsll_vx_i64m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vx_i64m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vx_i64m4(op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vx_i64m8(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
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

// Unified VECTOR_SLL template that auto-detects vector vs scalar shift
template<typename T, int LMUL, typename VecType, typename ShiftType>
inline auto VECTOR_SLL(VecType op1, ShiftType shift, size_t vl) {
    // If shift is a scalar arithmetic type, use vector-scalar left shift
    if constexpr (std::is_arithmetic_v<ShiftType>) {
        return VECTOR_SLL_VX<T, LMUL>(op1, static_cast<size_t>(shift), vl);
    }
    // Otherwise, use vector-vector left shift
    else {
        return VECTOR_SLL_VV<T, LMUL>(op1, shift, vl);
    }
}

/*************************************************************************************************/

// Vector-Vector Masked Left Shift Template
template<typename T, int LMUL, typename VecType, typename ShiftType, typename MaskType>
inline auto VECTOR_SLL_VV_M(MaskType mask, VecType op1, ShiftType shift, size_t vl) {
    if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vsll_vv_i8mf8_m(mask, op1, shift, vl);
        else if constexpr (LMUL == MF4) return __riscv_vsll_vv_i8mf4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsll_vv_i8mf2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vv_i8m1_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vv_i8m2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vv_i8m4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vv_i8m8_m(mask, op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vsll_vv_i16mf4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsll_vv_i16mf2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vv_i16m1_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vv_i16m2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vv_i16m4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vv_i16m8_m(mask, op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vsll_vv_i32mf2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vv_i32m1_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vv_i32m2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vv_i32m4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vv_i32m8_m(mask, op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vsll_vv_i64m1_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vv_i64m2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vv_i64m4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vv_i64m8_m(mask, op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vsll_vv_u8mf8_m(mask, op1, shift, vl);
        else if constexpr (LMUL == MF4) return __riscv_vsll_vv_u8mf4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsll_vv_u8mf2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vv_u8m1_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vv_u8m2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vv_u8m4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vv_u8m8_m(mask, op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vsll_vv_u16mf4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsll_vv_u16mf2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vv_u16m1_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vv_u16m2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vv_u16m4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vv_u16m8_m(mask, op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vsll_vv_u32mf2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vv_u32m1_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vv_u32m2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vv_u32m4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vv_u32m8_m(mask, op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vsll_vv_u64m1_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vv_u64m2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vv_u64m4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vv_u64m8_m(mask, op1, shift, vl);
    }
}

// Vector-Scalar Masked Left Shift Template
template<typename T, int LMUL, typename VecType, typename MaskType>
inline auto VECTOR_SLL_VX_M(MaskType mask, VecType op1, size_t shift, size_t vl) {
    if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vsll_vx_i8mf8_m(mask, op1, shift, vl);
        else if constexpr (LMUL == MF4) return __riscv_vsll_vx_i8mf4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsll_vx_i8mf2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vx_i8m1_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vx_i8m2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vx_i8m4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vx_i8m8_m(mask, op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vsll_vx_i16mf4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsll_vx_i16mf2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vx_i16m1_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vx_i16m2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vx_i16m4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vx_i16m8_m(mask, op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vsll_vx_i32mf2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vx_i32m1_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vx_i32m2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vx_i32m4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vx_i32m8_m(mask, op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vsll_vx_i64m1_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vx_i64m2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vx_i64m4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vx_i64m8_m(mask, op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vsll_vx_u8mf8_m(mask, op1, shift, vl);
        else if constexpr (LMUL == MF4) return __riscv_vsll_vx_u8mf4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsll_vx_u8mf2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vx_u8m1_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vx_u8m2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vx_u8m4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vx_u8m8_m(mask, op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vsll_vx_u16mf4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsll_vx_u16mf2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vx_u16m1_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vx_u16m2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vx_u16m4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vx_u16m8_m(mask, op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vsll_vx_u32mf2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vsll_vx_u32m1_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vx_u32m2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vx_u32m4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vx_u32m8_m(mask, op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vsll_vx_u64m1_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vsll_vx_u64m2_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vsll_vx_u64m4_m(mask, op1, shift, vl);
        else if constexpr (LMUL == M8) return __riscv_vsll_vx_u64m8_m(mask, op1, shift, vl);
    }
}

// Unified Masked VECTOR_SLL template that auto-detects vector vs scalar shift
template<typename T, int LMUL, typename MaskType, typename VecType, typename ShiftType>
inline auto VECTOR_SLL_MASKED(MaskType mask, VecType op1, ShiftType shift, size_t vl) {
    // If shift is a scalar type (size_t), use vector-scalar masked left shift
    if constexpr (std::is_same_v<ShiftType, size_t>) {
        return VECTOR_SLL_VX_M<T, LMUL>(mask, op1, shift, vl);
    }
    // Otherwise, use vector-vector masked left shift
    else {
        return VECTOR_SLL_VV_M<T, LMUL>(mask, op1, shift, vl);
    }
}

/*************************************************************************************************/
#endif // RVV_ARITHMETIC_HPP