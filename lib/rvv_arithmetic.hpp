#ifndef RVV_ARITHMETIC_HPP
#define RVV_ARITHMETIC_HPP

#include <cstddef> 
#include <riscv_vector.h>
#include <type_traits>

/*
vfloat16mf4_t __riscv_vfsub_vv_f16mf4 (vfloat16mf4_t op1, vfloat16mf4_t op2, size_t vl);
vfloat16mf4_t __riscv_vfsub_vf_f16mf4 (vfloat16mf4_t op1, float16_t op2, size_t vl);
vfloat16mf2_t __riscv_vfsub_vv_f16mf2 (vfloat16mf2_t op1, vfloat16mf2_t op2, size_t vl);
vfloat16mf2_t __riscv_vfsub_vf_f16mf2 (vfloat16mf2_t op1, float16_t op2, size_t vl);
vfloat16m1_t __riscv_vfsub_vv_f16m1 (vfloat16m1_t op1, vfloat16m1_t op2, size_t vl);
vfloat16m1_t __riscv_vfsub_vf_f16m1 (vfloat16m1_t op1, float16_t op2, size_t vl);
vfloat16m2_t __riscv_vfsub_vv_f16m2 (vfloat16m2_t op1, vfloat16m2_t op2, size_t vl);
vfloat16m2_t __riscv_vfsub_vf_f16m2 (vfloat16m2_t op1, float16_t op2, size_t vl);
vfloat16m4_t __riscv_vfsub_vv_f16m4 (vfloat16m4_t op1, vfloat16m4_t op2, size_t vl);
vfloat16m4_t __riscv_vfsub_vf_f16m4 (vfloat16m4_t op1, float16_t op2, size_t vl);
vfloat16m8_t __riscv_vfsub_vv_f16m8 (vfloat16m8_t op1, vfloat16m8_t op2, size_t vl);
vfloat16m8_t __riscv_vfsub_vf_f16m8 (vfloat16m8_t op1, float16_t op2, size_t vl);
vfloat32mf2_t __riscv_vfsub_vv_f32mf2 (vfloat32mf2_t op1, vfloat32mf2_t op2, size_t vl);
vfloat32mf2_t __riscv_vfsub_vf_f32mf2 (vfloat32mf2_t op1, float32_t op2, size_t vl);
vfloat32m1_t __riscv_vfsub_vv_f32m1 (vfloat32m1_t op1, vfloat32m1_t op2, size_t vl);
vfloat32m1_t __riscv_vfsub_vf_f32m1 (vfloat32m1_t op1, float32_t op2, size_t vl);
vfloat32m2_t __riscv_vfsub_vv_f32m2 (vfloat32m2_t op1, vfloat32m2_t op2, size_t vl);
vfloat32m2_t __riscv_vfsub_vf_f32m2 (vfloat32m2_t op1, float32_t op2, size_t vl);
vfloat32m4_t __riscv_vfsub_vv_f32m4 (vfloat32m4_t op1, vfloat32m4_t op2, size_t vl);
vfloat32m4_t __riscv_vfsub_vf_f32m4 (vfloat32m4_t op1, float32_t op2, size_t vl);
vfloat32m8_t __riscv_vfsub_vv_f32m8 (vfloat32m8_t op1, vfloat32m8_t op2, size_t vl);
vfloat32m8_t __riscv_vfsub_vf_f32m8 (vfloat32m8_t op1, float32_t op2, size_t vl);
vfloat64m1_t __riscv_vfsub_vv_f64m1 (vfloat64m1_t op1, vfloat64m1_t op2, size_t vl);
vfloat64m1_t __riscv_vfsub_vf_f64m1 (vfloat64m1_t op1, float64_t op2, size_t vl);
vfloat64m2_t __riscv_vfsub_vv_f64m2 (vfloat64m2_t op1, vfloat64m2_t op2, size_t vl);
vfloat64m2_t __riscv_vfsub_vf_f64m2 (vfloat64m2_t op1, float64_t op2, size_t vl);
vfloat64m4_t __riscv_vfsub_vv_f64m4 (vfloat64m4_t op1, vfloat64m4_t op2, size_t vl);
vfloat64m4_t __riscv_vfsub_vf_f64m4 (vfloat64m4_t op1, float64_t op2, size_t vl);
vfloat64m8_t __riscv_vfsub_vv_f64m8 (vfloat64m8_t op1, vfloat64m8_t op2, size_t vl);
vfloat64m8_t __riscv_vfsub_vf_f64m8 (vfloat64m8_t op1, float64_t op2, size_t vl);
*/

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
}

// Unified VECTOR_SUB template that auto-detects vector vs scalar
template<typename T, int LMUL, typename VecType, typename Op2Type>
inline auto VECTOR_SUB(VecType op1, Op2Type op2, size_t vl) {
    // If op2 is a scalar type, use vector-scalar subtraction
    if constexpr (std::is_same_v<Op2Type, T>) {
        return VECTOR_SUB_VX<T, LMUL>(op1, op2, vl);
    }
    // Otherwise, use vector-vector subtraction
    else {
        return VECTOR_SUB_VV<T, LMUL>(op1, op2, vl);
    }
}

#endif // RVV_ARITHMETIC_HPP