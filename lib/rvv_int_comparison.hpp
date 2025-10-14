#ifndef RVV_INT_COMPARISON_HPP
#define RVV_INT_COMPARISON_HPP

#include <cstddef> 
#include <riscv_vector.h>
#include <type_traits>

/*
vbool64_t __riscv_vmfgt_vv_f16mf4_b64 (vfloat16mf4_t op1, vfloat16mf4_t op2, size_t vl);
vbool64_t __riscv_vmfgt_vf_f16mf4_b64 (vfloat16mf4_t op1, float16_t op2, size_t vl);
vbool32_t __riscv_vmfgt_vv_f16mf2_b32 (vfloat16mf2_t op1, vfloat16mf2_t op2, size_t vl);
vbool32_t __riscv_vmfgt_vf_f16mf2_b32 (vfloat16mf2_t op1, float16_t op2, size_t vl);
vbool16_t __riscv_vmfgt_vv_f16m1_b16 (vfloat16m1_t op1, vfloat16m1_t op2, size_t vl);
vbool16_t __riscv_vmfgt_vf_f16m1_b16 (vfloat16m1_t op1, float16_t op2, size_t vl);
vbool8_t __riscv_vmfgt_vv_f16m2_b8 (vfloat16m2_t op1, vfloat16m2_t op2, size_t vl);
vbool8_t __riscv_vmfgt_vf_f16m2_b8 (vfloat16m2_t op1, float16_t op2, size_t vl);
vbool4_t __riscv_vmfgt_vv_f16m4_b4 (vfloat16m4_t op1, vfloat16m4_t op2, size_t vl);
vbool4_t __riscv_vmfgt_vf_f16m4_b4 (vfloat16m4_t op1, float16_t op2, size_t vl);
vbool2_t __riscv_vmfgt_vv_f16m8_b2 (vfloat16m8_t op1, vfloat16m8_t op2, size_t vl);
vbool2_t __riscv_vmfgt_vf_f16m8_b2 (vfloat16m8_t op1, float16_t op2, size_t vl);
vbool64_t __riscv_vmfgt_vv_f32mf2_b64 (vfloat32mf2_t op1, vfloat32mf2_t op2, size_t vl);
vbool64_t __riscv_vmfgt_vf_f32mf2_b64 (vfloat32mf2_t op1, float32_t op2, size_t vl);
vbool32_t __riscv_vmfgt_vv_f32m1_b32 (vfloat32m1_t op1, vfloat32m1_t op2, size_t vl);
vbool32_t __riscv_vmfgt_vf_f32m1_b32 (vfloat32m1_t op1, float32_t op2, size_t vl);
vbool16_t __riscv_vmfgt_vv_f32m2_b16 (vfloat32m2_t op1, vfloat32m2_t op2, size_t vl);
vbool16_t __riscv_vmfgt_vf_f32m2_b16 (vfloat32m2_t op1, float32_t op2, size_t vl);
vbool8_t __riscv_vmfgt_vv_f32m4_b8 (vfloat32m4_t op1, vfloat32m4_t op2, size_t vl);
vbool8_t __riscv_vmfgt_vf_f32m4_b8 (vfloat32m4_t op1, float32_t op2, size_t vl);
vbool4_t __riscv_vmfgt_vv_f32m8_b4 (vfloat32m8_t op1, vfloat32m8_t op2, size_t vl);
vbool4_t __riscv_vmfgt_vf_f32m8_b4 (vfloat32m8_t op1, float32_t op2, size_t vl);
vbool64_t __riscv_vmfgt_vv_f64m1_b64 (vfloat64m1_t op1, vfloat64m1_t op2, size_t vl);
vbool64_t __riscv_vmfgt_vf_f64m1_b64 (vfloat64m1_t op1, float64_t op2, size_t vl);
vbool32_t __riscv_vmfgt_vv_f64m2_b32 (vfloat64m2_t op1, vfloat64m2_t op2, size_t vl);
vbool32_t __riscv_vmfgt_vf_f64m2_b32 (vfloat64m2_t op1, float64_t op2, size_t vl);
vbool16_t __riscv_vmfgt_vv_f64m4_b16 (vfloat64m4_t op1, vfloat64m4_t op2, size_t vl);
vbool16_t __riscv_vmfgt_vf_f64m4_b16 (vfloat64m4_t op1, float64_t op2, size_t vl);
vbool8_t __riscv_vmfgt_vv_f64m8_b8 (vfloat64m8_t op1, vfloat64m8_t op2, size_t vl);
vbool8_t __riscv_vmfgt_vf_f64m8_b8 (vfloat64m8_t op1, float64_t op2, size_t vl);

vbool64_t __riscv_vmsgt_vv_i8mf8_b64 (vint8mf8_t op1, vint8mf8_t op2, size_t vl);
vbool64_t __riscv_vmsgt_vx_i8mf8_b64 (vint8mf8_t op1, int8_t op2, size_t vl);
vbool32_t __riscv_vmsgt_vv_i8mf4_b32 (vint8mf4_t op1, vint8mf4_t op2, size_t vl);
vbool32_t __riscv_vmsgt_vx_i8mf4_b32 (vint8mf4_t op1, int8_t op2, size_t vl);
vbool16_t __riscv_vmsgt_vv_i8mf2_b16 (vint8mf2_t op1, vint8mf2_t op2, size_t vl);
vbool16_t __riscv_vmsgt_vx_i8mf2_b16 (vint8mf2_t op1, int8_t op2, size_t vl);
vbool8_t __riscv_vmsgt_vv_i8m1_b8 (vint8m1_t op1, vint8m1_t op2, size_t vl);
vbool8_t __riscv_vmsgt_vx_i8m1_b8 (vint8m1_t op1, int8_t op2, size_t vl);
vbool4_t __riscv_vmsgt_vv_i8m2_b4 (vint8m2_t op1, vint8m2_t op2, size_t vl);
vbool4_t __riscv_vmsgt_vx_i8m2_b4 (vint8m2_t op1, int8_t op2, size_t vl);
vbool2_t __riscv_vmsgt_vv_i8m4_b2 (vint8m4_t op1, vint8m4_t op2, size_t vl);
vbool2_t __riscv_vmsgt_vx_i8m4_b2 (vint8m4_t op1, int8_t op2, size_t vl);
vbool1_t __riscv_vmsgt_vv_i8m8_b1 (vint8m8_t op1, vint8m8_t op2, size_t vl);
vbool1_t __riscv_vmsgt_vx_i8m8_b1 (vint8m8_t op1, int8_t op2, size_t vl);
vbool64_t __riscv_vmsgt_vv_i16mf4_b64 (vint16mf4_t op1, vint16mf4_t op2, size_t vl);
vbool64_t __riscv_vmsgt_vx_i16mf4_b64 (vint16mf4_t op1, int16_t op2, size_t vl);
vbool32_t __riscv_vmsgt_vv_i16mf2_b32 (vint16mf2_t op1, vint16mf2_t op2, size_t vl);
vbool32_t __riscv_vmsgt_vx_i16mf2_b32 (vint16mf2_t op1, int16_t op2, size_t vl);
vbool16_t __riscv_vmsgt_vv_i16m1_b16 (vint16m1_t op1, vint16m1_t op2, size_t vl);
vbool16_t __riscv_vmsgt_vx_i16m1_b16 (vint16m1_t op1, int16_t op2, size_t vl);
vbool8_t __riscv_vmsgt_vv_i16m2_b8 (vint16m2_t op1, vint16m2_t op2, size_t vl);
vbool8_t __riscv_vmsgt_vx_i16m2_b8 (vint16m2_t op1, int16_t op2, size_t vl);
vbool4_t __riscv_vmsgt_vv_i16m4_b4 (vint16m4_t op1, vint16m4_t op2, size_t vl);
vbool4_t __riscv_vmsgt_vx_i16m4_b4 (vint16m4_t op1, int16_t op2, size_t vl);
vbool2_t __riscv_vmsgt_vv_i16m8_b2 (vint16m8_t op1, vint16m8_t op2, size_t vl);
vbool2_t __riscv_vmsgt_vx_i16m8_b2 (vint16m8_t op1, int16_t op2, size_t vl);
vbool64_t __riscv_vmsgt_vv_i32mf2_b64 (vint32mf2_t op1, vint32mf2_t op2, size_t vl);
vbool64_t __riscv_vmsgt_vx_i32mf2_b64 (vint32mf2_t op1, int32_t op2, size_t vl);
vbool32_t __riscv_vmsgt_vv_i32m1_b32 (vint32m1_t op1, vint32m1_t op2, size_t vl);
vbool32_t __riscv_vmsgt_vx_i32m1_b32 (vint32m1_t op1, int32_t op2, size_t vl);
vbool16_t __riscv_vmsgt_vv_i32m2_b16 (vint32m2_t op1, vint32m2_t op2, size_t vl);
vbool16_t __riscv_vmsgt_vx_i32m2_b16 (vint32m2_t op1, int32_t op2, size_t vl);
vbool8_t __riscv_vmsgt_vv_i32m4_b8 (vint32m4_t op1, vint32m4_t op2, size_t vl);
vbool8_t __riscv_vmsgt_vx_i32m4_b8 (vint32m4_t op1, int32_t op2, size_t vl);
vbool4_t __riscv_vmsgt_vv_i32m8_b4 (vint32m8_t op1, vint32m8_t op2, size_t vl);
vbool4_t __riscv_vmsgt_vx_i32m8_b4 (vint32m8_t op1, int32_t op2, size_t vl);
vbool64_t __riscv_vmsgt_vv_i64m1_b64 (vint64m1_t op1, vint64m1_t op2, size_t vl);
vbool64_t __riscv_vmsgt_vx_i64m1_b64 (vint64m1_t op1, int64_t op2, size_t vl);
vbool32_t __riscv_vmsgt_vv_i64m2_b32 (vint64m2_t op1, vint64m2_t op2, size_t vl);
vbool32_t __riscv_vmsgt_vx_i64m2_b32 (vint64m2_t op1, int64_t op2, size_t vl);
vbool16_t __riscv_vmsgt_vv_i64m4_b16 (vint64m4_t op1, vint64m4_t op2, size_t vl);
vbool16_t __riscv_vmsgt_vx_i64m4_b16 (vint64m4_t op1, int64_t op2, size_t vl);
vbool8_t __riscv_vmsgt_vv_i64m8_b8 (vint64m8_t op1, vint64m8_t op2, size_t vl);
vbool8_t __riscv_vmsgt_vx_i64m8_b8 (vint64m8_t op1, int64_t op2, size_t vl);
*/

// Vector-Vector comparison
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_GREATER_THAN(const VecType& op1, const VecType& op2, size_t vl) {
    if constexpr (std::is_same_v<T, _Float16>) {
        if constexpr (LMUL == MF4) return __riscv_vmfgt_vv_f16mf4_b64(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmfgt_vv_f16mf2_b32(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmfgt_vv_f16m1_b16(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmfgt_vv_f16m2_b8(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmfgt_vv_f16m4_b4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmfgt_vv_f16m8_b2(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vmfgt_vv_f32mf2_b64(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmfgt_vv_f32m1_b32(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmfgt_vv_f32m2_b16(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmfgt_vv_f32m4_b8(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmfgt_vv_f32m8_b4(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vmfgt_vv_f64m1_b64(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmfgt_vv_f64m2_b32(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmfgt_vv_f64m4_b16(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmfgt_vv_f64m8_b8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vmsgt_vv_i8mf8_b64(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vmsgt_vv_i8mf4_b32(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmsgt_vv_i8mf2_b16(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmsgt_vv_i8m1_b8(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmsgt_vv_i8m2_b4(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmsgt_vv_i8m4_b2(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmsgt_vv_i8m8_b1(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vmsgt_vv_i16mf4_b64(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmsgt_vv_i16mf2_b32(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmsgt_vv_i16m1_b16(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmsgt_vv_i16m2_b8(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmsgt_vv_i16m4_b4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmsgt_vv_i16m8_b2(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vmsgt_vv_i32mf2_b64(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmsgt_vv_i32m1_b32(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmsgt_vv_i32m2_b16(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmsgt_vv_i32m4_b8(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmsgt_vv_i32m8_b4(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmsgt_vv_i64m1_b64(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmsgt_vv_i64m2_b32(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmsgt_vv_i64m4_b16(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmsgt_vv_i64m8_b8(op1, op2, vl);
    }
}

// Vector-Scalar comparison
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_GREATER_THAN_SCALAR(const VecType& op1, T scalar, size_t vl) {
    if constexpr (std::is_same_v<T, _Float16>) {
        if constexpr (LMUL == MF4) return __riscv_vmfgt_vf_f16mf4_b64(op1, scalar, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmfgt_vf_f16mf2_b32(op1, scalar, vl);
        else if constexpr (LMUL == M1) return __riscv_vmfgt_vf_f16m1_b16(op1, scalar, vl);
        else if constexpr (LMUL == M2) return __riscv_vmfgt_vf_f16m2_b8(op1, scalar, vl);
        else if constexpr (LMUL == M4) return __riscv_vmfgt_vf_f16m4_b4(op1, scalar, vl);
        else if constexpr (LMUL == M8) return __riscv_vmfgt_vf_f16m8_b2(op1, scalar, vl);
    }
    else if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vmfgt_vf_f32mf2_b64(op1, scalar, vl);
        else if constexpr (LMUL == M1) return __riscv_vmfgt_vf_f32m1_b32(op1, scalar, vl);
        else if constexpr (LMUL == M2) return __riscv_vmfgt_vf_f32m2_b16(op1, scalar, vl);
        else if constexpr (LMUL == M4) return __riscv_vmfgt_vf_f32m4_b8(op1, scalar, vl);
        else if constexpr (LMUL == M8) return __riscv_vmfgt_vf_f32m8_b4(op1, scalar, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vmfgt_vf_f64m1_b64(op1, scalar, vl);
        else if constexpr (LMUL == M2) return __riscv_vmfgt_vf_f64m2_b32(op1, scalar, vl);
        else if constexpr (LMUL == M4) return __riscv_vmfgt_vf_f64m4_b16(op1, scalar, vl);
        else if constexpr (LMUL == M8) return __riscv_vmfgt_vf_f64m8_b8(op1, scalar, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vmsgt_vx_i8mf8_b64(op1, scalar, vl);
        else if constexpr (LMUL == MF4) return __riscv_vmsgt_vx_i8mf4_b32(op1, scalar, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmsgt_vx_i8mf2_b16(op1, scalar, vl);
        else if constexpr (LMUL == M1) return __riscv_vmsgt_vx_i8m1_b8(op1, scalar, vl);
        else if constexpr (LMUL == M2) return __riscv_vmsgt_vx_i8m2_b4(op1, scalar, vl);
        else if constexpr (LMUL == M4) return __riscv_vmsgt_vx_i8m4_b2(op1, scalar, vl);
        else if constexpr (LMUL == M8) return __riscv_vmsgt_vx_i8m8_b1(op1, scalar, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vmsgt_vx_i16mf4_b64(op1, scalar, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmsgt_vx_i16mf2_b32(op1, scalar, vl);
        else if constexpr (LMUL == M1) return __riscv_vmsgt_vx_i16m1_b16(op1, scalar, vl);
        else if constexpr (LMUL == M2) return __riscv_vmsgt_vx_i16m2_b8(op1, scalar, vl);
        else if constexpr (LMUL == M4) return __riscv_vmsgt_vx_i16m4_b4(op1, scalar, vl);
        else if constexpr (LMUL == M8) return __riscv_vmsgt_vx_i16m8_b2(op1, scalar, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vmsgt_vx_i32mf2_b64(op1, scalar, vl);
        else if constexpr (LMUL == M1) return __riscv_vmsgt_vx_i32m1_b32(op1, scalar, vl);
        else if constexpr (LMUL == M2) return __riscv_vmsgt_vx_i32m2_b16(op1, scalar, vl);
        else if constexpr (LMUL == M4) return __riscv_vmsgt_vx_i32m4_b8(op1, scalar, vl);
        else if constexpr (LMUL == M8) return __riscv_vmsgt_vx_i32m8_b4(op1, scalar, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmsgt_vx_i64m1_b64(op1, scalar, vl);
        else if constexpr (LMUL == M2) return __riscv_vmsgt_vx_i64m2_b32(op1, scalar, vl);
        else if constexpr (LMUL == M4) return __riscv_vmsgt_vx_i64m4_b16(op1, scalar, vl);
        else if constexpr (LMUL == M8) return __riscv_vmsgt_vx_i64m8_b8(op1, scalar, vl);
    }
}

#endif // RVV_INT_COMPARISON_HPP