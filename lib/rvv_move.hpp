#ifndef RVV_MOVE_HPP
#define RVV_MOVE_HPP

#include <cstddef> 
#include <riscv_vector.h>
#include <type_traits>

// Define LMUL constants
constexpr int MF8 = -8;
constexpr int MF4 = -4;
constexpr int MF2 = -2;
constexpr int M1 = 1;
constexpr int M2 = 2;
constexpr int M4 = 4;
constexpr int M8 = 8;

/*
vfloat16mf4_t __riscv_vmv_v_v_f16mf4 (vfloat16mf4_t src, size_t vl);
vfloat16mf4_t __riscv_vfmv_v_f_f16mf4 (float16_t src, size_t vl);
vfloat16mf2_t __riscv_vmv_v_v_f16mf2 (vfloat16mf2_t src, size_t vl);
vfloat16mf2_t __riscv_vfmv_v_f_f16mf2 (float16_t src, size_t vl);
vfloat16m1_t __riscv_vmv_v_v_f16m1 (vfloat16m1_t src, size_t vl);
vfloat16m1_t __riscv_vfmv_v_f_f16m1 (float16_t src, size_t vl);
vfloat16m2_t __riscv_vmv_v_v_f16m2 (vfloat16m2_t src, size_t vl);
vfloat16m2_t __riscv_vfmv_v_f_f16m2 (float16_t src, size_t vl);
vfloat16m4_t __riscv_vmv_v_v_f16m4 (vfloat16m4_t src, size_t vl);
vfloat16m4_t __riscv_vfmv_v_f_f16m4 (float16_t src, size_t vl);
vfloat16m8_t __riscv_vmv_v_v_f16m8 (vfloat16m8_t src, size_t vl);
vfloat16m8_t __riscv_vfmv_v_f_f16m8 (float16_t src, size_t vl);
vfloat32mf2_t __riscv_vmv_v_v_f32mf2 (vfloat32mf2_t src, size_t vl);
vfloat32mf2_t __riscv_vfmv_v_f_f32mf2 (float32_t src, size_t vl);
vfloat32m1_t __riscv_vmv_v_v_f32m1 (vfloat32m1_t src, size_t vl);
vfloat32m1_t __riscv_vfmv_v_f_f32m1 (float32_t src, size_t vl);
vfloat32m2_t __riscv_vmv_v_v_f32m2 (vfloat32m2_t src, size_t vl);
vfloat32m2_t __riscv_vfmv_v_f_f32m2 (float32_t src, size_t vl);
vfloat32m4_t __riscv_vmv_v_v_f32m4 (vfloat32m4_t src, size_t vl);
vfloat32m4_t __riscv_vfmv_v_f_f32m4 (float32_t src, size_t vl);
vfloat32m8_t __riscv_vmv_v_v_f32m8 (vfloat32m8_t src, size_t vl);
vfloat32m8_t __riscv_vfmv_v_f_f32m8 (float32_t src, size_t vl);
vfloat64m1_t __riscv_vmv_v_v_f64m1 (vfloat64m1_t src, size_t vl);
vfloat64m1_t __riscv_vfmv_v_f_f64m1 (float64_t src, size_t vl);
vfloat64m2_t __riscv_vmv_v_v_f64m2 (vfloat64m2_t src, size_t vl);
vfloat64m2_t __riscv_vfmv_v_f_f64m2 (float64_t src, size_t vl);
vfloat64m4_t __riscv_vmv_v_v_f64m4 (vfloat64m4_t src, size_t vl);
vfloat64m4_t __riscv_vfmv_v_f_f64m4 (float64_t src, size_t vl);
vfloat64m8_t __riscv_vmv_v_v_f64m8 (vfloat64m8_t src, size_t vl);
vfloat64m8_t __riscv_vfmv_v_f_f64m8 (float64_t src, size_t vl);

vint8mf8_t __riscv_vmv_v_v_i8mf8 (vint8mf8_t src, size_t vl);
vint8mf8_t __riscv_vmv_v_x_i8mf8 (int8_t src, size_t vl);
vint8mf4_t __riscv_vmv_v_v_i8mf4 (vint8mf4_t src, size_t vl);
vint8mf4_t __riscv_vmv_v_x_i8mf4 (int8_t src, size_t vl);
vint8mf2_t __riscv_vmv_v_v_i8mf2 (vint8mf2_t src, size_t vl);
vint8mf2_t __riscv_vmv_v_x_i8mf2 (int8_t src, size_t vl);
vint8m1_t __riscv_vmv_v_v_i8m1 (vint8m1_t src, size_t vl);
vint8m1_t __riscv_vmv_v_x_i8m1 (int8_t src, size_t vl);
vint8m2_t __riscv_vmv_v_v_i8m2 (vint8m2_t src, size_t vl);
vint8m2_t __riscv_vmv_v_x_i8m2 (int8_t src, size_t vl);
vint8m4_t __riscv_vmv_v_v_i8m4 (vint8m4_t src, size_t vl);
vint8m4_t __riscv_vmv_v_x_i8m4 (int8_t src, size_t vl);
vint8m8_t __riscv_vmv_v_v_i8m8 (vint8m8_t src, size_t vl);
vint8m8_t __riscv_vmv_v_x_i8m8 (int8_t src, size_t vl);
vint16mf4_t __riscv_vmv_v_v_i16mf4 (vint16mf4_t src, size_t vl);
vint16mf4_t __riscv_vmv_v_x_i16mf4 (int16_t src, size_t vl);
vint16mf2_t __riscv_vmv_v_v_i16mf2 (vint16mf2_t src, size_t vl);
vint16mf2_t __riscv_vmv_v_x_i16mf2 (int16_t src, size_t vl);
vint16m1_t __riscv_vmv_v_v_i16m1 (vint16m1_t src, size_t vl);
vint16m1_t __riscv_vmv_v_x_i16m1 (int16_t src, size_t vl);
vint16m2_t __riscv_vmv_v_v_i16m2 (vint16m2_t src, size_t vl);
vint16m2_t __riscv_vmv_v_x_i16m2 (int16_t src, size_t vl);
vint16m4_t __riscv_vmv_v_v_i16m4 (vint16m4_t src, size_t vl);
vint16m4_t __riscv_vmv_v_x_i16m4 (int16_t src, size_t vl);
vint16m8_t __riscv_vmv_v_v_i16m8 (vint16m8_t src, size_t vl);
vint16m8_t __riscv_vmv_v_x_i16m8 (int16_t src, size_t vl);
vint32mf2_t __riscv_vmv_v_v_i32mf2 (vint32mf2_t src, size_t vl);
vint32mf2_t __riscv_vmv_v_x_i32mf2 (int32_t src, size_t vl);
vint32m1_t __riscv_vmv_v_v_i32m1 (vint32m1_t src, size_t vl);
vint32m1_t __riscv_vmv_v_x_i32m1 (int32_t src, size_t vl);
vint32m2_t __riscv_vmv_v_v_i32m2 (vint32m2_t src, size_t vl);
vint32m2_t __riscv_vmv_v_x_i32m2 (int32_t src, size_t vl);
vint32m4_t __riscv_vmv_v_v_i32m4 (vint32m4_t src, size_t vl);
vint32m4_t __riscv_vmv_v_x_i32m4 (int32_t src, size_t vl);
vint32m8_t __riscv_vmv_v_v_i32m8 (vint32m8_t src, size_t vl);
vint32m8_t __riscv_vmv_v_x_i32m8 (int32_t src, size_t vl);
vint64m1_t __riscv_vmv_v_v_i64m1 (vint64m1_t src, size_t vl);
vint64m1_t __riscv_vmv_v_x_i64m1 (int64_t src, size_t vl);
vint64m2_t __riscv_vmv_v_v_i64m2 (vint64m2_t src, size_t vl);
vint64m2_t __riscv_vmv_v_x_i64m2 (int64_t src, size_t vl);
vint64m4_t __riscv_vmv_v_v_i64m4 (vint64m4_t src, size_t vl);
vint64m4_t __riscv_vmv_v_x_i64m4 (int64_t src, size_t vl);
vint64m8_t __riscv_vmv_v_v_i64m8 (vint64m8_t src, size_t vl);
*/

// Vector to Vector Copy (vmv_v_v)
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_COPY(VecType src, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == M1) return __riscv_vmv_v_v_f32m1(src, vl);
        else if constexpr (LMUL == M2) return __riscv_vmv_v_v_f32m2(src, vl);
        else if constexpr (LMUL == M4) return __riscv_vmv_v_v_f32m4(src, vl);
        else if constexpr (LMUL == M8) return __riscv_vmv_v_v_f32m8(src, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vmv_v_v_f64m1(src, vl);
        else if constexpr (LMUL == M2) return __riscv_vmv_v_v_f64m2(src, vl);
        else if constexpr (LMUL == M4) return __riscv_vmv_v_v_f64m4(src, vl);
        else if constexpr (LMUL == M8) return __riscv_vmv_v_v_f64m8(src, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == M1) return __riscv_vmv_v_v_i8m1(src, vl);
        else if constexpr (LMUL == M2) return __riscv_vmv_v_v_i8m2(src, vl);
        else if constexpr (LMUL == M4) return __riscv_vmv_v_v_i8m4(src, vl);
        else if constexpr (LMUL == M8) return __riscv_vmv_v_v_i8m8(src, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == M1) return __riscv_vmv_v_v_i16m1(src, vl);
        else if constexpr (LMUL == M2) return __riscv_vmv_v_v_i16m2(src, vl);
        else if constexpr (LMUL == M4) return __riscv_vmv_v_v_i16m4(src, vl);
        else if constexpr (LMUL == M8) return __riscv_vmv_v_v_i16m8(src, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == M1) return __riscv_vmv_v_v_i32m1(src, vl);
        else if constexpr (LMUL == M2) return __riscv_vmv_v_v_i32m2(src, vl);
        else if constexpr (LMUL == M4) return __riscv_vmv_v_v_i32m4(src, vl);
        else if constexpr (LMUL == M8) return __riscv_vmv_v_v_i32m8(src, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmv_v_v_i64m1(src, vl);
        else if constexpr (LMUL == M2) return __riscv_vmv_v_v_i64m2(src, vl);
        else if constexpr (LMUL == M4) return __riscv_vmv_v_v_i64m4(src, vl);
        else if constexpr (LMUL == M8) return __riscv_vmv_v_v_i64m8(src, vl);
    }
}

// Scalar to Vector Broadcast (vmv_v_x / vfmv_v_f)
template<typename T, int LMUL>
inline auto VECTOR_BROADCAST(T scalar, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == M1) return __riscv_vfmv_v_f_f32m1(scalar, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmv_v_f_f32m2(scalar, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmv_v_f_f32m4(scalar, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmv_v_f_f32m8(scalar, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfmv_v_f_f64m1(scalar, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmv_v_f_f64m2(scalar, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmv_v_f_f64m4(scalar, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmv_v_f_f64m8(scalar, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == M1) return __riscv_vmv_v_x_i8m1(scalar, vl);
        else if constexpr (LMUL == M2) return __riscv_vmv_v_x_i8m2(scalar, vl);
        else if constexpr (LMUL == M4) return __riscv_vmv_v_x_i8m4(scalar, vl);
        else if constexpr (LMUL == M8) return __riscv_vmv_v_x_i8m8(scalar, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == M1) return __riscv_vmv_v_x_i16m1(scalar, vl);
        else if constexpr (LMUL == M2) return __riscv_vmv_v_x_i16m2(scalar, vl);
        else if constexpr (LMUL == M4) return __riscv_vmv_v_x_i16m4(scalar, vl);
        else if constexpr (LMUL == M8) return __riscv_vmv_v_x_i16m8(scalar, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == M1) return __riscv_vmv_v_x_i32m1(scalar, vl);
        else if constexpr (LMUL == M2) return __riscv_vmv_v_x_i32m2(scalar, vl);
        else if constexpr (LMUL == M4) return __riscv_vmv_v_x_i32m4(scalar, vl);
        else if constexpr (LMUL == M8) return __riscv_vmv_v_x_i32m8(scalar, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmv_v_x_i64m1(scalar, vl);
        else if constexpr (LMUL == M2) return __riscv_vmv_v_x_i64m2(scalar, vl);
        else if constexpr (LMUL == M4) return __riscv_vmv_v_x_i64m4(scalar, vl);
        else if constexpr (LMUL == M8) return __riscv_vmv_v_x_i64m8(scalar, vl);
    }
}

// Unified VECTOR_MOVE template that handles both cases
template<typename T, int LMUL, typename SrcType>
inline auto VECTOR_MOVE(SrcType src, size_t vl) {
    // If src is a scalar type, broadcast it
    if constexpr (std::is_same_v<SrcType, T>) {
        return VECTOR_BROADCAST<T, LMUL>(src, vl);
    }
    // Otherwise, treat as vector-to-vector copy
    else {
        return VECTOR_COPY<T, LMUL>(src, vl);
    }
}

#endif // RVV_MOVE_HPP