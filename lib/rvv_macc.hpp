#ifndef RVV_MACC_HPP
#define RVV_MACC_HPP

#include <cstddef> 
#include <riscv_vector.h>
#include <type_traits>

/*
vfloat16mf4_t __riscv_vfmacc_vv_f16mf4 (vfloat16mf4_t vd, vfloat16mf4_t vs1, vfloat16mf4_t vs2, size_t vl);
vfloat16mf4_t __riscv_vfmacc_vf_f16mf4 (vfloat16mf4_t vd, float16_t rs1, vfloat16mf4_t vs2, size_t vl);
vfloat16mf2_t __riscv_vfmacc_vv_f16mf2 (vfloat16mf2_t vd, vfloat16mf2_t vs1, vfloat16mf2_t vs2, size_t vl);
vfloat16mf2_t __riscv_vfmacc_vf_f16mf2 (vfloat16mf2_t vd, float16_t rs1, vfloat16mf2_t vs2, size_t vl);
vfloat16m1_t __riscv_vfmacc_vv_f16m1 (vfloat16m1_t vd, vfloat16m1_t vs1, vfloat16m1_t vs2, size_t vl);
vfloat16m1_t __riscv_vfmacc_vf_f16m1 (vfloat16m1_t vd, float16_t rs1, vfloat16m1_t vs2, size_t vl);
vfloat16m2_t __riscv_vfmacc_vv_f16m2 (vfloat16m2_t vd, vfloat16m2_t vs1, vfloat16m2_t vs2, size_t vl);
vfloat16m2_t __riscv_vfmacc_vf_f16m2 (vfloat16m2_t vd, float16_t rs1, vfloat16m2_t vs2, size_t vl);
vfloat16m4_t __riscv_vfmacc_vv_f16m4 (vfloat16m4_t vd, vfloat16m4_t vs1, vfloat16m4_t vs2, size_t vl);
vfloat16m4_t __riscv_vfmacc_vf_f16m4 (vfloat16m4_t vd, float16_t rs1, vfloat16m4_t vs2, size_t vl);
vfloat16m8_t __riscv_vfmacc_vv_f16m8 (vfloat16m8_t vd, vfloat16m8_t vs1, vfloat16m8_t vs2, size_t vl);
vfloat16m8_t __riscv_vfmacc_vf_f16m8 (vfloat16m8_t vd, float16_t rs1, vfloat16m8_t vs2, size_t vl);
vfloat32mf2_t __riscv_vfmacc_vv_f32mf2 (vfloat32mf2_t vd, vfloat32mf2_t vs1, vfloat32mf2_t vs2, size_t vl);
vfloat32mf2_t __riscv_vfmacc_vf_f32mf2 (vfloat32mf2_t vd, float32_t rs1, vfloat32mf2_t vs2, size_t vl);
vfloat32m1_t __riscv_vfmacc_vv_f32m1 (vfloat32m1_t vd, vfloat32m1_t vs1, vfloat32m1_t vs2, size_t vl);
vfloat32m1_t __riscv_vfmacc_vf_f32m1 (vfloat32m1_t vd, float32_t rs1, vfloat32m1_t vs2, size_t vl);
vfloat32m2_t __riscv_vfmacc_vv_f32m2 (vfloat32m2_t vd, vfloat32m2_t vs1, vfloat32m2_t vs2, size_t vl);
vfloat32m2_t __riscv_vfmacc_vf_f32m2 (vfloat32m2_t vd, float32_t rs1, vfloat32m2_t vs2, size_t vl);
vfloat32m4_t __riscv_vfmacc_vv_f32m4 (vfloat32m4_t vd, vfloat32m4_t vs1, vfloat32m4_t vs2, size_t vl);
vfloat32m4_t __riscv_vfmacc_vf_f32m4 (vfloat32m4_t vd, float32_t rs1, vfloat32m4_t vs2, size_t vl);
vfloat32m8_t __riscv_vfmacc_vv_f32m8 (vfloat32m8_t vd, vfloat32m8_t vs1, vfloat32m8_t vs2, size_t vl);
vfloat32m8_t __riscv_vfmacc_vf_f32m8 (vfloat32m8_t vd, float32_t rs1, vfloat32m8_t vs2, size_t vl);
vfloat64m1_t __riscv_vfmacc_vv_f64m1 (vfloat64m1_t vd, vfloat64m1_t vs1, vfloat64m1_t vs2, size_t vl);
vfloat64m1_t __riscv_vfmacc_vf_f64m1 (vfloat64m1_t vd, float64_t rs1, vfloat64m1_t vs2, size_t vl);
vfloat64m2_t __riscv_vfmacc_vv_f64m2 (vfloat64m2_t vd, vfloat64m2_t vs1, vfloat64m2_t vs2, size_t vl);
vfloat64m2_t __riscv_vfmacc_vf_f64m2 (vfloat64m2_t vd, float64_t rs1, vfloat64m2_t vs2, size_t vl);
vfloat64m4_t __riscv_vfmacc_vv_f64m4 (vfloat64m4_t vd, vfloat64m4_t vs1, vfloat64m4_t vs2, size_t vl);
vfloat64m4_t __riscv_vfmacc_vf_f64m4 (vfloat64m4_t vd, float64_t rs1, vfloat64m4_t vs2, size_t vl);
vfloat64m8_t __riscv_vfmacc_vv_f64m8 (vfloat64m8_t vd, vfloat64m8_t vs1, vfloat64m8_t vs2, size_t vl);
vfloat64m8_t __riscv_vfmacc_vf_f64m8 (vfloat64m8_t vd, float64_t rs1, vfloat64m8_t vs2, size_t vl);

vint8mf8_t __riscv_vmacc_vv_i8mf8 (vint8mf8_t vd, vint8mf8_t vs1, vint8mf8_t vs2, size_t vl);
vint8mf8_t __riscv_vmacc_vx_i8mf8 (vint8mf8_t vd, int8_t rs1, vint8mf8_t vs2, size_t vl);
vint8mf4_t __riscv_vmacc_vv_i8mf4 (vint8mf4_t vd, vint8mf4_t vs1, vint8mf4_t vs2, size_t vl);
vint8mf4_t __riscv_vmacc_vx_i8mf4 (vint8mf4_t vd, int8_t rs1, vint8mf4_t vs2, size_t vl);
vint8mf2_t __riscv_vmacc_vv_i8mf2 (vint8mf2_t vd, vint8mf2_t vs1, vint8mf2_t vs2, size_t vl);
vint8mf2_t __riscv_vmacc_vx_i8mf2 (vint8mf2_t vd, int8_t rs1, vint8mf2_t vs2, size_t vl);
vint8m1_t __riscv_vmacc_vv_i8m1 (vint8m1_t vd, vint8m1_t vs1, vint8m1_t vs2, size_t vl);
vint8m1_t __riscv_vmacc_vx_i8m1 (vint8m1_t vd, int8_t rs1, vint8m1_t vs2, size_t vl);
vint8m2_t __riscv_vmacc_vv_i8m2 (vint8m2_t vd, vint8m2_t vs1, vint8m2_t vs2, size_t vl);
vint8m2_t __riscv_vmacc_vx_i8m2 (vint8m2_t vd, int8_t rs1, vint8m2_t vs2, size_t vl);
vint8m4_t __riscv_vmacc_vv_i8m4 (vint8m4_t vd, vint8m4_t vs1, vint8m4_t vs2, size_t vl);
vint8m4_t __riscv_vmacc_vx_i8m4 (vint8m4_t vd, int8_t rs1, vint8m4_t vs2, size_t vl);
vint8m8_t __riscv_vmacc_vv_i8m8 (vint8m8_t vd, vint8m8_t vs1, vint8m8_t vs2, size_t vl);
vint8m8_t __riscv_vmacc_vx_i8m8 (vint8m8_t vd, int8_t rs1, vint8m8_t vs2, size_t vl);
vint16mf4_t __riscv_vmacc_vv_i16mf4 (vint16mf4_t vd, vint16mf4_t vs1, vint16mf4_t vs2, size_t vl);
vint16mf4_t __riscv_vmacc_vx_i16mf4 (vint16mf4_t vd, int16_t rs1, vint16mf4_t vs2, size_t vl);
vint16mf2_t __riscv_vmacc_vv_i16mf2 (vint16mf2_t vd, vint16mf2_t vs1, vint16mf2_t vs2, size_t vl);
vint16mf2_t __riscv_vmacc_vx_i16mf2 (vint16mf2_t vd, int16_t rs1, vint16mf2_t vs2, size_t vl);
vint16m1_t __riscv_vmacc_vv_i16m1 (vint16m1_t vd, vint16m1_t vs1, vint16m1_t vs2, size_t vl);
vint16m1_t __riscv_vmacc_vx_i16m1 (vint16m1_t vd, int16_t rs1, vint16m1_t vs2, size_t vl);
vint16m2_t __riscv_vmacc_vv_i16m2 (vint16m2_t vd, vint16m2_t vs1, vint16m2_t vs2, size_t vl);
vint16m2_t __riscv_vmacc_vx_i16m2 (vint16m2_t vd, int16_t rs1, vint16m2_t vs2, size_t vl);
vint16m4_t __riscv_vmacc_vv_i16m4 (vint16m4_t vd, vint16m4_t vs1, vint16m4_t vs2, size_t vl);
vint16m4_t __riscv_vmacc_vx_i16m4 (vint16m4_t vd, int16_t rs1, vint16m4_t vs2, size_t vl);
vint16m8_t __riscv_vmacc_vv_i16m8 (vint16m8_t vd, vint16m8_t vs1, vint16m8_t vs2, size_t vl);
vint16m8_t __riscv_vmacc_vx_i16m8 (vint16m8_t vd, int16_t rs1, vint16m8_t vs2, size_t vl);
vint32mf2_t __riscv_vmacc_vv_i32mf2 (vint32mf2_t vd, vint32mf2_t vs1, vint32mf2_t vs2, size_t vl);
vint32mf2_t __riscv_vmacc_vx_i32mf2 (vint32mf2_t vd, int32_t rs1, vint32mf2_t vs2, size_t vl);
vint32m1_t __riscv_vmacc_vv_i32m1 (vint32m1_t vd, vint32m1_t vs1, vint32m1_t vs2, size_t vl);
vint32m1_t __riscv_vmacc_vx_i32m1 (vint32m1_t vd, int32_t rs1, vint32m1_t vs2, size_t vl);
vint32m2_t __riscv_vmacc_vv_i32m2 (vint32m2_t vd, vint32m2_t vs1, vint32m2_t vs2, size_t vl);
vint32m2_t __riscv_vmacc_vx_i32m2 (vint32m2_t vd, int32_t rs1, vint32m2_t vs2, size_t vl);
vint32m4_t __riscv_vmacc_vv_i32m4 (vint32m4_t vd, vint32m4_t vs1, vint32m4_t vs2, size_t vl);
vint32m4_t __riscv_vmacc_vx_i32m4 (vint32m4_t vd, int32_t rs1, vint32m4_t vs2, size_t vl);
vint32m8_t __riscv_vmacc_vv_i32m8 (vint32m8_t vd, vint32m8_t vs1, vint32m8_t vs2, size_t vl);
vint32m8_t __riscv_vmacc_vx_i32m8 (vint32m8_t vd, int32_t rs1, vint32m8_t vs2, size_t vl);
vint64m1_t __riscv_vmacc_vv_i64m1 (vint64m1_t vd, vint64m1_t vs1, vint64m1_t vs2, size_t vl);
vint64m1_t __riscv_vmacc_vx_i64m1 (vint64m1_t vd, int64_t rs1, vint64m1_t vs2, size_t vl);
vint64m2_t __riscv_vmacc_vv_i64m2 (vint64m2_t vd, vint64m2_t vs1, vint64m2_t vs2, size_t vl);
vint64m2_t __riscv_vmacc_vx_i64m2 (vint64m2_t vd, int64_t rs1, vint64m2_t vs2, size_t vl);
vint64m4_t __riscv_vmacc_vv_i64m4 (vint64m4_t vd, vint64m4_t vs1, vint64m4_t vs2, size_t vl);
vint64m4_t __riscv_vmacc_vx_i64m4 (vint64m4_t vd, int64_t rs1, vint64m4_t vs2, size_t vl);
vint64m8_t __riscv_vmacc_vv_i64m8 (vint64m8_t vd, vint64m8_t vs1, vint64m8_t vs2, size_t vl);
vint64m8_t __riscv_vmacc_vx_i64m8 (vint64m8_t vd, int64_t rs1, vint64m8_t vs2, size_t vl);
*/

template<typename T, int LMUL, typename VD, typename VS1, typename VS2>
inline auto VECTOR_FMACC_VV(VD vd, VS1 vs1, VS2 vs2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == M1) return __riscv_vfmacc_vv_f32m1(vd, vs1, vs2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmacc_vv_f32m2(vd, vs1, vs2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmacc_vv_f32m4(vd, vs1, vs2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmacc_vv_f32m8(vd, vs1, vs2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfmacc_vv_f64m1(vd, vs1, vs2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmacc_vv_f64m2(vd, vs1, vs2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmacc_vv_f64m4(vd, vs1, vs2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmacc_vv_f64m8(vd, vs1, vs2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == M1) return __riscv_vmacc_vv_i32m1(vd, vs1, vs2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmacc_vv_i32m2(vd, vs1, vs2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmacc_vv_i32m4(vd, vs1, vs2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmacc_vv_i32m8(vd, vs1, vs2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmacc_vv_i64m1(vd, vs1, vs2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmacc_vv_i64m2(vd, vs1, vs2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmacc_vv_i64m4(vd, vs1, vs2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmacc_vv_i64m8(vd, vs1, vs2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == M1) return __riscv_vmacc_vv_i16m1(vd, vs1, vs2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmacc_vv_i16m2(vd, vs1, vs2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmacc_vv_i16m4(vd, vs1, vs2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmacc_vv_i16m8(vd, vs1, vs2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == M1) return __riscv_vmacc_vv_i8m1(vd, vs1, vs2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmacc_vv_i8m2(vd, vs1, vs2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmacc_vv_i8m4(vd, vs1, vs2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmacc_vv_i8m8(vd, vs1, vs2, vl);
    }
}

template<typename T, int LMUL, typename VD, typename VS2>
inline auto VECTOR_FMACC_VF(VD vd, T rs1, VS2 vs2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == M1) return __riscv_vfmacc_vf_f32m1(vd, rs1, vs2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmacc_vf_f32m2(vd, rs1, vs2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmacc_vf_f32m4(vd, rs1, vs2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmacc_vf_f32m8(vd, rs1, vs2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfmacc_vf_f64m1(vd, rs1, vs2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmacc_vf_f64m2(vd, rs1, vs2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmacc_vf_f64m4(vd, rs1, vs2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmacc_vf_f64m8(vd, rs1, vs2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == M1) return __riscv_vmacc_vx_i32m1(vd, rs1, vs2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmacc_vx_i32m2(vd, rs1, vs2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmacc_vx_i32m4(vd, rs1, vs2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmacc_vx_i32m8(vd, rs1, vs2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmacc_vx_i64m1(vd, rs1, vs2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmacc_vx_i64m2(vd, rs1, vs2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmacc_vx_i64m4(vd, rs1, vs2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmacc_vx_i64m8(vd, rs1, vs2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == M1) return __riscv_vmacc_vx_i16m1(vd, rs1, vs2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmacc_vx_i16m2(vd, rs1, vs2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmacc_vx_i16m4(vd, rs1, vs2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmacc_vx_i16m8(vd, rs1, vs2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == M1) return __riscv_vmacc_vx_i8m1(vd, rs1, vs2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmacc_vx_i8m2(vd, rs1, vs2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmacc_vx_i8m4(vd, rs1, vs2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmacc_vx_i8m8(vd, rs1, vs2, vl);
    }
}

// Unified VECTOR_FMACC template that handles both VV and VF/VX cases
template<typename T, int LMUL, typename SrcType, typename VD, typename VS2>
inline auto VECTOR_FMACC(VD vd, SrcType src, VS2 vs2, size_t vl) {
    // If src is a scalar type, use VF/VX variant
    if constexpr (std::is_same_v<SrcType, T>) {
        return VECTOR_FMACC_VF<T, LMUL>(vd, src, vs2, vl);
    }
    // Otherwise, treat as vector-to-vector (VV variant)
    else {
        return VECTOR_FMACC_VV<T, LMUL>(vd, src, vs2, vl);
    }
}

#endif // RVV_MACC_HPP