#ifndef RVV_VECTOR_LOAD_HPP
#define RVV_VECTOR_LOAD_HPP

#include <cstddef> 
#include <riscv_vector.h>
#include <type_traits>

/*
vfloat16mf4_t __riscv_vle16_v_f16mf4 (const float16_t *base, size_t vl);
vfloat16mf2_t __riscv_vle16_v_f16mf2 (const float16_t *base, size_t vl);
vfloat16m1_t __riscv_vle16_v_f16m1 (const float16_t *base, size_t vl);
vfloat16m2_t __riscv_vle16_v_f16m2 (const float16_t *base, size_t vl);
vfloat16m4_t __riscv_vle16_v_f16m4 (const float16_t *base, size_t vl);
vfloat16m8_t __riscv_vle16_v_f16m8 (const float16_t *base, size_t vl);
vfloat32mf2_t __riscv_vle32_v_f32mf2 (const float32_t *base, size_t vl);
vfloat32m1_t __riscv_vle32_v_f32m1 (const float32_t *base, size_t vl);
vfloat32m2_t __riscv_vle32_v_f32m2 (const float32_t *base, size_t vl);
vfloat32m4_t __riscv_vle32_v_f32m4 (const float32_t *base, size_t vl);
vfloat32m8_t __riscv_vle32_v_f32m8 (const float32_t *base, size_t vl);
vfloat64m1_t __riscv_vle64_v_f64m1 (const float64_t *base, size_t vl);
vfloat64m2_t __riscv_vle64_v_f64m2 (const float64_t *base, size_t vl);
vfloat64m4_t __riscv_vle64_v_f64m4 (const float64_t *base, size_t vl);
vfloat64m8_t __riscv_vle64_v_f64m8 (const float64_t *base, size_t vl);
vint8mf8_t __riscv_vle8_v_i8mf8 (const int8_t *base, size_t vl);
vint8mf4_t __riscv_vle8_v_i8mf4 (const int8_t *base, size_t vl);
vint8mf2_t __riscv_vle8_v_i8mf2 (const int8_t *base, size_t vl);
vint8m1_t __riscv_vle8_v_i8m1 (const int8_t *base, size_t vl);
vint8m2_t __riscv_vle8_v_i8m2 (const int8_t *base, size_t vl);
vint8m4_t __riscv_vle8_v_i8m4 (const int8_t *base, size_t vl);
vint8m8_t __riscv_vle8_v_i8m8 (const int8_t *base, size_t vl);
vint16mf4_t __riscv_vle16_v_i16mf4 (const int16_t *base, size_t vl);
vint16mf2_t __riscv_vle16_v_i16mf2 (const int16_t *base, size_t vl);
vint16m1_t __riscv_vle16_v_i16m1 (const int16_t *base, size_t vl);
vint16m2_t __riscv_vle16_v_i16m2 (const int16_t *base, size_t vl);
vint16m4_t __riscv_vle16_v_i16m4 (const int16_t *base, size_t vl);
vint16m8_t __riscv_vle16_v_i16m8 (const int16_t *base, size_t vl);
vint32mf2_t __riscv_vle32_v_i32mf2 (const int32_t *base, size_t vl);
vint32m1_t __riscv_vle32_v_i32m1 (const int32_t *base, size_t vl);
vint32m2_t __riscv_vle32_v_i32m2 (const int32_t *base, size_t vl);
vint32m4_t __riscv_vle32_v_i32m4 (const int32_t *base, size_t vl);
vint32m8_t __riscv_vle32_v_i32m8 (const int32_t *base, size_t vl);
vint64m1_t __riscv_vle64_v_i64m1 (const int64_t *base, size_t vl);
vint64m2_t __riscv_vle64_v_i64m2 (const int64_t *base, size_t vl);
vint64m4_t __riscv_vle64_v_i64m4 (const int64_t *base, size_t vl);
vint64m8_t __riscv_vle64_v_i64m8 (const int64_t *base, size_t vl);
*/

template<typename T, int LMUL>
inline auto VECTOR_LOAD(const T* base, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == M1) return __riscv_vle32_v_f32m1(base, vl);
        else if constexpr (LMUL == M2) return __riscv_vle32_v_f32m2(base, vl);
        else if constexpr (LMUL == M4) return __riscv_vle32_v_f32m4(base, vl);
        else if constexpr (LMUL == M8) return __riscv_vle32_v_f32m8(base, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == M1) return __riscv_vle32_v_i32m1(base, vl);
        else if constexpr (LMUL == M2) return __riscv_vle32_v_i32m2(base, vl);
        else if constexpr (LMUL == M4) return __riscv_vle32_v_i32m4(base, vl);
        else if constexpr (LMUL == M8) return __riscv_vle32_v_i32m8(base, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vle64_v_f64m1(base, vl);
        else if constexpr (LMUL == M2) return __riscv_vle64_v_f64m2(base, vl);
        else if constexpr (LMUL == M4) return __riscv_vle64_v_f64m4(base, vl);
        else if constexpr (LMUL == M8) return __riscv_vle64_v_f64m8(base, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == M1) return __riscv_vle16_v_i16m1(base, vl);
        else if constexpr (LMUL == M2) return __riscv_vle16_v_i16m2(base, vl);
        else if constexpr (LMUL == M4) return __riscv_vle16_v_i16m4(base, vl);
        else if constexpr (LMUL == M8) return __riscv_vle16_v_i16m8(base, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == M1) return __riscv_vle8_v_i8m1(base, vl);
        else if constexpr (LMUL == M2) return __riscv_vle8_v_i8m2(base, vl);
        else if constexpr (LMUL == M4) return __riscv_vle8_v_i8m4(base, vl);
        else if constexpr (LMUL == M8) return __riscv_vle8_v_i8m8(base, vl);
    }
}

#endif // RVV_VECTOR_LOAD_HPP