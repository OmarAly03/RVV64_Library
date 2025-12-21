#ifndef RVV_VECTOR_STORE_HPP
#define RVV_VECTOR_STORE_HPP

#include <cstddef> 
#include <riscv_vector.h>
#include <type_traits>

/*
void __riscv_vse16_v_f16mf4 (float16_t *base, vfloat16mf4_t value, size_t vl);
void __riscv_vse16_v_f16mf2 (float16_t *base, vfloat16mf2_t value, size_t vl);
void __riscv_vse16_v_f16m1 (float16_t *base, vfloat16m1_t value, size_t vl);
void __riscv_vse16_v_f16m2 (float16_t *base, vfloat16m2_t value, size_t vl);
void __riscv_vse16_v_f16m4 (float16_t *base, vfloat16m4_t value, size_t vl);
void __riscv_vse16_v_f16m8 (float16_t *base, vfloat16m8_t value, size_t vl);
void __riscv_vse32_v_f32mf2 (float32_t *base, vfloat32mf2_t value, size_t vl);
void __riscv_vse32_v_f32m1 (float32_t *base, vfloat32m1_t value, size_t vl);
void __riscv_vse32_v_f32m2 (float32_t *base, vfloat32m2_t value, size_t vl);
void __riscv_vse32_v_f32m4 (float32_t *base, vfloat32m4_t value, size_t vl);
void __riscv_vse32_v_f32m8 (float32_t *base, vfloat32m8_t value, size_t vl);
void __riscv_vse64_v_f64m1 (float64_t *base, vfloat64m1_t value, size_t vl);
void __riscv_vse64_v_f64m2 (float64_t *base, vfloat64m2_t value, size_t vl);
void __riscv_vse64_v_f64m4 (float64_t *base, vfloat64m4_t value, size_t vl);
void __riscv_vse64_v_f64m8 (float64_t *base, vfloat64m8_t value, size_t vl);
void __riscv_vse8_v_i8mf8 (int8_t *base, vint8mf8_t value, size_t vl);
void __riscv_vse8_v_i8mf4 (int8_t *base, vint8mf4_t value, size_t vl);
void __riscv_vse8_v_i8mf2 (int8_t *base, vint8mf2_t value, size_t vl);
void __riscv_vse8_v_i8m1 (int8_t *base, vint8m1_t value, size_t vl);
void __riscv_vse8_v_i8m2 (int8_t *base, vint8m2_t value, size_t vl);
void __riscv_vse8_v_i8m4 (int8_t *base, vint8m4_t value, size_t vl);
void __riscv_vse8_v_i8m8 (int8_t *base, vint8m8_t value, size_t vl);
void __riscv_vse16_v_i16mf4 (int16_t *base, vint16mf4_t value, size_t vl);
void __riscv_vse16_v_i16mf2 (int16_t *base, vint16mf2_t value, size_t vl);
void __riscv_vse16_v_i16m1 (int16_t *base, vint16m1_t value, size_t vl);
void __riscv_vse16_v_i16m2 (int16_t *base, vint16m2_t value, size_t vl);
void __riscv_vse16_v_i16m4 (int16_t *base, vint16m4_t value, size_t vl);
void __riscv_vse16_v_i16m8 (int16_t *base, vint16m8_t value, size_t vl);
void __riscv_vse32_v_i32mf2 (int32_t *base, vint32mf2_t value, size_t vl);
void __riscv_vse32_v_i32m1 (int32_t *base, vint32m1_t value, size_t vl);
void __riscv_vse32_v_i32m2 (int32_t *base, vint32m2_t value, size_t vl);
void __riscv_vse32_v_i32m4 (int32_t *base, vint32m4_t value, size_t vl);
void __riscv_vse32_v_i32m8 (int32_t *base, vint32m8_t value, size_t vl);
void __riscv_vse64_v_i64m1 (int64_t *base, vint64m1_t value, size_t vl);
void __riscv_vse64_v_i64m2 (int64_t *base, vint64m2_t value, size_t vl);
void __riscv_vse64_v_i64m4 (int64_t *base, vint64m4_t value, size_t vl);
void __riscv_vse64_v_i64m8 (int64_t *base, vint64m8_t value, size_t vl);
*/

template<typename T, int LMUL, typename VecType>
inline void VECTOR_STORE(T* base, VecType value, size_t vl) {
	if constexpr (std::is_same_v<T, _Float16>) {
		if constexpr (LMUL == MF4) __riscv_vse16_v_f16mf4(base, value, vl);
		else if constexpr (LMUL == MF2) __riscv_vse16_v_f16mf2(base, value, vl);
		else if constexpr (LMUL == M1) __riscv_vse16_v_f16m1(base, value, vl);
		else if constexpr (LMUL == M2) __riscv_vse16_v_f16m2(base, value, vl);
		else if constexpr (LMUL == M4) __riscv_vse16_v_f16m4(base, value, vl);
		else if constexpr (LMUL == M8) __riscv_vse16_v_f16m8(base, value, vl);
	}
	else if constexpr (std::is_same_v<T, float>) {
		if constexpr (LMUL == MF2) __riscv_vse32_v_f32mf2(base, value, vl);
		else if constexpr (LMUL == M1) __riscv_vse32_v_f32m1(base, value, vl);
		else if constexpr (LMUL == M2) __riscv_vse32_v_f32m2(base, value, vl);
		else if constexpr (LMUL == M4) __riscv_vse32_v_f32m4(base, value, vl);
		else if constexpr (LMUL == M8) __riscv_vse32_v_f32m8(base, value, vl);
	}
	else if constexpr (std::is_same_v<T, double>) {
		if constexpr (LMUL == M1) __riscv_vse64_v_f64m1(base, value, vl);
		else if constexpr (LMUL == M2) __riscv_vse64_v_f64m2(base, value, vl);
		else if constexpr (LMUL == M4) __riscv_vse64_v_f64m4(base, value, vl);
		else if constexpr (LMUL == M8) __riscv_vse64_v_f64m8(base, value, vl);
	}
	else if constexpr (std::is_same_v<T, int8_t>) {
		if constexpr (LMUL == MF8) __riscv_vse8_v_i8mf8(base, value, vl);
		else if constexpr (LMUL == MF4) __riscv_vse8_v_i8mf4(base, value, vl);
		else if constexpr (LMUL == MF2) __riscv_vse8_v_i8mf2(base, value, vl);
		else if constexpr (LMUL == M1) __riscv_vse8_v_i8m1(base, value, vl);
		else if constexpr (LMUL == M2) __riscv_vse8_v_i8m2(base, value, vl);
		else if constexpr (LMUL == M4) __riscv_vse8_v_i8m4(base, value, vl);
		else if constexpr (LMUL == M8) __riscv_vse8_v_i8m8(base, value, vl);
	}
	else if constexpr (std::is_same_v<T, int16_t>) {
		if constexpr (LMUL == MF4) __riscv_vse16_v_i16mf4(base, value, vl);
		else if constexpr (LMUL == MF2) __riscv_vse16_v_i16mf2(base, value, vl);
		else if constexpr (LMUL == M1) __riscv_vse16_v_i16m1(base, value, vl);
		else if constexpr (LMUL == M2) __riscv_vse16_v_i16m2(base, value, vl);
		else if constexpr (LMUL == M4) __riscv_vse16_v_i16m4(base, value, vl);
		else if constexpr (LMUL == M8) __riscv_vse16_v_i16m8(base, value, vl);
	}
	else if constexpr (std::is_same_v<T, int32_t>) {
		if constexpr (LMUL == MF2) __riscv_vse32_v_i32mf2(base, value, vl);
		else if constexpr (LMUL == M1) __riscv_vse32_v_i32m1(base, value, vl);
		else if constexpr (LMUL == M2) __riscv_vse32_v_i32m2(base, value, vl);
		else if constexpr (LMUL == M4) __riscv_vse32_v_i32m4(base, value, vl);
		else if constexpr (LMUL == M8) __riscv_vse32_v_i32m8(base, value, vl);
	}
	else if constexpr (std::is_same_v<T, int64_t>) {
		if constexpr (LMUL == M1) __riscv_vse64_v_i64m1(base, value, vl);
		else if constexpr (LMUL == M2) __riscv_vse64_v_i64m2(base, value, vl);
		else if constexpr (LMUL == M4) __riscv_vse64_v_i64m4(base, value, vl);
		else if constexpr (LMUL == M8) __riscv_vse64_v_i64m8(base, value, vl);
	}
	else if constexpr (std::is_same_v<T, uint8_t>) {
		if constexpr (LMUL == MF8) __riscv_vse8_v_u8mf8(base, value, vl);
		else if constexpr (LMUL == MF4) __riscv_vse8_v_u8mf4(base, value, vl);
		else if constexpr (LMUL == MF2) __riscv_vse8_v_u8mf2(base, value, vl);
		else if constexpr (LMUL == M1) __riscv_vse8_v_u8m1(base, value, vl);
		else if constexpr (LMUL == M2) __riscv_vse8_v_u8m2(base, value, vl);
		else if constexpr (LMUL == M4) __riscv_vse8_v_u8m4(base, value, vl);
		else if constexpr (LMUL == M8) __riscv_vse8_v_u8m8(base, value, vl);
	}
	else if constexpr (std::is_same_v<T, uint16_t>) {
		if constexpr (LMUL == MF4) __riscv_vse16_v_u16mf4(base, value, vl);
		else if constexpr (LMUL == MF2) __riscv_vse16_v_u16mf2(base, value, vl);
		else if constexpr (LMUL == M1) __riscv_vse16_v_u16m1(base, value, vl);
		else if constexpr (LMUL == M2) __riscv_vse16_v_u16m2(base, value, vl);
		else if constexpr (LMUL == M4) __riscv_vse16_v_u16m4(base, value, vl);
		else if constexpr (LMUL == M8) __riscv_vse16_v_u16m8(base, value, vl);
	}
	else if constexpr (std::is_same_v<T, uint32_t>) {
		if constexpr (LMUL == MF2) __riscv_vse32_v_u32mf2(base, value, vl);
		else if constexpr (LMUL == M1) __riscv_vse32_v_u32m1(base, value, vl);
		else if constexpr (LMUL == M2) __riscv_vse32_v_u32m2(base, value, vl);
		else if constexpr (LMUL == M4) __riscv_vse32_v_u32m4(base, value, vl);
		else if constexpr (LMUL == M8) __riscv_vse32_v_u32m8(base, value, vl);
	}
	else if constexpr (std::is_same_v<T, uint64_t>) {
		if constexpr (LMUL == M1) __riscv_vse64_v_u64m1(base, value, vl);
		else if constexpr (LMUL == M2) __riscv_vse64_v_u64m2(base, value, vl);
		else if constexpr (LMUL == M4) __riscv_vse64_v_u64m4(base, value, vl);
		else if constexpr (LMUL == M8) __riscv_vse64_v_u64m8(base, value, vl);
	}
}

template<typename T, int LMUL, typename VecType>
inline void VECTOR_STRIDED_STORE(T* base, ptrdiff_t stride, VecType value, size_t vl) {
	if constexpr (std::is_same_v<T, float>) {
		if constexpr (LMUL == M1) __riscv_vsse32_v_f32m1(base, stride, value, vl);
		else if constexpr (LMUL == M2) __riscv_vsse32_v_f32m2(base, stride, value, vl);
		else if constexpr (LMUL == M4) __riscv_vsse32_v_f32m4(base, stride, value, vl);
		else if constexpr (LMUL == M8) __riscv_vsse32_v_f32m8(base, stride, value, vl);
	}
	else if constexpr (std::is_same_v<T, int32_t>) {
		if constexpr (LMUL == M1) __riscv_vsse32_v_i32m1(base, stride, value, vl);
		else if constexpr (LMUL == M2) __riscv_vsse32_v_i32m2(base, stride, value, vl);
		else if constexpr (LMUL == M4) __riscv_vsse32_v_i32m4(base, stride, value, vl);
		else if constexpr (LMUL == M8) __riscv_vsse32_v_i32m8(base, stride, value, vl);
	}
	else if constexpr (std::is_same_v<T, double>) {
		if constexpr (LMUL == M1) __riscv_vsse64_v_f64m1(base, stride, value, vl);
		else if constexpr (LMUL == M2) __riscv_vsse64_v_f64m2(base, stride, value, vl);
		else if constexpr (LMUL == M4) __riscv_vsse64_v_f64m4(base, stride, value, vl);
		else if constexpr (LMUL == M8) __riscv_vsse64_v_f64m8(base, stride, value, vl);
	}
	else if constexpr (std::is_same_v<T, int64_t>) {
		if constexpr (LMUL == M1) __riscv_vsse64_v_i64m1(base, stride, value, vl);
		else if constexpr (LMUL == M2) __riscv_vsse64_v_i64m2(base, stride, value, vl);
		else if constexpr (LMUL == M4) __riscv_vsse64_v_i64m4(base, stride, value, vl);
		else if constexpr (LMUL == M8) __riscv_vsse64_v_i64m8(base, stride, value, vl);
	}
	else if constexpr (std::is_same_v<T, int16_t>) {
		if constexpr (LMUL == M1) __riscv_vsse16_v_i16m1(base, stride, value, vl);
		else if constexpr (LMUL == M2) __riscv_vsse16_v_i16m2(base, stride, value, vl);
		else if constexpr (LMUL == M4) __riscv_vsse16_v_i16m4(base, stride, value, vl);
		else if constexpr (LMUL == M8) __riscv_vsse16_v_i16m8(base, stride, value, vl);
	}
	else if constexpr (std::is_same_v<T, int8_t>) {
		if constexpr (LMUL == M1) __riscv_vsse8_v_i8m1(base, stride, value, vl);
		else if constexpr (LMUL == M2) __riscv_vsse8_v_i8m2(base, stride, value, vl);
		else if constexpr (LMUL == M4) __riscv_vsse8_v_i8m4(base, stride, value, vl);
		else if constexpr (LMUL == M8) __riscv_vsse8_v_i8m8(base, stride, value, vl);
	}
	else if constexpr (std::is_same_v<T, uint32_t>) {
		if constexpr (LMUL == M1) __riscv_vsse32_v_u32m1(base, stride, value, vl);
		else if constexpr (LMUL == M2) __riscv_vsse32_v_u32m2(base, stride, value, vl);
		else if constexpr (LMUL == M4) __riscv_vsse32_v_u32m4(base, stride, value, vl);
		else if constexpr (LMUL == M8) __riscv_vsse32_v_u32m8(base, stride, value, vl);
	}
	else if constexpr (std::is_same_v<T, uint64_t>) {
		if constexpr (LMUL == M1) __riscv_vsse64_v_u64m1(base, stride, value, vl);
		else if constexpr (LMUL == M2) __riscv_vsse64_v_u64m2(base, stride, value, vl);
		else if constexpr (LMUL == M4) __riscv_vsse64_v_u64m4(base, stride, value, vl);
		else if constexpr (LMUL == M8) __riscv_vsse64_v_u64m8(base, stride, value, vl);
	}
	else if constexpr (std::is_same_v<T, uint16_t>) {
		if constexpr (LMUL == M1) __riscv_vsse16_v_u16m1(base, stride, value, vl);
		else if constexpr (LMUL == M2) __riscv_vsse16_v_u16m2(base, stride, value, vl);
		else if constexpr (LMUL == M4) __riscv_vsse16_v_u16m4(base, stride, value, vl);
		else if constexpr (LMUL == M8) __riscv_vsse16_v_u16m8(base, stride, value, vl);
	}
	else if constexpr (std::is_same_v<T, uint8_t>) {
		if constexpr (LMUL == M1) __riscv_vsse8_v_u8m1(base, stride, value, vl);
		else if constexpr (LMUL == M2) __riscv_vsse8_v_u8m2(base, stride, value, vl);
		else if constexpr (LMUL == M4) __riscv_vsse8_v_u8m4(base, stride, value, vl);
		else if constexpr (LMUL == M8) __riscv_vsse8_v_u8m8(base, stride, value, vl);
	}
}

/*
void __riscv_vsuxei8_v_f16mf4 (float16_t *base, vuint8mf8_t bindex, vfloat16mf4_t value, size_t vl);
void __riscv_vsuxei8_v_f16mf2 (float16_t *base, vuint8mf4_t bindex, vfloat16mf2_t value, size_t vl);
void __riscv_vsuxei8_v_f16m1 (float16_t *base, vuint8mf2_t bindex, vfloat16m1_t value, size_t vl);
void __riscv_vsuxei8_v_f16m2 (float16_t *base, vuint8m1_t bindex, vfloat16m2_t value, size_t vl);
void __riscv_vsuxei8_v_f16m4 (float16_t *base, vuint8m2_t bindex, vfloat16m4_t value, size_t vl);
void __riscv_vsuxei8_v_f16m8 (float16_t *base, vuint8m4_t bindex, vfloat16m8_t value, size_t vl);
void __riscv_vsuxei16_v_f16mf4 (float16_t *base, vuint16mf4_t bindex, vfloat16mf4_t value, size_t vl);
void __riscv_vsuxei16_v_f16mf2 (float16_t *base, vuint16mf2_t bindex, vfloat16mf2_t value, size_t vl);
void __riscv_vsuxei16_v_f16m1 (float16_t *base, vuint16m1_t bindex, vfloat16m1_t value, size_t vl);
void __riscv_vsuxei16_v_f16m2 (float16_t *base, vuint16m2_t bindex, vfloat16m2_t value, size_t vl);
void __riscv_vsuxei16_v_f16m4 (float16_t *base, vuint16m4_t bindex, vfloat16m4_t value, size_t vl);
void __riscv_vsuxei16_v_f16m8 (float16_t *base, vuint16m8_t bindex, vfloat16m8_t value, size_t vl);
void __riscv_vsuxei32_v_f16mf4 (float16_t *base, vuint32mf2_t bindex, vfloat16mf4_t value, size_t vl);
void __riscv_vsuxei32_v_f16mf2 (float16_t *base, vuint32m1_t bindex, vfloat16mf2_t value, size_t vl);
void __riscv_vsuxei32_v_f16m1 (float16_t *base, vuint32m2_t bindex, vfloat16m1_t value, size_t vl);
void __riscv_vsuxei32_v_f16m2 (float16_t *base, vuint32m4_t bindex, vfloat16m2_t value, size_t vl);
void __riscv_vsuxei32_v_f16m4 (float16_t *base, vuint32m8_t bindex, vfloat16m4_t value, size_t vl);
void __riscv_vsuxei64_v_f16mf4 (float16_t *base, vuint64m1_t bindex, vfloat16mf4_t value, size_t vl);
void __riscv_vsuxei64_v_f16mf2 (float16_t *base, vuint64m2_t bindex, vfloat16mf2_t value, size_t vl);
void __riscv_vsuxei64_v_f16m1 (float16_t *base, vuint64m4_t bindex, vfloat16m1_t value, size_t vl);
void __riscv_vsuxei64_v_f16m2 (float16_t *base, vuint64m8_t bindex, vfloat16m2_t value, size_t vl);
void __riscv_vsuxei8_v_f32mf2 (float32_t *base, vuint8mf8_t bindex, vfloat32mf2_t value, size_t vl);
void __riscv_vsuxei8_v_f32m1 (float32_t *base, vuint8mf4_t bindex, vfloat32m1_t value, size_t vl);
void __riscv_vsuxei8_v_f32m2 (float32_t *base, vuint8mf2_t bindex, vfloat32m2_t value, size_t vl);
void __riscv_vsuxei8_v_f32m4 (float32_t *base, vuint8m1_t bindex, vfloat32m4_t value, size_t vl);
void __riscv_vsuxei8_v_f32m8 (float32_t *base, vuint8m2_t bindex, vfloat32m8_t value, size_t vl);
void __riscv_vsuxei16_v_f32mf2 (float32_t *base, vuint16mf4_t bindex, vfloat32mf2_t value, size_t vl);
void __riscv_vsuxei16_v_f32m1 (float32_t *base, vuint16mf2_t bindex, vfloat32m1_t value, size_t vl);
void __riscv_vsuxei16_v_f32m2 (float32_t *base, vuint16m1_t bindex, vfloat32m2_t value, size_t vl);
void __riscv_vsuxei16_v_f32m4 (float32_t *base, vuint16m2_t bindex, vfloat32m4_t value, size_t vl);
void __riscv_vsuxei16_v_f32m8 (float32_t *base, vuint16m4_t bindex, vfloat32m8_t value, size_t vl);
void __riscv_vsuxei32_v_f32mf2 (float32_t *base, vuint32mf2_t bindex, vfloat32mf2_t value, size_t vl);
void __riscv_vsuxei32_v_f32m1 (float32_t *base, vuint32m1_t bindex, vfloat32m1_t value, size_t vl);
void __riscv_vsuxei32_v_f32m2 (float32_t *base, vuint32m2_t bindex, vfloat32m2_t value, size_t vl);
void __riscv_vsuxei32_v_f32m4 (float32_t *base, vuint32m4_t bindex, vfloat32m4_t value, size_t vl);
void __riscv_vsuxei32_v_f32m8 (float32_t *base, vuint32m8_t bindex, vfloat32m8_t value, size_t vl);
void __riscv_vsuxei64_v_f32mf2 (float32_t *base, vuint64m1_t bindex, vfloat32mf2_t value, size_t vl);
void __riscv_vsuxei64_v_f32m1 (float32_t *base, vuint64m2_t bindex, vfloat32m1_t value, size_t vl);
void __riscv_vsuxei64_v_f32m2 (float32_t *base, vuint64m4_t bindex, vfloat32m2_t value, size_t vl);
void __riscv_vsuxei64_v_f32m4 (float32_t *base, vuint64m8_t bindex, vfloat32m4_t value, size_t vl);
void __riscv_vsuxei8_v_f64m1 (float64_t *base, vuint8mf8_t bindex, vfloat64m1_t value, size_t vl);
void __riscv_vsuxei8_v_f64m2 (float64_t *base, vuint8mf4_t bindex, vfloat64m2_t value, size_t vl);
void __riscv_vsuxei8_v_f64m4 (float64_t *base, vuint8mf2_t bindex, vfloat64m4_t value, size_t vl);
void __riscv_vsuxei8_v_f64m8 (float64_t *base, vuint8m1_t bindex, vfloat64m8_t value, size_t vl);
void __riscv_vsuxei16_v_f64m1 (float64_t *base, vuint16mf4_t bindex, vfloat64m1_t value, size_t vl);
void __riscv_vsuxei16_v_f64m2 (float64_t *base, vuint16mf2_t bindex, vfloat64m2_t value, size_t vl);
void __riscv_vsuxei16_v_f64m4 (float64_t *base, vuint16m1_t bindex, vfloat64m4_t value, size_t vl);
void __riscv_vsuxei16_v_f64m8 (float64_t *base, vuint16m2_t bindex, vfloat64m8_t value, size_t vl);
void __riscv_vsuxei32_v_f64m1 (float64_t *base, vuint32mf2_t bindex, vfloat64m1_t value, size_t vl);
void __riscv_vsuxei32_v_f64m2 (float64_t *base, vuint32m1_t bindex, vfloat64m2_t value, size_t vl);
void __riscv_vsuxei32_v_f64m4 (float64_t *base, vuint32m2_t bindex, vfloat64m4_t value, size_t vl);
void __riscv_vsuxei32_v_f64m8 (float64_t *base, vuint32m4_t bindex, vfloat64m8_t value, size_t vl);
void __riscv_vsuxei64_v_f64m1 (float64_t *base, vuint64m1_t bindex, vfloat64m1_t value, size_t vl);
void __riscv_vsuxei64_v_f64m2 (float64_t *base, vuint64m2_t bindex, vfloat64m2_t value, size_t vl);
void __riscv_vsuxei64_v_f64m4 (float64_t *base, vuint64m4_t bindex, vfloat64m4_t value, size_t vl);
void __riscv_vsuxei64_v_f64m8 (float64_t *base, vuint64m8_t bindex, vfloat64m8_t value, size_t vl);

void __riscv_vsuxei8_v_i8mf8 (int8_t *base, vuint8mf8_t bindex, vint8mf8_t value, size_t vl);
void __riscv_vsuxei8_v_i8mf4 (int8_t *base, vuint8mf4_t bindex, vint8mf4_t value, size_t vl);
void __riscv_vsuxei8_v_i8mf2 (int8_t *base, vuint8mf2_t bindex, vint8mf2_t value, size_t vl);
void __riscv_vsuxei8_v_i8m1 (int8_t *base, vuint8m1_t bindex, vint8m1_t value, size_t vl);
void __riscv_vsuxei8_v_i8m2 (int8_t *base, vuint8m2_t bindex, vint8m2_t value, size_t vl);
void __riscv_vsuxei8_v_i8m4 (int8_t *base, vuint8m4_t bindex, vint8m4_t value, size_t vl);
void __riscv_vsuxei8_v_i8m8 (int8_t *base, vuint8m8_t bindex, vint8m8_t value, size_t vl);
void __riscv_vsuxei16_v_i8mf8 (int8_t *base, vuint16mf4_t bindex, vint8mf8_t value, size_t vl);
void __riscv_vsuxei16_v_i8mf4 (int8_t *base, vuint16mf2_t bindex, vint8mf4_t value, size_t vl);
void __riscv_vsuxei16_v_i8mf2 (int8_t *base, vuint16m1_t bindex, vint8mf2_t value, size_t vl);
void __riscv_vsuxei16_v_i8m1 (int8_t *base, vuint16m2_t bindex, vint8m1_t value, size_t vl);
void __riscv_vsuxei16_v_i8m2 (int8_t *base, vuint16m4_t bindex, vint8m2_t value, size_t vl);
void __riscv_vsuxei16_v_i8m4 (int8_t *base, vuint16m8_t bindex, vint8m4_t value, size_t vl);
void __riscv_vsuxei32_v_i8mf8 (int8_t *base, vuint32mf2_t bindex, vint8mf8_t value, size_t vl);
void __riscv_vsuxei32_v_i8mf4 (int8_t *base, vuint32m1_t bindex, vint8mf4_t value, size_t vl);
void __riscv_vsuxei32_v_i8mf2 (int8_t *base, vuint32m2_t bindex, vint8mf2_t value, size_t vl);
void __riscv_vsuxei32_v_i8m1 (int8_t *base, vuint32m4_t bindex, vint8m1_t value, size_t vl);
void __riscv_vsuxei32_v_i8m2 (int8_t *base, vuint32m8_t bindex, vint8m2_t value, size_t vl);
void __riscv_vsuxei64_v_i8mf8 (int8_t *base, vuint64m1_t bindex, vint8mf8_t value, size_t vl);
void __riscv_vsuxei64_v_i8mf4 (int8_t *base, vuint64m2_t bindex, vint8mf4_t value, size_t vl);
void __riscv_vsuxei64_v_i8mf2 (int8_t *base, vuint64m4_t bindex, vint8mf2_t value, size_t vl);
void __riscv_vsuxei64_v_i8m1 (int8_t *base, vuint64m8_t bindex, vint8m1_t value, size_t vl);
void __riscv_vsuxei8_v_i16mf4 (int16_t *base, vuint8mf8_t bindex, vint16mf4_t value, size_t vl);
void __riscv_vsuxei8_v_i16mf2 (int16_t *base, vuint8mf4_t bindex, vint16mf2_t value, size_t vl);
void __riscv_vsuxei8_v_i16m1 (int16_t *base, vuint8mf2_t bindex, vint16m1_t value, size_t vl);
void __riscv_vsuxei8_v_i16m2 (int16_t *base, vuint8m1_t bindex, vint16m2_t value, size_t vl);
void __riscv_vsuxei8_v_i16m4 (int16_t *base, vuint8m2_t bindex, vint16m4_t value, size_t vl);
void __riscv_vsuxei8_v_i16m8 (int16_t *base, vuint8m4_t bindex, vint16m8_t value, size_t vl);
void __riscv_vsuxei16_v_i16mf4 (int16_t *base, vuint16mf4_t bindex, vint16mf4_t value, size_t vl);
void __riscv_vsuxei16_v_i16mf2 (int16_t *base, vuint16mf2_t bindex, vint16mf2_t value, size_t vl);
void __riscv_vsuxei16_v_i16m1 (int16_t *base, vuint16m1_t bindex, vint16m1_t value, size_t vl);
void __riscv_vsuxei16_v_i16m2 (int16_t *base, vuint16m2_t bindex, vint16m2_t value, size_t vl);
void __riscv_vsuxei16_v_i16m4 (int16_t *base, vuint16m4_t bindex, vint16m4_t value, size_t vl);
void __riscv_vsuxei16_v_i16m8 (int16_t *base, vuint16m8_t bindex, vint16m8_t value, size_t vl);
void __riscv_vsuxei32_v_i16mf4 (int16_t *base, vuint32mf2_t bindex, vint16mf4_t value, size_t vl);
void __riscv_vsuxei32_v_i16mf2 (int16_t *base, vuint32m1_t bindex, vint16mf2_t value, size_t vl);
void __riscv_vsuxei32_v_i16m1 (int16_t *base, vuint32m2_t bindex, vint16m1_t value, size_t vl);
void __riscv_vsuxei32_v_i16m2 (int16_t *base, vuint32m4_t bindex, vint16m2_t value, size_t vl);
void __riscv_vsuxei32_v_i16m4 (int16_t *base, vuint32m8_t bindex, vint16m4_t value, size_t vl);
void __riscv_vsuxei64_v_i16mf4 (int16_t *base, vuint64m1_t bindex, vint16mf4_t value, size_t vl);
void __riscv_vsuxei64_v_i16mf2 (int16_t *base, vuint64m2_t bindex, vint16mf2_t value, size_t vl);
void __riscv_vsuxei64_v_i16m1 (int16_t *base, vuint64m4_t bindex, vint16m1_t value, size_t vl);
void __riscv_vsuxei64_v_i16m2 (int16_t *base, vuint64m8_t bindex, vint16m2_t value, size_t vl);
void __riscv_vsuxei8_v_i32mf2 (int32_t *base, vuint8mf8_t bindex, vint32mf2_t value, size_t vl);
void __riscv_vsuxei8_v_i32m1 (int32_t *base, vuint8mf4_t bindex, vint32m1_t value, size_t vl);
void __riscv_vsuxei8_v_i32m2 (int32_t *base, vuint8mf2_t bindex, vint32m2_t value, size_t vl);
void __riscv_vsuxei8_v_i32m4 (int32_t *base, vuint8m1_t bindex, vint32m4_t value, size_t vl);
void __riscv_vsuxei8_v_i32m8 (int32_t *base, vuint8m2_t bindex, vint32m8_t value, size_t vl);
void __riscv_vsuxei16_v_i32mf2 (int32_t *base, vuint16mf4_t bindex, vint32mf2_t value, size_t vl);
void __riscv_vsuxei16_v_i32m1 (int32_t *base, vuint16mf2_t bindex, vint32m1_t value, size_t vl);
void __riscv_vsuxei16_v_i32m2 (int32_t *base, vuint16m1_t bindex, vint32m2_t value, size_t vl);
void __riscv_vsuxei16_v_i32m4 (int32_t *base, vuint16m2_t bindex, vint32m4_t value, size_t vl);
void __riscv_vsuxei16_v_i32m8 (int32_t *base, vuint16m4_t bindex, vint32m8_t value, size_t vl);
void __riscv_vsuxei32_v_i32mf2 (int32_t *base, vuint32mf2_t bindex, vint32mf2_t value, size_t vl);
void __riscv_vsuxei32_v_i32m1 (int32_t *base, vuint32m1_t bindex, vint32m1_t value, size_t vl);
void __riscv_vsuxei32_v_i32m2 (int32_t *base, vuint32m2_t bindex, vint32m2_t value, size_t vl);
void __riscv_vsuxei32_v_i32m4 (int32_t *base, vuint32m4_t bindex, vint32m4_t value, size_t vl);
void __riscv_vsuxei32_v_i32m8 (int32_t *base, vuint32m8_t bindex, vint32m8_t value, size_t vl);
void __riscv_vsuxei64_v_i32mf2 (int32_t *base, vuint64m1_t bindex, vint32mf2_t value, size_t vl);
void __riscv_vsuxei64_v_i32m1 (int32_t *base, vuint64m2_t bindex, vint32m1_t value, size_t vl);
void __riscv_vsuxei64_v_i32m2 (int32_t *base, vuint64m4_t bindex, vint32m2_t value, size_t vl);
void __riscv_vsuxei64_v_i32m4 (int32_t *base, vuint64m8_t bindex, vint32m4_t value, size_t vl);
void __riscv_vsuxei8_v_i64m1 (int64_t *base, vuint8mf8_t bindex, vint64m1_t value, size_t vl);
void __riscv_vsuxei8_v_i64m2 (int64_t *base, vuint8mf4_t bindex, vint64m2_t value, size_t vl);
void __riscv_vsuxei8_v_i64m4 (int64_t *base, vuint8mf2_t bindex, vint64m4_t value, size_t vl);
void __riscv_vsuxei8_v_i64m8 (int64_t *base, vuint8m1_t bindex, vint64m8_t value, size_t vl);
void __riscv_vsuxei16_v_i64m1 (int64_t *base, vuint16mf4_t bindex, vint64m1_t value, size_t vl);
void __riscv_vsuxei16_v_i64m2 (int64_t *base, vuint16mf2_t bindex, vint64m2_t value, size_t vl);
void __riscv_vsuxei16_v_i64m4 (int64_t *base, vuint16m1_t bindex, vint64m4_t value, size_t vl);
void __riscv_vsuxei16_v_i64m8 (int64_t *base, vuint16m2_t bindex, vint64m8_t value, size_t vl);
void __riscv_vsuxei32_v_i64m1 (int64_t *base, vuint32mf2_t bindex, vint64m1_t value, size_t vl);
void __riscv_vsuxei32_v_i64m2 (int64_t *base, vuint32m1_t bindex, vint64m2_t value, size_t vl);
void __riscv_vsuxei32_v_i64m4 (int64_t *base, vuint32m2_t bindex, vint64m4_t value, size_t vl);
void __riscv_vsuxei32_v_i64m8 (int64_t *base, vuint32m4_t bindex, vint64m8_t value, size_t vl);
void __riscv_vsuxei64_v_i64m1 (int64_t *base, vuint64m1_t bindex, vint64m1_t value, size_t vl);
void __riscv_vsuxei64_v_i64m2 (int64_t *base, vuint64m2_t bindex, vint64m2_t value, size_t vl);
void __riscv_vsuxei64_v_i64m4 (int64_t *base, vuint64m4_t bindex, vint64m4_t value, size_t vl);
void __riscv_vsuxei64_v_i64m8 (int64_t *base, vuint64m8_t bindex, vint64m8_t value, size_t vl);

void __riscv_vsuxei8_v_u8mf8 (uint8_t *base, vuint8mf8_t bindex, vuint8mf8_t value, size_t vl);
void __riscv_vsuxei8_v_u8mf4 (uint8_t *base, vuint8mf4_t bindex, vuint8mf4_t value, size_t vl);
void __riscv_vsuxei8_v_u8mf2 (uint8_t *base, vuint8mf2_t bindex, vuint8mf2_t value, size_t vl);
void __riscv_vsuxei8_v_u8m1 (uint8_t *base, vuint8m1_t bindex, vuint8m1_t value, size_t vl);
void __riscv_vsuxei8_v_u8m2 (uint8_t *base, vuint8m2_t bindex, vuint8m2_t value, size_t vl);
void __riscv_vsuxei8_v_u8m4 (uint8_t *base, vuint8m4_t bindex, vuint8m4_t value, size_t vl);
void __riscv_vsuxei8_v_u8m8 (uint8_t *base, vuint8m8_t bindex, vuint8m8_t value, size_t vl);
void __riscv_vsuxei16_v_u8mf8 (uint8_t *base, vuint16mf4_t bindex, vuint8mf8_t value, size_t vl);
void __riscv_vsuxei16_v_u8mf4 (uint8_t *base, vuint16mf2_t bindex, vuint8mf4_t value, size_t vl);
void __riscv_vsuxei16_v_u8mf2 (uint8_t *base, vuint16m1_t bindex, vuint8mf2_t value, size_t vl);
void __riscv_vsuxei16_v_u8m1 (uint8_t *base, vuint16m2_t bindex, vuint8m1_t value, size_t vl);
void __riscv_vsuxei16_v_u8m2 (uint8_t *base, vuint16m4_t bindex, vuint8m2_t value, size_t vl);
void __riscv_vsuxei16_v_u8m4 (uint8_t *base, vuint16m8_t bindex, vuint8m4_t value, size_t vl);
void __riscv_vsuxei32_v_u8mf8 (uint8_t *base, vuint32mf2_t bindex, vuint8mf8_t value, size_t vl);
void __riscv_vsuxei32_v_u8mf4 (uint8_t *base, vuint32m1_t bindex, vuint8mf4_t value, size_t vl);
void __riscv_vsuxei32_v_u8mf2 (uint8_t *base, vuint32m2_t bindex, vuint8mf2_t value, size_t vl);
void __riscv_vsuxei32_v_u8m1 (uint8_t *base, vuint32m4_t bindex, vuint8m1_t value, size_t vl);
void __riscv_vsuxei32_v_u8m2 (uint8_t *base, vuint32m8_t bindex, vuint8m2_t value, size_t vl);
void __riscv_vsuxei64_v_u8mf8 (uint8_t *base, vuint64m1_t bindex, vuint8mf8_t value, size_t vl);
void __riscv_vsuxei64_v_u8mf4 (uint8_t *base, vuint64m2_t bindex, vuint8mf4_t value, size_t vl);
void __riscv_vsuxei64_v_u8mf2 (uint8_t *base, vuint64m4_t bindex, vuint8mf2_t value, size_t vl);
void __riscv_vsuxei64_v_u8m1 (uint8_t *base, vuint64m8_t bindex, vuint8m1_t value, size_t vl);
void __riscv_vsuxei8_v_u16mf4 (uint16_t *base, vuint8mf8_t bindex, vuint16mf4_t value, size_t vl);
void __riscv_vsuxei8_v_u16mf2 (uint16_t *base, vuint8mf4_t bindex, vuint16mf2_t value, size_t vl);
void __riscv_vsuxei8_v_u16m1 (uint16_t *base, vuint8mf2_t bindex, vuint16m1_t value, size_t vl);
void __riscv_vsuxei8_v_u16m2 (uint16_t *base, vuint8m1_t bindex, vuint16m2_t value, size_t vl);
void __riscv_vsuxei8_v_u16m4 (uint16_t *base, vuint8m2_t bindex, vuint16m4_t value, size_t vl);
void __riscv_vsuxei8_v_u16m8 (uint16_t *base, vuint8m4_t bindex, vuint16m8_t value, size_t vl);
void __riscv_vsuxei16_v_u16mf4 (uint16_t *base, vuint16mf4_t bindex, vuint16mf4_t value, size_t vl);
void __riscv_vsuxei16_v_u16mf2 (uint16_t *base, vuint16mf2_t bindex, vuint16mf2_t value, size_t vl);
void __riscv_vsuxei16_v_u16m1 (uint16_t *base, vuint16m1_t bindex, vuint16m1_t value, size_t vl);
void __riscv_vsuxei16_v_u16m2 (uint16_t *base, vuint16m2_t bindex, vuint16m2_t value, size_t vl);
void __riscv_vsuxei16_v_u16m4 (uint16_t *base, vuint16m4_t bindex, vuint16m4_t value, size_t vl);
void __riscv_vsuxei16_v_u16m8 (uint16_t *base, vuint16m8_t bindex, vuint16m8_t value, size_t vl);
void __riscv_vsuxei32_v_u16mf4 (uint16_t *base, vuint32mf2_t bindex, vuint16mf4_t value, size_t vl);
void __riscv_vsuxei32_v_u16mf2 (uint16_t *base, vuint32m1_t bindex, vuint16mf2_t value, size_t vl);
void __riscv_vsuxei32_v_u16m1 (uint16_t *base, vuint32m2_t bindex, vuint16m1_t value, size_t vl);
void __riscv_vsuxei32_v_u16m2 (uint16_t *base, vuint32m4_t bindex, vuint16m2_t value, size_t vl);
void __riscv_vsuxei32_v_u16m4 (uint16_t *base, vuint32m8_t bindex, vuint16m4_t value, size_t vl);
void __riscv_vsuxei64_v_u16mf4 (uint16_t *base, vuint64m1_t bindex, vuint16mf4_t value, size_t vl);
void __riscv_vsuxei64_v_u16mf2 (uint16_t *base, vuint64m2_t bindex, vuint16mf2_t value, size_t vl);
void __riscv_vsuxei64_v_u16m1 (uint16_t *base, vuint64m4_t bindex, vuint16m1_t value, size_t vl);
void __riscv_vsuxei64_v_u16m2 (uint16_t *base, vuint64m8_t bindex, vuint16m2_t value, size_t vl);
void __riscv_vsuxei8_v_u32mf2 (uint32_t *base, vuint8mf8_t bindex, vuint32mf2_t value, size_t vl);
void __riscv_vsuxei8_v_u32m1 (uint32_t *base, vuint8mf4_t bindex, vuint32m1_t value, size_t vl);
void __riscv_vsuxei8_v_u32m2 (uint32_t *base, vuint8mf2_t bindex, vuint32m2_t value, size_t vl);
void __riscv_vsuxei8_v_u32m4 (uint32_t *base, vuint8m1_t bindex, vuint32m4_t value, size_t vl);
void __riscv_vsuxei8_v_u32m8 (uint32_t *base, vuint8m2_t bindex, vuint32m8_t value, size_t vl);
void __riscv_vsuxei16_v_u32mf2 (uint32_t *base, vuint16mf4_t bindex, vuint32mf2_t value, size_t vl);
void __riscv_vsuxei16_v_u32m1 (uint32_t *base, vuint16mf2_t bindex, vuint32m1_t value, size_t vl);
void __riscv_vsuxei16_v_u32m2 (uint32_t *base, vuint16m1_t bindex, vuint32m2_t value, size_t vl);
void __riscv_vsuxei16_v_u32m4 (uint32_t *base, vuint16m2_t bindex, vuint32m4_t value, size_t vl);
void __riscv_vsuxei16_v_u32m8 (uint32_t *base, vuint16m4_t bindex, vuint32m8_t value, size_t vl);
void __riscv_vsuxei32_v_u32mf2 (uint32_t *base, vuint32mf2_t bindex, vuint32mf2_t value, size_t vl);
void __riscv_vsuxei32_v_u32m1 (uint32_t *base, vuint32m1_t bindex, vuint32m1_t value, size_t vl);
void __riscv_vsuxei32_v_u32m2 (uint32_t *base, vuint32m2_t bindex, vuint32m2_t value, size_t vl);
void __riscv_vsuxei32_v_u32m4 (uint32_t *base, vuint32m4_t bindex, vuint32m4_t value, size_t vl);
void __riscv_vsuxei32_v_u32m8 (uint32_t *base, vuint32m8_t bindex, vuint32m8_t value, size_t vl);
void __riscv_vsuxei64_v_u32mf2 (uint32_t *base, vuint64m1_t bindex, vuint32mf2_t value, size_t vl);
void __riscv_vsuxei64_v_u32m1 (uint32_t *base, vuint64m2_t bindex, vuint32m1_t value, size_t vl);
void __riscv_vsuxei64_v_u32m2 (uint32_t *base, vuint64m4_t bindex, vuint32m2_t value, size_t vl);
void __riscv_vsuxei64_v_u32m4 (uint32_t *base, vuint64m8_t bindex, vuint32m4_t value, size_t vl);
void __riscv_vsuxei8_v_u64m1 (uint64_t *base, vuint8mf8_t bindex, vuint64m1_t value, size_t vl);
void __riscv_vsuxei8_v_u64m2 (uint64_t *base, vuint8mf4_t bindex, vuint64m2_t value, size_t vl);
void __riscv_vsuxei8_v_u64m4 (uint64_t *base, vuint8mf2_t bindex, vuint64m4_t value, size_t vl);
void __riscv_vsuxei8_v_u64m8 (uint64_t *base, vuint8m1_t bindex, vuint64m8_t value, size_t vl);
void __riscv_vsuxei16_v_u64m1 (uint64_t *base, vuint16mf4_t bindex, vuint64m1_t value, size_t vl);
void __riscv_vsuxei16_v_u64m2 (uint64_t *base, vuint16mf2_t bindex, vuint64m2_t value, size_t vl);
void __riscv_vsuxei16_v_u64m4 (uint64_t *base, vuint16m1_t bindex, vuint64m4_t value, size_t vl);
void __riscv_vsuxei16_v_u64m8 (uint64_t *base, vuint16m2_t bindex, vuint64m8_t value, size_t vl);
void __riscv_vsuxei32_v_u64m1 (uint64_t *base, vuint32mf2_t bindex, vuint64m1_t value, size_t vl);
void __riscv_vsuxei32_v_u64m2 (uint64_t *base, vuint32m1_t bindex, vuint64m2_t value, size_t vl);
void __riscv_vsuxei32_v_u64m4 (uint64_t *base, vuint32m2_t bindex, vuint64m4_t value, size_t vl);
void __riscv_vsuxei32_v_u64m8 (uint64_t *base, vuint32m4_t bindex, vuint64m8_t value, size_t vl);
void __riscv_vsuxei64_v_u64m1 (uint64_t *base, vuint64m1_t bindex, vuint64m1_t value, size_t vl);
void __riscv_vsuxei64_v_u64m2 (uint64_t *base, vuint64m2_t bindex, vuint64m2_t value, size_t vl);
void __riscv_vsuxei64_v_u64m4 (uint64_t *base, vuint64m4_t bindex, vuint64m4_t value, size_t vl);
void __riscv_vsuxei64_v_u64m8 (uint64_t *base, vuint64m8_t bindex, vuint64m8_t value, size_t vl);
*/

// Vector Indexed Store with 8-bit indices
template<typename T, int LMUL, typename VecType, typename IndexType>
inline void VECTOR_INDEXED_STORE_8(T* base, IndexType bindex, VecType value, size_t vl) {
    if constexpr (std::is_same_v<T, _Float16>) {
        if constexpr (LMUL == MF4) __riscv_vsuxei8_v_f16mf4(base, bindex, value, vl);
        else if constexpr (LMUL == MF2) __riscv_vsuxei8_v_f16mf2(base, bindex, value, vl);
        else if constexpr (LMUL == M1) __riscv_vsuxei8_v_f16m1(base, bindex, value, vl);
        else if constexpr (LMUL == M2) __riscv_vsuxei8_v_f16m2(base, bindex, value, vl);
        else if constexpr (LMUL == M4) __riscv_vsuxei8_v_f16m4(base, bindex, value, vl);
        else if constexpr (LMUL == M8) __riscv_vsuxei8_v_f16m8(base, bindex, value, vl);
    }
    else if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) __riscv_vsuxei8_v_f32mf2(base, bindex, value, vl);
        else if constexpr (LMUL == M1) __riscv_vsuxei8_v_f32m1(base, bindex, value, vl);
        else if constexpr (LMUL == M2) __riscv_vsuxei8_v_f32m2(base, bindex, value, vl);
        else if constexpr (LMUL == M4) __riscv_vsuxei8_v_f32m4(base, bindex, value, vl);
        else if constexpr (LMUL == M8) __riscv_vsuxei8_v_f32m8(base, bindex, value, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) __riscv_vsuxei8_v_f64m1(base, bindex, value, vl);
        else if constexpr (LMUL == M2) __riscv_vsuxei8_v_f64m2(base, bindex, value, vl);
        else if constexpr (LMUL == M4) __riscv_vsuxei8_v_f64m4(base, bindex, value, vl);
        else if constexpr (LMUL == M8) __riscv_vsuxei8_v_f64m8(base, bindex, value, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) __riscv_vsuxei8_v_i8mf8(base, bindex, value, vl);
        else if constexpr (LMUL == MF4) __riscv_vsuxei8_v_i8mf4(base, bindex, value, vl);
        else if constexpr (LMUL == MF2) __riscv_vsuxei8_v_i8mf2(base, bindex, value, vl);
        else if constexpr (LMUL == M1) __riscv_vsuxei8_v_i8m1(base, bindex, value, vl);
        else if constexpr (LMUL == M2) __riscv_vsuxei8_v_i8m2(base, bindex, value, vl);
        else if constexpr (LMUL == M4) __riscv_vsuxei8_v_i8m4(base, bindex, value, vl);
        else if constexpr (LMUL == M8) __riscv_vsuxei8_v_i8m8(base, bindex, value, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) __riscv_vsuxei8_v_i16mf4(base, bindex, value, vl);
        else if constexpr (LMUL == MF2) __riscv_vsuxei8_v_i16mf2(base, bindex, value, vl);
        else if constexpr (LMUL == M1) __riscv_vsuxei8_v_i16m1(base, bindex, value, vl);
        else if constexpr (LMUL == M2) __riscv_vsuxei8_v_i16m2(base, bindex, value, vl);
        else if constexpr (LMUL == M4) __riscv_vsuxei8_v_i16m4(base, bindex, value, vl);
        else if constexpr (LMUL == M8) __riscv_vsuxei8_v_i16m8(base, bindex, value, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) __riscv_vsuxei8_v_i32mf2(base, bindex, value, vl);
        else if constexpr (LMUL == M1) __riscv_vsuxei8_v_i32m1(base, bindex, value, vl);
        else if constexpr (LMUL == M2) __riscv_vsuxei8_v_i32m2(base, bindex, value, vl);
        else if constexpr (LMUL == M4) __riscv_vsuxei8_v_i32m4(base, bindex, value, vl);
        else if constexpr (LMUL == M8) __riscv_vsuxei8_v_i32m8(base, bindex, value, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) __riscv_vsuxei8_v_i64m1(base, bindex, value, vl);
        else if constexpr (LMUL == M2) __riscv_vsuxei8_v_i64m2(base, bindex, value, vl);
        else if constexpr (LMUL == M4) __riscv_vsuxei8_v_i64m4(base, bindex, value, vl);
        else if constexpr (LMUL == M8) __riscv_vsuxei8_v_i64m8(base, bindex, value, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) __riscv_vsuxei8_v_u8mf8(base, bindex, value, vl);
        else if constexpr (LMUL == MF4) __riscv_vsuxei8_v_u8mf4(base, bindex, value, vl);
        else if constexpr (LMUL == MF2) __riscv_vsuxei8_v_u8mf2(base, bindex, value, vl);
        else if constexpr (LMUL == M1) __riscv_vsuxei8_v_u8m1(base, bindex, value, vl);
        else if constexpr (LMUL == M2) __riscv_vsuxei8_v_u8m2(base, bindex, value, vl);
        else if constexpr (LMUL == M4) __riscv_vsuxei8_v_u8m4(base, bindex, value, vl);
        else if constexpr (LMUL == M8) __riscv_vsuxei8_v_u8m8(base, bindex, value, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) __riscv_vsuxei8_v_u16mf4(base, bindex, value, vl);
        else if constexpr (LMUL == MF2) __riscv_vsuxei8_v_u16mf2(base, bindex, value, vl);
        else if constexpr (LMUL == M1) __riscv_vsuxei8_v_u16m1(base, bindex, value, vl);
        else if constexpr (LMUL == M2) __riscv_vsuxei8_v_u16m2(base, bindex, value, vl);
        else if constexpr (LMUL == M4) __riscv_vsuxei8_v_u16m4(base, bindex, value, vl);
        else if constexpr (LMUL == M8) __riscv_vsuxei8_v_u16m8(base, bindex, value, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) __riscv_vsuxei8_v_u32mf2(base, bindex, value, vl);
        else if constexpr (LMUL == M1) __riscv_vsuxei8_v_u32m1(base, bindex, value, vl);
        else if constexpr (LMUL == M2) __riscv_vsuxei8_v_u32m2(base, bindex, value, vl);
        else if constexpr (LMUL == M4) __riscv_vsuxei8_v_u32m4(base, bindex, value, vl);
        else if constexpr (LMUL == M8) __riscv_vsuxei8_v_u32m8(base, bindex, value, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) __riscv_vsuxei8_v_u64m1(base, bindex, value, vl);
        else if constexpr (LMUL == M2) __riscv_vsuxei8_v_u64m2(base, bindex, value, vl);
        else if constexpr (LMUL == M4) __riscv_vsuxei8_v_u64m4(base, bindex, value, vl);
        else if constexpr (LMUL == M8) __riscv_vsuxei8_v_u64m8(base, bindex, value, vl);
    }
}

// Vector Indexed Store with 16-bit indices
template<typename T, int LMUL, typename VecType, typename IndexType>
inline void VECTOR_INDEXED_STORE_16(T* base, IndexType bindex, VecType value, size_t vl) {
    if constexpr (std::is_same_v<T, _Float16>) {
        if constexpr (LMUL == MF4) __riscv_vsuxei16_v_f16mf4(base, bindex, value, vl);
        else if constexpr (LMUL == MF2) __riscv_vsuxei16_v_f16mf2(base, bindex, value, vl);
        else if constexpr (LMUL == M1) __riscv_vsuxei16_v_f16m1(base, bindex, value, vl);
        else if constexpr (LMUL == M2) __riscv_vsuxei16_v_f16m2(base, bindex, value, vl);
        else if constexpr (LMUL == M4) __riscv_vsuxei16_v_f16m4(base, bindex, value, vl);
        else if constexpr (LMUL == M8) __riscv_vsuxei16_v_f16m8(base, bindex, value, vl);
    }
    else if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) __riscv_vsuxei16_v_f32mf2(base, bindex, value, vl);
        else if constexpr (LMUL == M1) __riscv_vsuxei16_v_f32m1(base, bindex, value, vl);
        else if constexpr (LMUL == M2) __riscv_vsuxei16_v_f32m2(base, bindex, value, vl);
        else if constexpr (LMUL == M4) __riscv_vsuxei16_v_f32m4(base, bindex, value, vl);
        else if constexpr (LMUL == M8) __riscv_vsuxei16_v_f32m8(base, bindex, value, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) __riscv_vsuxei16_v_f64m1(base, bindex, value, vl);
        else if constexpr (LMUL == M2) __riscv_vsuxei16_v_f64m2(base, bindex, value, vl);
        else if constexpr (LMUL == M4) __riscv_vsuxei16_v_f64m4(base, bindex, value, vl);
        else if constexpr (LMUL == M8) __riscv_vsuxei16_v_f64m8(base, bindex, value, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) __riscv_vsuxei16_v_i8mf8(base, bindex, value, vl);
        else if constexpr (LMUL == MF4) __riscv_vsuxei16_v_i8mf4(base, bindex, value, vl);
        else if constexpr (LMUL == MF2) __riscv_vsuxei16_v_i8mf2(base, bindex, value, vl);
        else if constexpr (LMUL == M1) __riscv_vsuxei16_v_i8m1(base, bindex, value, vl);
        else if constexpr (LMUL == M2) __riscv_vsuxei16_v_i8m2(base, bindex, value, vl);
        else if constexpr (LMUL == M4) __riscv_vsuxei16_v_i8m4(base, bindex, value, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) __riscv_vsuxei16_v_i16mf4(base, bindex, value, vl);
        else if constexpr (LMUL == MF2) __riscv_vsuxei16_v_i16mf2(base, bindex, value, vl);
        else if constexpr (LMUL == M1) __riscv_vsuxei16_v_i16m1(base, bindex, value, vl);
        else if constexpr (LMUL == M2) __riscv_vsuxei16_v_i16m2(base, bindex, value, vl);
        else if constexpr (LMUL == M4) __riscv_vsuxei16_v_i16m4(base, bindex, value, vl);
        else if constexpr (LMUL == M8) __riscv_vsuxei16_v_i16m8(base, bindex, value, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) __riscv_vsuxei16_v_i32mf2(base, bindex, value, vl);
        else if constexpr (LMUL == M1) __riscv_vsuxei16_v_i32m1(base, bindex, value, vl);
        else if constexpr (LMUL == M2) __riscv_vsuxei16_v_i32m2(base, bindex, value, vl);
        else if constexpr (LMUL == M4) __riscv_vsuxei16_v_i32m4(base, bindex, value, vl);
        else if constexpr (LMUL == M8) __riscv_vsuxei16_v_i32m8(base, bindex, value, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) __riscv_vsuxei16_v_i64m1(base, bindex, value, vl);
        else if constexpr (LMUL == M2) __riscv_vsuxei16_v_i64m2(base, bindex, value, vl);
        else if constexpr (LMUL == M4) __riscv_vsuxei16_v_i64m4(base, bindex, value, vl);
        else if constexpr (LMUL == M8) __riscv_vsuxei16_v_i64m8(base, bindex, value, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) __riscv_vsuxei16_v_u8mf8(base, bindex, value, vl);
        else if constexpr (LMUL == MF4) __riscv_vsuxei16_v_u8mf4(base, bindex, value, vl);
        else if constexpr (LMUL == MF2) __riscv_vsuxei16_v_u8mf2(base, bindex, value, vl);
        else if constexpr (LMUL == M1) __riscv_vsuxei16_v_u8m1(base, bindex, value, vl);
        else if constexpr (LMUL == M2) __riscv_vsuxei16_v_u8m2(base, bindex, value, vl);
        else if constexpr (LMUL == M4) __riscv_vsuxei16_v_u8m4(base, bindex, value, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) __riscv_vsuxei16_v_u16mf4(base, bindex, value, vl);
        else if constexpr (LMUL == MF2) __riscv_vsuxei16_v_u16mf2(base, bindex, value, vl);
        else if constexpr (LMUL == M1) __riscv_vsuxei16_v_u16m1(base, bindex, value, vl);
        else if constexpr (LMUL == M2) __riscv_vsuxei16_v_u16m2(base, bindex, value, vl);
        else if constexpr (LMUL == M4) __riscv_vsuxei16_v_u16m4(base, bindex, value, vl);
        else if constexpr (LMUL == M8) __riscv_vsuxei16_v_u16m8(base, bindex, value, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) __riscv_vsuxei16_v_u32mf2(base, bindex, value, vl);
			else if constexpr (LMUL == M1)__riscv_vsuxei16_v_u32m1(base, bindex, value, vl);
			else if constexpr (LMUL == M2) __riscv_vsuxei16_v_u32m2(base, bindex, value, vl);
			else if constexpr (LMUL == M4) __riscv_vsuxei16_v_u32m4(base, bindex, value, vl);
			else if constexpr (LMUL == M8) __riscv_vsuxei16_v_u32m8(base, bindex, value, vl);
			}
			else if constexpr (std::is_same_v<T, uint64_t>) {
			if constexpr (LMUL == M1) __riscv_vsuxei16_v_u64m1(base, bindex, value, vl);
			else if constexpr (LMUL == M2) __riscv_vsuxei16_v_u64m2(base, bindex, value, vl);
			else if constexpr (LMUL == M4) __riscv_vsuxei16_v_u64m4(base, bindex, value, vl);
			else if constexpr (LMUL == M8) __riscv_vsuxei16_v_u64m8(base, bindex, value, vl);
			}
		}

// Vector Indexed Store with 32-bit indices
template<typename T, int LMUL, typename VecType, typename IndexType>
inline void VECTOR_INDEXED_STORE_32(T* base, IndexType bindex, VecType value, size_t vl) {
	if constexpr (std::is_same_v<T, _Float16>) {
	if constexpr (LMUL == MF4) __riscv_vsuxei32_v_f16mf4(base, bindex, value, vl);
	else if constexpr (LMUL == MF2) __riscv_vsuxei32_v_f16mf2(base, bindex, value, vl);
	else if constexpr (LMUL == M1) __riscv_vsuxei32_v_f16m1(base, bindex, value, vl);
	else if constexpr (LMUL == M2) __riscv_vsuxei32_v_f16m2(base, bindex, value, vl);
	else if constexpr (LMUL == M4) __riscv_vsuxei32_v_f16m4(base, bindex, value, vl);
	}
	else if constexpr (std::is_same_v<T, float>) {
	if constexpr (LMUL == MF2) __riscv_vsuxei32_v_f32mf2(base, bindex, value, vl);
	else if constexpr (LMUL == M1) __riscv_vsuxei32_v_f32m1(base, bindex, value, vl);
	else if constexpr (LMUL == M2) __riscv_vsuxei32_v_f32m2(base, bindex, value, vl);
	else if constexpr (LMUL == M4) __riscv_vsuxei32_v_f32m4(base, bindex, value, vl);
	else if constexpr (LMUL == M8) __riscv_vsuxei32_v_f32m8(base, bindex, value, vl);
	}
	else if constexpr (std::is_same_v<T, double>) {
	if constexpr (LMUL == M1) __riscv_vsuxei32_v_f64m1(base, bindex, value, vl);
	else if constexpr (LMUL == M2) __riscv_vsuxei32_v_f64m2(base, bindex, value, vl);
	else if constexpr (LMUL == M4) __riscv_vsuxei32_v_f64m4(base, bindex, value, vl);
	else if constexpr (LMUL == M8) __riscv_vsuxei32_v_f64m8(base, bindex, value, vl);
	}
	else if constexpr (std::is_same_v<T, int8_t>) {
	if constexpr (LMUL == MF8) __riscv_vsuxei32_v_i8mf8(base, bindex, value, vl);
	else if constexpr (LMUL == MF4) __riscv_vsuxei32_v_i8mf4(base, bindex, value, vl);
	else if constexpr (LMUL == MF2) __riscv_vsuxei32_v_i8mf2(base, bindex, value, vl);
	else if constexpr (LMUL == M1) __riscv_vsuxei32_v_i8m1(base, bindex, value, vl);
	else if constexpr (LMUL == M2) __riscv_vsuxei32_v_i8m2(base, bindex, value, vl);
	}
	else if constexpr (std::is_same_v<T, int16_t>) {
	if constexpr (LMUL == MF4) __riscv_vsuxei32_v_i16mf4(base, bindex, value, vl);
	else if constexpr (LMUL == MF2) __riscv_vsuxei32_v_i16mf2(base, bindex, value, vl);
	else if constexpr (LMUL == M1) __riscv_vsuxei32_v_i16m1(base, bindex, value, vl);
	else if constexpr (LMUL == M2) __riscv_vsuxei32_v_i16m2(base, bindex, value, vl);
	else if constexpr (LMUL == M4) __riscv_vsuxei32_v_i16m4(base, bindex, value, vl);
	}
	else if constexpr (std::is_same_v<T, int32_t>) {
	if constexpr (LMUL == MF2) __riscv_vsuxei32_v_i32mf2(base, bindex, value, vl);
	else if constexpr (LMUL == M1) __riscv_vsuxei32_v_i32m1(base, bindex, value, vl);
	else if constexpr (LMUL == M2) __riscv_vsuxei32_v_i32m2(base, bindex, value, vl);
	else if constexpr (LMUL == M4) __riscv_vsuxei32_v_i32m4(base, bindex, value, vl);
	else if constexpr (LMUL == M8) __riscv_vsuxei32_v_i32m8(base, bindex, value, vl);
	}
	else if constexpr (std::is_same_v<T, int64_t>) {
	if constexpr (LMUL == M1) __riscv_vsuxei32_v_i64m1(base, bindex, value, vl);
	else if constexpr (LMUL == M2) __riscv_vsuxei32_v_i64m2(base, bindex, value, vl);
	else if constexpr (LMUL == M4) __riscv_vsuxei32_v_i64m4(base, bindex, value, vl);
	else if constexpr (LMUL == M8) __riscv_vsuxei32_v_i64m8(base, bindex, value, vl);
	}
	else if constexpr (std::is_same_v<T, uint8_t>) {
	if constexpr (LMUL == MF8) __riscv_vsuxei32_v_u8mf8(base, bindex, value, vl);
	else if constexpr (LMUL == MF4) __riscv_vsuxei32_v_u8mf4(base, bindex, value, vl);
	else if constexpr (LMUL == MF2) __riscv_vsuxei32_v_u8mf2(base, bindex, value, vl);
	else if constexpr (LMUL == M1) __riscv_vsuxei32_v_u8m1(base, bindex, value, vl);
	else if constexpr (LMUL == M2) __riscv_vsuxei32_v_u8m2(base, bindex, value, vl);
	}
	else if constexpr (std::is_same_v<T, uint16_t>) {
	if constexpr (LMUL == MF4) __riscv_vsuxei32_v_u16mf4(base, bindex, value, vl);
	else if constexpr (LMUL == MF2) __riscv_vsuxei32_v_u16mf2(base, bindex, value, vl);
	else if constexpr (LMUL == M1) __riscv_vsuxei32_v_u16m1(base, bindex, value, vl);
	else if constexpr (LMUL == M2) __riscv_vsuxei32_v_u16m2(base, bindex, value, vl);
	else if constexpr (LMUL == M4) __riscv_vsuxei32_v_u16m4(base, bindex, value, vl);
	}
	else if constexpr (std::is_same_v<T, uint32_t>) {
	if constexpr (LMUL == MF2) __riscv_vsuxei32_v_u32mf2(base, bindex, value, vl);
	else if constexpr (LMUL == M1) __riscv_vsuxei32_v_u32m1(base, bindex, value, vl);
	else if constexpr (LMUL == M2) __riscv_vsuxei32_v_u32m2(base, bindex, value, vl);
	else if constexpr (LMUL == M4) __riscv_vsuxei32_v_u32m4(base, bindex, value, vl);
	else if constexpr (LMUL == M8) __riscv_vsuxei32_v_u32m8(base, bindex, value, vl);
	}
	else if constexpr (std::is_same_v<T, uint64_t>) {
	if constexpr (LMUL == M1) __riscv_vsuxei32_v_u64m1(base, bindex, value, vl);
	else if constexpr (LMUL == M2) __riscv_vsuxei32_v_u64m2(base, bindex, value, vl);
	else if constexpr (LMUL == M4) __riscv_vsuxei32_v_u64m4(base, bindex, value, vl);
	else if constexpr (LMUL == M8) __riscv_vsuxei32_v_u64m8(base, bindex, value, vl);
	}
}

// Vector Indexed Store with 64-bit indices
template<typename T, int LMUL, typename VecType, typename IndexType>
inline void VECTOR_INDEXED_STORE_64(T* base, IndexType bindex, VecType value, size_t vl) {
	if constexpr (std::is_same_v<T, _Float16>) {
	if constexpr (LMUL == MF4) __riscv_vsuxei64_v_f16mf4(base, bindex, value, vl);
	else if constexpr (LMUL == MF2) __riscv_vsuxei64_v_f16mf2(base, bindex, value, vl);
	else if constexpr (LMUL == M1) __riscv_vsuxei64_v_f16m1(base, bindex, value, vl);
	else if constexpr (LMUL == M2) __riscv_vsuxei64_v_f16m2(base, bindex, value, vl);
	}
	else if constexpr (std::is_same_v<T, float>) {
	if constexpr (LMUL == MF2) __riscv_vsuxei64_v_f32mf2(base, bindex, value, vl);
	else if constexpr (LMUL == M1) __riscv_vsuxei64_v_f32m1(base, bindex, value, vl);
	else if constexpr (LMUL == M2) __riscv_vsuxei64_v_f32m2(base, bindex, value, vl);
	else if constexpr (LMUL == M4) __riscv_vsuxei64_v_f32m4(base, bindex, value, vl);
	}
	else if constexpr (std::is_same_v<T, double>) {
	if constexpr (LMUL == M1) __riscv_vsuxei64_v_f64m1(base, bindex, value, vl);
	else if constexpr (LMUL == M2) __riscv_vsuxei64_v_f64m2(base, bindex, value, vl);
	else if constexpr (LMUL == M4) __riscv_vsuxei64_v_f64m4(base, bindex, value, vl);
	else if constexpr (LMUL == M8) __riscv_vsuxei64_v_f64m8(base, bindex, value, vl);
	}
	else if constexpr (std::is_same_v<T, int8_t>) {
	if constexpr (LMUL == MF8) __riscv_vsuxei64_v_i8mf8(base, bindex, value, vl);
	else if constexpr (LMUL == MF4) __riscv_vsuxei64_v_i8mf4(base, bindex, value, vl);
	else if constexpr (LMUL == MF2) __riscv_vsuxei64_v_i8mf2(base, bindex, value, vl);
	else if constexpr (LMUL == M1) __riscv_vsuxei64_v_i8m1(base, bindex, value, vl);
	}
	else if constexpr (std::is_same_v<T, int16_t>) {
	if constexpr (LMUL == MF4) __riscv_vsuxei64_v_i16mf4(base, bindex, value, vl);
	else if constexpr (LMUL == MF2) __riscv_vsuxei64_v_i16mf2(base, bindex, value, vl);
	else if constexpr (LMUL == M1) __riscv_vsuxei64_v_i16m1(base, bindex, value, vl);
	else if constexpr (LMUL == M2) __riscv_vsuxei64_v_i16m2(base, bindex, value, vl);
	}
	else if constexpr (std::is_same_v<T, int32_t>) {
	if constexpr (LMUL == MF2) __riscv_vsuxei64_v_i32mf2(base, bindex, value, vl);
	else if constexpr (LMUL == M1) __riscv_vsuxei64_v_i32m1(base, bindex, value, vl);
	else if constexpr (LMUL == M2) __riscv_vsuxei64_v_i32m2(base, bindex, value, vl);
	else if constexpr (LMUL == M4) __riscv_vsuxei64_v_i32m4(base, bindex, value, vl);
	}
	else if constexpr (std::is_same_v<T, int64_t>) {
	if constexpr (LMUL == M1) __riscv_vsuxei64_v_i64m1(base, bindex, value, vl);
	else if constexpr (LMUL == M2) __riscv_vsuxei64_v_i64m2(base, bindex, value, vl);
	else if constexpr (LMUL == M4) __riscv_vsuxei64_v_i64m4(base, bindex, value, vl);
	else if constexpr (LMUL == M8) __riscv_vsuxei64_v_i64m8(base, bindex, value, vl);
	}
	else if constexpr (std::is_same_v<T, uint8_t>) {
	if constexpr (LMUL == MF8) __riscv_vsuxei64_v_u8mf8(base, bindex, value, vl);
	else if constexpr (LMUL == MF4) __riscv_vsuxei64_v_u8mf4(base, bindex, value, vl);
	else if constexpr (LMUL == MF2) __riscv_vsuxei64_v_u8mf2(base, bindex, value, vl);
	else if constexpr (LMUL == M1) __riscv_vsuxei64_v_u8m1(base, bindex, value, vl);
	}
	else if constexpr (std::is_same_v<T, uint16_t>) {
	if constexpr (LMUL == MF4) __riscv_vsuxei64_v_u16mf4(base, bindex, value, vl);
	else if constexpr (LMUL == MF2) __riscv_vsuxei64_v_u16mf2(base, bindex, value, vl);
	else if constexpr (LMUL == M1) __riscv_vsuxei64_v_u16m1(base, bindex, value, vl);
	else if constexpr (LMUL == M2) __riscv_vsuxei64_v_u16m2(base, bindex, value, vl);
	}
	else if constexpr (std::is_same_v<T, uint32_t>) {
	if constexpr (LMUL == MF2) __riscv_vsuxei64_v_u32mf2(base, bindex, value, vl);
	else if constexpr (LMUL == M1) __riscv_vsuxei64_v_u32m1(base, bindex, value, vl);
	else if constexpr (LMUL == M2) __riscv_vsuxei64_v_u32m2(base, bindex, value, vl);
	else if constexpr (LMUL == M4) __riscv_vsuxei64_v_u32m4(base, bindex, value, vl);
	}
	else if constexpr (std::is_same_v<T, uint64_t>) {
	if constexpr (LMUL == M1) __riscv_vsuxei64_v_u64m1(base, bindex, value, vl);
	else if constexpr (LMUL == M2) __riscv_vsuxei64_v_u64m2(base, bindex, value, vl);
	else if constexpr (LMUL == M4) __riscv_vsuxei64_v_u64m4(base, bindex, value, vl);
	else if constexpr (LMUL == M8) __riscv_vsuxei64_v_u64m8(base, bindex, value, vl);
	}
}

template<typename T, int LMUL, typename VecType, typename IndexType>
inline void VECTOR_INDEXED_STORE(T* base, IndexType bindex, VecType value, size_t vl) {
	constexpr size_t index_width = sizeof(typename IndexType::element_type);

	if constexpr (index_width == 1) { // 8-bit indices
		VECTOR_INDEXED_STORE_8<T, LMUL, VecType, IndexType>(base, bindex, value, vl);
	}
	else if constexpr (index_width == 2) { // 16-bit indices
		VECTOR_INDEXED_STORE_16<T, LMUL, VecType, IndexType>(base, bindex, value, vl);
	}
	else if constexpr (index_width == 4) { // 32-bit indices
		VECTOR_INDEXED_STORE_32<T, LMUL, VecType, IndexType>(base, bindex, value, vl);
	}
	else if constexpr (index_width == 8) { // 64-bit indices
		VECTOR_INDEXED_STORE_64<T, LMUL, VecType, IndexType>(base, bindex, value, vl);
	}
}

#endif // RVV_VECTOR_STORE_HPP