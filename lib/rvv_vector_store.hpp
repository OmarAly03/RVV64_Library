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
	if constexpr (std::is_same_v<T, float>) {
		if constexpr (LMUL == M1) __riscv_vse32_v_f32m1(base, value, vl);
		else if constexpr (LMUL == M2) __riscv_vse32_v_f32m2(base, value, vl);
		else if constexpr (LMUL == M4) __riscv_vse32_v_f32m4(base, value, vl);
		else if constexpr (LMUL == M8) __riscv_vse32_v_f32m8(base, value, vl);
	}
	else if constexpr (std::is_same_v<T, int32_t>) {
		if constexpr (LMUL == M1) __riscv_vse32_v_i32m1(base, value, vl);
		else if constexpr (LMUL == M2) __riscv_vse32_v_i32m2(base, value, vl);
		else if constexpr (LMUL == M4) __riscv_vse32_v_i32m4(base, value, vl);
		else if constexpr (LMUL == M8) __riscv_vse32_v_i32m8(base, value, vl);
	}
	else if constexpr (std::is_same_v<T, double>) {
		if constexpr (LMUL == M1) __riscv_vse64_v_f64m1(base, value, vl);
		else if constexpr (LMUL == M2) __riscv_vse64_v_f64m2(base, value, vl);
		else if constexpr (LMUL == M4) __riscv_vse64_v_f64m4(base, value, vl);
		else if constexpr (LMUL == M8) __riscv_vse64_v_f64m8(base, value, vl);
	}
	else if constexpr (std::is_same_v<T, int16_t>) {
		if constexpr (LMUL == M1) __riscv_vse16_v_i16m1(base, value, vl);
		else if constexpr (LMUL == M2) __riscv_vse16_v_i16m2(base, value, vl);
		else if constexpr (LMUL == M4) __riscv_vse16_v_i16m4(base, value, vl);
		else if constexpr (LMUL == M8) __riscv_vse16_v_i16m8(base, value, vl);
	}
	else if constexpr (std::is_same_v<T, int8_t>) {
		if constexpr (LMUL == M1) __riscv_vse8_v_i8m1(base, value, vl);
		else if constexpr (LMUL == M2) __riscv_vse8_v_i8m2(base, value, vl);
		else if constexpr (LMUL == M4) __riscv_vse8_v_i8m4(base, value, vl);
		else if constexpr (LMUL == M8) __riscv_vse8_v_i8m8(base, value, vl);
	}
}

#endif // RVV_VECTOR_STORE_HPP