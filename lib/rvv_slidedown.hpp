#ifndef RVV_SLIDEDOWN_HPP
#define RVV_SLIDEDOWN_HPP

#include <cstddef>
#include <riscv_vector.h>
#include <type_traits>

template<typename T, int LMUL, typename VecType>
inline auto VECTOR_SLIDEDOWN(VecType src, size_t offset, size_t vl) {
	if constexpr (std::is_same_v<T, _Float16>) {
        if constexpr (LMUL == MF4) return __riscv_vslidedown_vx_f16mf4(src, offset, vl);
		else if constexpr (LMUL == MF2) return __riscv_vslidedown_vx_f16mf2(src, offset, vl);
        else if constexpr (LMUL == M1) return __riscv_vslidedown_vx_f16m1(src, offset, vl);
        else if constexpr (LMUL == M2) return __riscv_vslidedown_vx_f16m2(src, offset, vl);
        else if constexpr (LMUL == M4) return __riscv_vslidedown_vx_f16m4(src, offset, vl);
        else if constexpr (LMUL == M8) return __riscv_vslidedown_vx_f16m8(src, offset, vl);
    }
	else if constexpr (std::is_same_v<T, float>) {
		if constexpr (LMUL == MF2) return __riscv_vslidedown_vx_f32mf2(src, offset, vl);
        else if constexpr (LMUL == M1) return __riscv_vslidedown_vx_f32m1(src, offset, vl);
        else if constexpr (LMUL == M2) return __riscv_vslidedown_vx_f32m2(src, offset, vl);
        else if constexpr (LMUL == M4) return __riscv_vslidedown_vx_f32m4(src, offset, vl);
        else if constexpr (LMUL == M8) return __riscv_vslidedown_vx_f32m8(src, offset, vl);
	}
	else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vslidedown_vx_f64m1(src, offset, vl);
        else if constexpr (LMUL == M2) return __riscv_vslidedown_vx_f64m2(src, offset, vl);
        else if constexpr (LMUL == M4) return __riscv_vslidedown_vx_f64m4(src, offset, vl);
        else if constexpr (LMUL == M8) return __riscv_vslidedown_vx_f64m8(src, offset, vl);
	}
	else if constexpr (std::is_same_v<T, int8_t>) {
		if constexpr (LMUL == MF8) return __riscv_vslidedown_vx_i8mf8(src, offset, vl);
        else if constexpr (LMUL == MF4) return __riscv_vslidedown_vx_i8mf4(src, offset, vl);
		else if constexpr (LMUL == MF2) return __riscv_vslidedown_vx_i8mf2(src, offset, vl);
        else if constexpr (LMUL == M1) return __riscv_vslidedown_vx_i8m1(src, offset, vl);
        else if constexpr (LMUL == M2) return __riscv_vslidedown_vx_i8m2(src, offset, vl);
        else if constexpr (LMUL == M4) return __riscv_vslidedown_vx_i8m4(src, offset, vl);
        else if constexpr (LMUL == M8) return __riscv_vslidedown_vx_i8m8(src, offset, vl);
    }
	else if constexpr (std::is_same_v<T, int8_t>) {
		if constexpr (LMUL == MF8) return __riscv_vslidedown_vx_i8mf8(src, offset, vl);
        else if constexpr (LMUL == MF4) return __riscv_vslidedown_vx_i8mf4(src, offset, vl);
		else if constexpr (LMUL == MF2) return __riscv_vslidedown_vx_i8mf2(src, offset, vl);
        else if constexpr (LMUL == M1) return __riscv_vslidedown_vx_i8m1(src, offset, vl);
        else if constexpr (LMUL == M2) return __riscv_vslidedown_vx_i8m2(src, offset, vl);
        else if constexpr (LMUL == M4) return __riscv_vslidedown_vx_i8m4(src, offset, vl);
        else if constexpr (LMUL == M8) return __riscv_vslidedown_vx_i8m8(src, offset, vl);
    }
	else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vslidedown_vx_i16mf4(src, offset, vl);
		else if constexpr (LMUL == MF2) return __riscv_vslidedown_vx_i16mf2(src, offset, vl);
        else if constexpr (LMUL == M1) return __riscv_vslidedown_vx_i16m1(src, offset, vl);
        else if constexpr (LMUL == M2) return __riscv_vslidedown_vx_i16m2(src, offset, vl);
        else if constexpr (LMUL == M4) return __riscv_vslidedown_vx_i16m4(src, offset, vl);
        else if constexpr (LMUL == M8) return __riscv_vslidedown_vx_i16m8(src, offset, vl);
    }
	else if constexpr (std::is_same_v<T, int32_t>) {
		if constexpr (LMUL == MF2) return __riscv_vslidedown_vx_i32mf2(src, offset, vl);
        else if constexpr (LMUL == M1) return __riscv_vslidedown_vx_i32m1(src, offset, vl);
        else if constexpr (LMUL == M2) return __riscv_vslidedown_vx_i32m2(src, offset, vl);
        else if constexpr (LMUL == M4) return __riscv_vslidedown_vx_i32m4(src, offset, vl);
        else if constexpr (LMUL == M8) return __riscv_vslidedown_vx_i32m8(src, offset, vl);
    }
	else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vslidedown_vx_i64m1(src, offset, vl);
        else if constexpr (LMUL == M2) return __riscv_vslidedown_vx_i64m2(src, offset, vl);
        else if constexpr (LMUL == M4) return __riscv_vslidedown_vx_i64m4(src, offset, vl);
        else if constexpr (LMUL == M8) return __riscv_vslidedown_vx_i64m8(src, offset, vl);
    }
	else if constexpr (std::is_same_v<T, uint8_t>) {
		if constexpr (LMUL == MF8) return __riscv_vslidedown_vx_u8mf8(src, offset, vl);
        else if constexpr (LMUL == MF4) return __riscv_vslidedown_vx_u8mf4(src, offset, vl);
		else if constexpr (LMUL == MF2) return __riscv_vslidedown_vx_u8mf2(src, offset, vl);
        else if constexpr (LMUL == M1) return __riscv_vslidedown_vx_u8m1(src, offset, vl);
        else if constexpr (LMUL == M2) return __riscv_vslidedown_vx_u8m2(src, offset, vl);
        else if constexpr (LMUL == M4) return __riscv_vslidedown_vx_u8m4(src, offset, vl);
        else if constexpr (LMUL == M8) return __riscv_vslidedown_vx_u8m8(src, offset, vl);
    }
	else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vslidedown_vx_u16mf4(src, offset, vl);
		else if constexpr (LMUL == MF2) return __riscv_vslidedown_vx_u16mf2(src, offset, vl);
        else if constexpr (LMUL == M1) return __riscv_vslidedown_vx_u16m1(src, offset, vl);
        else if constexpr (LMUL == M2) return __riscv_vslidedown_vx_u16m2(src, offset, vl);
        else if constexpr (LMUL == M4) return __riscv_vslidedown_vx_u16m4(src, offset, vl);
        else if constexpr (LMUL == M8) return __riscv_vslidedown_vx_u16m8(src, offset, vl);
    }
	else if constexpr (std::is_same_v<T, uint32_t>) {
		if constexpr (LMUL == MF2) return __riscv_vslidedown_vx_u32mf2(src, offset, vl);
        else if constexpr (LMUL == M1) return __riscv_vslidedown_vx_u32m1(src, offset, vl);
        else if constexpr (LMUL == M2) return __riscv_vslidedown_vx_u32m2(src, offset, vl);
        else if constexpr (LMUL == M4) return __riscv_vslidedown_vx_u32m4(src, offset, vl);
        else if constexpr (LMUL == M8) return __riscv_vslidedown_vx_u32m8(src, offset, vl);
    }
	else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vslidedown_vx_u64m1(src, offset, vl);
        else if constexpr (LMUL == M2) return __riscv_vslidedown_vx_u64m2(src, offset, vl);
        else if constexpr (LMUL == M4) return __riscv_vslidedown_vx_u64m4(src, offset, vl);
        else if constexpr (LMUL == M8) return __riscv_vslidedown_vx_u64m8(src, offset, vl);
    }
}



#endif // RVV_SLIDEDOWN_HPP