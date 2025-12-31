#ifndef RVV_VECTOR_NARROW
#define RVV_VECTOR_NARROW

#include <cstddef>
#include <riscv_vector.h>
#include <type_traits>

/*
vint8mf8_t __riscv_vnsra_wv_i8mf8 (vint16mf4_t op1, vuint8mf8_t shift, size_t vl);
vint8mf8_t __riscv_vnsra_wx_i8mf8 (vint16mf4_t op1, size_t shift, size_t vl);
vint8mf4_t __riscv_vnsra_wv_i8mf4 (vint16mf2_t op1, vuint8mf4_t shift, size_t vl);
vint8mf4_t __riscv_vnsra_wx_i8mf4 (vint16mf2_t op1, size_t shift, size_t vl);
vint8mf2_t __riscv_vnsra_wv_i8mf2 (vint16m1_t op1, vuint8mf2_t shift, size_t vl);
vint8mf2_t __riscv_vnsra_wx_i8mf2 (vint16m1_t op1, size_t shift, size_t vl);
vint8m1_t __riscv_vnsra_wv_i8m1 (vint16m2_t op1, vuint8m1_t shift, size_t vl);
vint8m1_t __riscv_vnsra_wx_i8m1 (vint16m2_t op1, size_t shift, size_t vl);
vint8m2_t __riscv_vnsra_wv_i8m2 (vint16m4_t op1, vuint8m2_t shift, size_t vl);
vint8m2_t __riscv_vnsra_wx_i8m2 (vint16m4_t op1, size_t shift, size_t vl);
vint8m4_t __riscv_vnsra_wv_i8m4 (vint16m8_t op1, vuint8m4_t shift, size_t vl);
vint8m4_t __riscv_vnsra_wx_i8m4 (vint16m8_t op1, size_t shift, size_t vl);
vint16mf4_t __riscv_vnsra_wv_i16mf4 (vint32mf2_t op1, vuint16mf4_t shift, size_t vl);
vint16mf4_t __riscv_vnsra_wx_i16mf4 (vint32mf2_t op1, size_t shift, size_t vl);
vint16mf2_t __riscv_vnsra_wv_i16mf2 (vint32m1_t op1, vuint16mf2_t shift, size_t vl);
vint16mf2_t __riscv_vnsra_wx_i16mf2 (vint32m1_t op1, size_t shift, size_t vl);
vint16m1_t __riscv_vnsra_wv_i16m1 (vint32m2_t op1, vuint16m1_t shift, size_t vl);
vint16m1_t __riscv_vnsra_wx_i16m1 (vint32m2_t op1, size_t shift, size_t vl);
vint16m2_t __riscv_vnsra_wv_i16m2 (vint32m4_t op1, vuint16m2_t shift, size_t vl);
vint16m2_t __riscv_vnsra_wx_i16m2 (vint32m4_t op1, size_t shift, size_t vl);
vint16m4_t __riscv_vnsra_wv_i16m4 (vint32m8_t op1, vuint16m4_t shift, size_t vl);
vint16m4_t __riscv_vnsra_wx_i16m4 (vint32m8_t op1, size_t shift, size_t vl);
vint32mf2_t __riscv_vnsra_wv_i32mf2 (vint64m1_t op1, vuint32mf2_t shift, size_t vl);
vint32mf2_t __riscv_vnsra_wx_i32mf2 (vint64m1_t op1, size_t shift, size_t vl);
vint32m1_t __riscv_vnsra_wv_i32m1 (vint64m2_t op1, vuint32m1_t shift, size_t vl);
vint32m1_t __riscv_vnsra_wx_i32m1 (vint64m2_t op1, size_t shift, size_t vl);
vint32m2_t __riscv_vnsra_wv_i32m2 (vint64m4_t op1, vuint32m2_t shift, size_t vl);
vint32m2_t __riscv_vnsra_wx_i32m2 (vint64m4_t op1, size_t shift, size_t vl);
vint32m4_t __riscv_vnsra_wv_i32m4 (vint64m8_t op1, vuint32m4_t shift, size_t vl);
vint32m4_t __riscv_vnsra_wx_i32m4 (vint64m8_t op1, size_t shift, size_t vl);
vuint8mf8_t __riscv_vnsrl_wv_u8mf8 (vuint16mf4_t op1, vuint8mf8_t shift, size_t vl);
vuint8mf8_t __riscv_vnsrl_wx_u8mf8 (vuint16mf4_t op1, size_t shift, size_t vl);
vuint8mf4_t __riscv_vnsrl_wv_u8mf4 (vuint16mf2_t op1, vuint8mf4_t shift, size_t vl);
vuint8mf4_t __riscv_vnsrl_wx_u8mf4 (vuint16mf2_t op1, size_t shift, size_t vl);
vuint8mf2_t __riscv_vnsrl_wv_u8mf2 (vuint16m1_t op1, vuint8mf2_t shift, size_t vl);
vuint8mf2_t __riscv_vnsrl_wx_u8mf2 (vuint16m1_t op1, size_t shift, size_t vl);
vuint8m1_t __riscv_vnsrl_wv_u8m1 (vuint16m2_t op1, vuint8m1_t shift, size_t vl);
vuint8m1_t __riscv_vnsrl_wx_u8m1 (vuint16m2_t op1, size_t shift, size_t vl);
vuint8m2_t __riscv_vnsrl_wv_u8m2 (vuint16m4_t op1, vuint8m2_t shift, size_t vl);
vuint8m2_t __riscv_vnsrl_wx_u8m2 (vuint16m4_t op1, size_t shift, size_t vl);
vuint8m4_t __riscv_vnsrl_wv_u8m4 (vuint16m8_t op1, vuint8m4_t shift, size_t vl);
vuint8m4_t __riscv_vnsrl_wx_u8m4 (vuint16m8_t op1, size_t shift, size_t vl);
vuint16mf4_t __riscv_vnsrl_wv_u16mf4 (vuint32mf2_t op1, vuint16mf4_t shift, size_t vl);
vuint16mf4_t __riscv_vnsrl_wx_u16mf4 (vuint32mf2_t op1, size_t shift, size_t vl);
vuint16mf2_t __riscv_vnsrl_wv_u16mf2 (vuint32m1_t op1, vuint16mf2_t shift, size_t vl);
vuint16mf2_t __riscv_vnsrl_wx_u16mf2 (vuint32m1_t op1, size_t shift, size_t vl);
vuint16m1_t __riscv_vnsrl_wv_u16m1 (vuint32m2_t op1, vuint16m1_t shift, size_t vl);
vuint16m1_t __riscv_vnsrl_wx_u16m1 (vuint32m2_t op1, size_t shift, size_t vl);
vuint16m2_t __riscv_vnsrl_wv_u16m2 (vuint32m4_t op1, vuint16m2_t shift, size_t vl);
vuint16m2_t __riscv_vnsrl_wx_u16m2 (vuint32m4_t op1, size_t shift, size_t vl);
vuint16m4_t __riscv_vnsrl_wv_u16m4 (vuint32m8_t op1, vuint16m4_t shift, size_t vl);
vuint16m4_t __riscv_vnsrl_wx_u16m4 (vuint32m8_t op1, size_t shift, size_t vl);
vuint32mf2_t __riscv_vnsrl_wv_u32mf2 (vuint64m1_t op1, vuint32mf2_t shift, size_t vl);
vuint32mf2_t __riscv_vnsrl_wx_u32mf2 (vuint64m1_t op1, size_t shift, size_t vl);
vuint32m1_t __riscv_vnsrl_wv_u32m1 (vuint64m2_t op1, vuint32m1_t shift, size_t vl);
vuint32m1_t __riscv_vnsrl_wx_u32m1 (vuint64m2_t op1, size_t shift, size_t vl);
vuint32m2_t __riscv_vnsrl_wv_u32m2 (vuint64m4_t op1, vuint32m2_t shift, size_t vl);
vuint32m2_t __riscv_vnsrl_wx_u32m2 (vuint64m4_t op1, size_t shift, size_t vl);
vuint32m4_t __riscv_vnsrl_wv_u32m4 (vuint64m8_t op1, vuint32m4_t shift, size_t vl);
vuint32m4_t __riscv_vnsrl_wx_u32m4 (vuint64m8_t op1, size_t shift, size_t vl);
*/

// Vector-Vector Narrowing Shift Right Arithmetic for signed integers
template<typename T, int LMUL, typename WideVecType, typename ShiftVecType>
inline auto VECTOR_NARROW_SRA_VV(const WideVecType& op1, const ShiftVecType& shift, size_t vl) {
    if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vnsra_wv_i8mf8(op1, shift, vl);
        else if constexpr (LMUL == MF4) return __riscv_vnsra_wv_i8mf4(op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vnsra_wv_i8mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vnsra_wv_i8m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vnsra_wv_i8m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vnsra_wv_i8m4(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vnsra_wv_i16mf4(op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vnsra_wv_i16mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vnsra_wv_i16m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vnsra_wv_i16m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vnsra_wv_i16m4(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vnsra_wv_i32mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vnsra_wv_i32m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vnsra_wv_i32m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vnsra_wv_i32m4(op1, shift, vl);
    }
}

// Vector-Scalar Narrowing Shift Right Arithmetic for signed integers
template<typename T, int LMUL, typename WideVecType>
inline auto VECTOR_NARROW_SRA_VX(const WideVecType& op1, size_t shift, size_t vl) {
    if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vnsra_wx_i8mf8(op1, shift, vl);
        else if constexpr (LMUL == MF4) return __riscv_vnsra_wx_i8mf4(op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vnsra_wx_i8mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vnsra_wx_i8m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vnsra_wx_i8m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vnsra_wx_i8m4(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vnsra_wx_i16mf4(op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vnsra_wx_i16mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vnsra_wx_i16m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vnsra_wx_i16m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vnsra_wx_i16m4(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vnsra_wx_i32mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vnsra_wx_i32m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vnsra_wx_i32m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vnsra_wx_i32m4(op1, shift, vl);
    }
}

// Vector-Vector Narrowing Shift Right Logical for unsigned integers
template<typename T, int LMUL, typename WideVecType, typename ShiftVecType>
inline auto VECTOR_NARROW_SRL_VV(const WideVecType& op1, const ShiftVecType& shift, size_t vl) {
    if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vnsrl_wv_u8mf8(op1, shift, vl);
        else if constexpr (LMUL == MF4) return __riscv_vnsrl_wv_u8mf4(op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vnsrl_wv_u8mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vnsrl_wv_u8m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vnsrl_wv_u8m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vnsrl_wv_u8m4(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vnsrl_wv_u16mf4(op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vnsrl_wv_u16mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vnsrl_wv_u16m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vnsrl_wv_u16m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vnsrl_wv_u16m4(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vnsrl_wv_u32mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vnsrl_wv_u32m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vnsrl_wv_u32m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vnsrl_wv_u32m4(op1, shift, vl);
    }
}

// Vector-Scalar Narrowing Shift Right Logical for unsigned integers
template<typename T, int LMUL, typename WideVecType>
inline auto VECTOR_NARROW_SRL_VX(const WideVecType& op1, size_t shift, size_t vl) {
    if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vnsrl_wx_u8mf8(op1, shift, vl);
        else if constexpr (LMUL == MF4) return __riscv_vnsrl_wx_u8mf4(op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vnsrl_wx_u8mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vnsrl_wx_u8m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vnsrl_wx_u8m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vnsrl_wx_u8m4(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vnsrl_wx_u16mf4(op1, shift, vl);
        else if constexpr (LMUL == MF2) return __riscv_vnsrl_wx_u16mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vnsrl_wx_u16m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vnsrl_wx_u16m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vnsrl_wx_u16m4(op1, shift, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vnsrl_wx_u32mf2(op1, shift, vl);
        else if constexpr (LMUL == M1) return __riscv_vnsrl_wx_u32m1(op1, shift, vl);
        else if constexpr (LMUL == M2) return __riscv_vnsrl_wx_u32m2(op1, shift, vl);
        else if constexpr (LMUL == M4) return __riscv_vnsrl_wx_u32m4(op1, shift, vl);
    }
}

template<typename T, int LMUL, typename WideVecType, typename ShiftType>
auto VECTOR_NARROW_SRA(const WideVecType& op1, const ShiftType& shift, size_t vl) {
	if constexpr (std::is_scalar_v<ShiftType>) {
		return VECTOR_NARROW_SRA_VX<T, LMUL>(op1, shift, vl);
	} else {
		return VECTOR_NARROW_SRA_VV<T, LMUL>(op1, shift, vl);
	}
}

template<typename T, int LMUL, typename WideVecType, typename ShiftType>
auto VECTOR_NARROW_SRL(const WideVecType& op1, const ShiftType& shift, size_t vl) {
	if constexpr (std::is_scalar_v<ShiftType>) {
		return VECTOR_NARROW_SRL_VX<T, LMUL>(op1, shift, vl);
	} else {
		return VECTOR_NARROW_SRL_VV<T, LMUL>(op1, shift, vl);
	}
}

#endif // RVV_VECTOR_NARROW