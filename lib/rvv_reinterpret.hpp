#ifndef RVV_REINTERPRET_HPP
#define RVV_REINTERPRET_HPP

#include <cstddef> 
#include <riscv_vector.h>
#include <type_traits>

// Vector Type Reinterpretation from unsigned to signed
template<typename TFrom, typename TTo, int LMUL, typename VecType>
inline auto VECTOR_REINTERPRET(VecType vec) {
    // uint32 to int32
    if constexpr (std::is_same_v<TFrom, uint32_t> && std::is_same_v<TTo, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vreinterpret_v_u32mf2_i32mf2(vec);
        else if constexpr (LMUL == M1) return __riscv_vreinterpret_v_u32m1_i32m1(vec);
        else if constexpr (LMUL == M2) return __riscv_vreinterpret_v_u32m2_i32m2(vec);
        else if constexpr (LMUL == M4) return __riscv_vreinterpret_v_u32m4_i32m4(vec);
        else if constexpr (LMUL == M8) return __riscv_vreinterpret_v_u32m8_i32m8(vec);
    }
    // int32 to uint32
    else if constexpr (std::is_same_v<TFrom, int32_t> && std::is_same_v<TTo, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vreinterpret_v_i32mf2_u32mf2(vec);
        else if constexpr (LMUL == M1) return __riscv_vreinterpret_v_i32m1_u32m1(vec);
        else if constexpr (LMUL == M2) return __riscv_vreinterpret_v_i32m2_u32m2(vec);
        else if constexpr (LMUL == M4) return __riscv_vreinterpret_v_i32m4_u32m4(vec);
        else if constexpr (LMUL == M8) return __riscv_vreinterpret_v_i32m8_u32m8(vec);
    }
    // uint64 to int64
    else if constexpr (std::is_same_v<TFrom, uint64_t> && std::is_same_v<TTo, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vreinterpret_v_u64m1_i64m1(vec);
        else if constexpr (LMUL == M2) return __riscv_vreinterpret_v_u64m2_i64m2(vec);
        else if constexpr (LMUL == M4) return __riscv_vreinterpret_v_u64m4_i64m4(vec);
        else if constexpr (LMUL == M8) return __riscv_vreinterpret_v_u64m8_i64m8(vec);
    }
    // int64 to uint64
    else if constexpr (std::is_same_v<TFrom, int64_t> && std::is_same_v<TTo, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vreinterpret_v_i64m1_u64m1(vec);
        else if constexpr (LMUL == M2) return __riscv_vreinterpret_v_i64m2_u64m2(vec);
        else if constexpr (LMUL == M4) return __riscv_vreinterpret_v_i64m4_u64m4(vec);
        else if constexpr (LMUL == M8) return __riscv_vreinterpret_v_i64m8_u64m8(vec);
    }
    // uint16 to int16
    else if constexpr (std::is_same_v<TFrom, uint16_t> && std::is_same_v<TTo, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vreinterpret_v_u16mf4_i16mf4(vec);
        else if constexpr (LMUL == MF2) return __riscv_vreinterpret_v_u16mf2_i16mf2(vec);
        else if constexpr (LMUL == M1) return __riscv_vreinterpret_v_u16m1_i16m1(vec);
        else if constexpr (LMUL == M2) return __riscv_vreinterpret_v_u16m2_i16m2(vec);
        else if constexpr (LMUL == M4) return __riscv_vreinterpret_v_u16m4_i16m4(vec);
        else if constexpr (LMUL == M8) return __riscv_vreinterpret_v_u16m8_i16m8(vec);
    }
    // int16 to uint16
    else if constexpr (std::is_same_v<TFrom, int16_t> && std::is_same_v<TTo, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vreinterpret_v_i16mf4_u16mf4(vec);
        else if constexpr (LMUL == MF2) return __riscv_vreinterpret_v_i16mf2_u16mf2(vec);
        else if constexpr (LMUL == M1) return __riscv_vreinterpret_v_i16m1_u16m1(vec);
        else if constexpr (LMUL == M2) return __riscv_vreinterpret_v_i16m2_u16m2(vec);
        else if constexpr (LMUL == M4) return __riscv_vreinterpret_v_i16m4_u16m4(vec);
        else if constexpr (LMUL == M8) return __riscv_vreinterpret_v_i16m8_u16m8(vec);
    }
    // uint8 to int8
    else if constexpr (std::is_same_v<TFrom, uint8_t> && std::is_same_v<TTo, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vreinterpret_v_u8mf8_i8mf8(vec);
        else if constexpr (LMUL == MF4) return __riscv_vreinterpret_v_u8mf4_i8mf4(vec);
        else if constexpr (LMUL == MF2) return __riscv_vreinterpret_v_u8mf2_i8mf2(vec);
        else if constexpr (LMUL == M1) return __riscv_vreinterpret_v_u8m1_i8m1(vec);
        else if constexpr (LMUL == M2) return __riscv_vreinterpret_v_u8m2_i8m2(vec);
        else if constexpr (LMUL == M4) return __riscv_vreinterpret_v_u8m4_i8m4(vec);
        else if constexpr (LMUL == M8) return __riscv_vreinterpret_v_u8m8_i8m8(vec);
    }
    // int8 to uint8
    else if constexpr (std::is_same_v<TFrom, int8_t> && std::is_same_v<TTo, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vreinterpret_v_i8mf8_u8mf8(vec);
        else if constexpr (LMUL == MF4) return __riscv_vreinterpret_v_i8mf4_u8mf4(vec);
        else if constexpr (LMUL == MF2) return __riscv_vreinterpret_v_i8mf2_u8mf2(vec);
        else if constexpr (LMUL == M1) return __riscv_vreinterpret_v_i8m1_u8m1(vec);
        else if constexpr (LMUL == M2) return __riscv_vreinterpret_v_i8m2_u8m2(vec);
        else if constexpr (LMUL == M4) return __riscv_vreinterpret_v_i8m4_u8m4(vec);
        else if constexpr (LMUL == M8) return __riscv_vreinterpret_v_i8m8_u8m8(vec);
    }
}

#endif // RVV_REINTERPRET_HPP
