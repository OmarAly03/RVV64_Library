#ifndef RVV_INDEXED_LOAD_HPP
#define RVV_INDEXED_LOAD_HPP

#include <cstddef> 
#include <riscv_vector.h>
#include <type_traits>

// Unordered Indexed Load (Gather) - Masked Undisturbed
template<typename T, int LMUL, typename MaskType, typename VecType, typename IdxType>
inline auto VECTOR_INDEXED_LOAD_MU(MaskType mask, VecType maskedoff, const T* base, IdxType index, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vluxei32_v_f32mf2_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M1) return __riscv_vluxei32_v_f32m1_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M2) return __riscv_vluxei32_v_f32m2_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M4) return __riscv_vluxei32_v_f32m4_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M8) return __riscv_vluxei32_v_f32m8_mu(mask, maskedoff, base, index, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vluxei64_v_f64m1_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M2) return __riscv_vluxei64_v_f64m2_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M4) return __riscv_vluxei64_v_f64m4_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M8) return __riscv_vluxei64_v_f64m8_mu(mask, maskedoff, base, index, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vluxei32_v_i32mf2_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M1) return __riscv_vluxei32_v_i32m1_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M2) return __riscv_vluxei32_v_i32m2_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M4) return __riscv_vluxei32_v_i32m4_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M8) return __riscv_vluxei32_v_i32m8_mu(mask, maskedoff, base, index, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vluxei32_v_u32mf2_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M1) return __riscv_vluxei32_v_u32m1_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M2) return __riscv_vluxei32_v_u32m2_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M4) return __riscv_vluxei32_v_u32m4_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M8) return __riscv_vluxei32_v_u32m8_mu(mask, maskedoff, base, index, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vluxei64_v_i64m1_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M2) return __riscv_vluxei64_v_i64m2_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M4) return __riscv_vluxei64_v_i64m4_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M8) return __riscv_vluxei64_v_i64m8_mu(mask, maskedoff, base, index, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vluxei64_v_u64m1_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M2) return __riscv_vluxei64_v_u64m2_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M4) return __riscv_vluxei64_v_u64m4_mu(mask, maskedoff, base, index, vl);
        else if constexpr (LMUL == M8) return __riscv_vluxei64_v_u64m8_mu(mask, maskedoff, base, index, vl);
    }
}

// Unordered Indexed Load (Gather) - Regular (non-masked)
template<typename T, int LMUL, typename IdxType>
inline auto VECTOR_INDEXED_LOAD(const T* base, IdxType index, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vluxei32_v_f32mf2(base, index, vl);
        else if constexpr (LMUL == M1) return __riscv_vluxei32_v_f32m1(base, index, vl);
        else if constexpr (LMUL == M2) return __riscv_vluxei32_v_f32m2(base, index, vl);
        else if constexpr (LMUL == M4) return __riscv_vluxei32_v_f32m4(base, index, vl);
        else if constexpr (LMUL == M8) return __riscv_vluxei32_v_f32m8(base, index, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vluxei64_v_f64m1(base, index, vl);
        else if constexpr (LMUL == M2) return __riscv_vluxei64_v_f64m2(base, index, vl);
        else if constexpr (LMUL == M4) return __riscv_vluxei64_v_f64m4(base, index, vl);
        else if constexpr (LMUL == M8) return __riscv_vluxei64_v_f64m8(base, index, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vluxei32_v_i32mf2(base, index, vl);
        else if constexpr (LMUL == M1) return __riscv_vluxei32_v_i32m1(base, index, vl);
        else if constexpr (LMUL == M2) return __riscv_vluxei32_v_i32m2(base, index, vl);
        else if constexpr (LMUL == M4) return __riscv_vluxei32_v_i32m4(base, index, vl);
        else if constexpr (LMUL == M8) return __riscv_vluxei32_v_i32m8(base, index, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vluxei32_v_u32mf2(base, index, vl);
        else if constexpr (LMUL == M1) return __riscv_vluxei32_v_u32m1(base, index, vl);
        else if constexpr (LMUL == M2) return __riscv_vluxei32_v_u32m2(base, index, vl);
        else if constexpr (LMUL == M4) return __riscv_vluxei32_v_u32m4(base, index, vl);
        else if constexpr (LMUL == M8) return __riscv_vluxei32_v_u32m8(base, index, vl);
    }
}

#endif // RVV_INDEXED_LOAD_HPP
