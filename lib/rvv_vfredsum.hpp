#ifndef RVV_VFREDSUM_HPP
#define RVV_VFREDSUM_HPP

#include <cstddef> 
#include <riscv_vector.h>
#include <type_traits>

template<typename T, int LMUL, typename VecType, typename ScalarType>
inline auto VECTOR_VFREDSUM(VecType vector, ScalarType scalar, size_t vl) {
    if constexpr (std::is_same_v<T, _Float16>) {
        if constexpr (LMUL == MF4) return __riscv_vfredusum_vs_f16mf4_f16m1(vector, scalar, vl);
        else if constexpr (LMUL == MF2) return __riscv_vfredusum_vs_f16mf2_f16m1(vector, scalar, vl);
        else if constexpr (LMUL == M1) return __riscv_vfredusum_vs_f16m1_f16m1(vector, scalar, vl);
        else if constexpr (LMUL == M2) return __riscv_vfredusum_vs_f16m2_f16m1(vector, scalar, vl);
        else if constexpr (LMUL == M4) return __riscv_vfredusum_vs_f16m4_f16m1(vector, scalar, vl);
        else if constexpr (LMUL == M8) return __riscv_vfredusum_vs_f16m8_f16m1(vector, scalar, vl);
    }
    else if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfredusum_vs_f32mf2_f32m1(vector, scalar, vl);
        else if constexpr (LMUL == M1) return __riscv_vfredusum_vs_f32m1_f32m1(vector, scalar, vl);
        else if constexpr (LMUL == M2) return __riscv_vfredusum_vs_f32m2_f32m1(vector, scalar, vl);
        else if constexpr (LMUL == M4) return __riscv_vfredusum_vs_f32m4_f32m1(vector, scalar, vl);
        else if constexpr (LMUL == M8) return __riscv_vfredusum_vs_f32m8_f32m1(vector, scalar, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfredusum_vs_f64m1_f64m1(vector, scalar, vl);
        else if constexpr (LMUL == M2) return __riscv_vfredusum_vs_f64m2_f64m1(vector, scalar, vl);
        else if constexpr (LMUL == M4) return __riscv_vfredusum_vs_f64m4_f64m1(vector, scalar, vl);
        else if constexpr (LMUL == M8) return __riscv_vfredusum_vs_f64m8_f64m1(vector, scalar, vl);
    }
}

#endif // RVV_VFREDSUM_HPP