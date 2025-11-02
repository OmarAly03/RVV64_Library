#ifndef RVV_COUNT_POP_HPP
#define RVV_COUNT_POP_HPP

#include <cstddef> 
#include <riscv_vector.h>
#include <type_traits>

template<typename BoolType>
inline size_t VECTOR_COUNT_POP(const BoolType& mask, size_t vl) {
    if constexpr (std::is_same_v<BoolType, vbool1_t>) return __riscv_vcpop_m_b1(mask, vl);
    else if constexpr (std::is_same_v<BoolType, vbool2_t>) return __riscv_vcpop_m_b2(mask, vl);
    else if constexpr (std::is_same_v<BoolType, vbool4_t>) return __riscv_vcpop_m_b4(mask, vl);
    else if constexpr (std::is_same_v<BoolType, vbool8_t>) return __riscv_vcpop_m_b8(mask, vl);
    else if constexpr (std::is_same_v<BoolType, vbool16_t>) return __riscv_vcpop_m_b16(mask, vl);
    else if constexpr (std::is_same_v<BoolType, vbool32_t>) return __riscv_vcpop_m_b32(mask, vl);
    else if constexpr (std::is_same_v<BoolType, vbool64_t>) return __riscv_vcpop_m_b64(mask, vl);
}

#endif // RVV_COUNT_POP_HPP