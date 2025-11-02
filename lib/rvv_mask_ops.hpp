#ifndef RVV_MASK_OPS_HPP
#define RVV_MASK_OPS_HPP

#include <cstddef> 
#include <riscv_vector.h>

// Mask-Mask AND operation
template<int LMUL, typename MaskType>
inline auto MASK_AND(MaskType mask1, MaskType mask2, size_t vl) {
    if constexpr (LMUL == MF8) return __riscv_vmand_mm_b64(mask1, mask2, vl);
    else if constexpr (LMUL == MF4) return __riscv_vmand_mm_b32(mask1, mask2, vl);
    else if constexpr (LMUL == MF2) return __riscv_vmand_mm_b16(mask1, mask2, vl);
    else if constexpr (LMUL == M1) return __riscv_vmand_mm_b8(mask1, mask2, vl);
    else if constexpr (LMUL == M2) return __riscv_vmand_mm_b4(mask1, mask2, vl);
    else if constexpr (LMUL == M4) return __riscv_vmand_mm_b2(mask1, mask2, vl);
    else if constexpr (LMUL == M8) return __riscv_vmand_mm_b1(mask1, mask2, vl);
}

// Note: For element width 32 with LMUL=8, mask width is b4
template<typename T, int LMUL, typename MaskType>
inline auto MASK_AND_E32(MaskType mask1, MaskType mask2, size_t vl) {
    if constexpr (LMUL == MF2) return __riscv_vmand_mm_b64(mask1, mask2, vl);
    else if constexpr (LMUL == M1) return __riscv_vmand_mm_b32(mask1, mask2, vl);
    else if constexpr (LMUL == M2) return __riscv_vmand_mm_b16(mask1, mask2, vl);
    else if constexpr (LMUL == M4) return __riscv_vmand_mm_b8(mask1, mask2, vl);
    else if constexpr (LMUL == M8) return __riscv_vmand_mm_b4(mask1, mask2, vl);
}

// Mask-Mask OR operation
template<int LMUL, typename MaskType>
inline auto MASK_OR(MaskType mask1, MaskType mask2, size_t vl) {
    if constexpr (LMUL == MF8) return __riscv_vmor_mm_b64(mask1, mask2, vl);
    else if constexpr (LMUL == MF4) return __riscv_vmor_mm_b32(mask1, mask2, vl);
    else if constexpr (LMUL == MF2) return __riscv_vmor_mm_b16(mask1, mask2, vl);
    else if constexpr (LMUL == M1) return __riscv_vmor_mm_b8(mask1, mask2, vl);
    else if constexpr (LMUL == M2) return __riscv_vmor_mm_b4(mask1, mask2, vl);
    else if constexpr (LMUL == M4) return __riscv_vmor_mm_b2(mask1, mask2, vl);
    else if constexpr (LMUL == M8) return __riscv_vmor_mm_b1(mask1, mask2, vl);
}

template<typename T, int LMUL, typename MaskType>
inline auto MASK_OR_E32(MaskType mask1, MaskType mask2, size_t vl) {
    if constexpr (LMUL == MF2) return __riscv_vmor_mm_b64(mask1, mask2, vl);
    else if constexpr (LMUL == M1) return __riscv_vmor_mm_b32(mask1, mask2, vl);
    else if constexpr (LMUL == M2) return __riscv_vmor_mm_b16(mask1, mask2, vl);
    else if constexpr (LMUL == M4) return __riscv_vmor_mm_b8(mask1, mask2, vl);
    else if constexpr (LMUL == M8) return __riscv_vmor_mm_b4(mask1, mask2, vl);
}

// Mask-Mask XOR operation
template<int LMUL, typename MaskType>
inline auto MASK_XOR(MaskType mask1, MaskType mask2, size_t vl) {
    if constexpr (LMUL == MF8) return __riscv_vmxor_mm_b64(mask1, mask2, vl);
    else if constexpr (LMUL == MF4) return __riscv_vmxor_mm_b32(mask1, mask2, vl);
    else if constexpr (LMUL == MF2) return __riscv_vmxor_mm_b16(mask1, mask2, vl);
    else if constexpr (LMUL == M1) return __riscv_vmxor_mm_b8(mask1, mask2, vl);
    else if constexpr (LMUL == M2) return __riscv_vmxor_mm_b4(mask1, mask2, vl);
    else if constexpr (LMUL == M4) return __riscv_vmxor_mm_b2(mask1, mask2, vl);
    else if constexpr (LMUL == M8) return __riscv_vmxor_mm_b1(mask1, mask2, vl);
}

#endif // RVV_MASK_OPS_HPP
