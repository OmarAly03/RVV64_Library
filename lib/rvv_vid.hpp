#ifndef RVV_VID_HPP
#define RVV_VID_HPP

#include <cstddef> 
#include <riscv_vector.h>
#include <type_traits>

template<typename T, int LMUL>
inline auto VECTOR_VID(size_t vl) {
    if constexpr (std::is_same_v<T, uint8_t>){
		if constexpr (LMUL == MF8) return __riscv_vid_v_u8mf8(vl);
        else if constexpr (LMUL == MF4) return __riscv_vid_v_u8mf4(vl);
		else if constexpr (LMUL == MF2) return __riscv_vid_v_u8mf2(vl);
        else if constexpr (LMUL == M1) return __riscv_vid_v_u8m1(vl);
        else if constexpr (LMUL == M2) return __riscv_vid_v_u8m2(vl);
        else if constexpr (LMUL == M4) return __riscv_vid_v_u8m4(vl);
        else if constexpr (LMUL == M8) return __riscv_vid_v_u8m8(vl);
	}
	else if constexpr (std::is_same_v<T, uint16_t>){
        if constexpr (LMUL == MF4) return __riscv_vid_v_u16mf4(vl);
		else if constexpr (LMUL == MF2) return __riscv_vid_v_u16mf2(vl);
        else if constexpr (LMUL == M1) return __riscv_vid_v_u16m1(vl);
        else if constexpr (LMUL == M2) return __riscv_vid_v_u16m2(vl);
        else if constexpr (LMUL == M4) return __riscv_vid_v_u16m4(vl);
        else if constexpr (LMUL == M8) return __riscv_vid_v_u16m8(vl);
	}
	else if constexpr (std::is_same_v<T, uint32_t>){
		if constexpr (LMUL == MF2) return __riscv_vid_v_u32mf2(vl);
        else if constexpr (LMUL == M1) return __riscv_vid_v_u32m1(vl);
        else if constexpr (LMUL == M2) return __riscv_vid_v_u32m2(vl);
        else if constexpr (LMUL == M4) return __riscv_vid_v_u32m4(vl);
        else if constexpr (LMUL == M8) return __riscv_vid_v_u32m8(vl);
	}
	else if constexpr (std::is_same_v<T, uint64_t>){
        if constexpr (LMUL == M1) return __riscv_vid_v_u64m1(vl);
        else if constexpr (LMUL == M2) return __riscv_vid_v_u64m2(vl);
        else if constexpr (LMUL == M4) return __riscv_vid_v_u64m4(vl);
        else if constexpr (LMUL == M8) return __riscv_vid_v_u64m8(vl);
	}
}

#endif // RVV_VID_HPP