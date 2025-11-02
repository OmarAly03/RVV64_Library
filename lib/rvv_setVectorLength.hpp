#ifndef RVV_SETVECTOR_LENGTH_HPP
#define RVV_SETVECTOR_LENGTH_HPP

#include <cstddef> 
#include <riscv_vector.h>
#include <type_traits>

/*
size_t __riscv_vsetvl_e8mf8 (size_t avl);
size_t __riscv_vsetvl_e8mf4 (size_t avl);
size_t __riscv_vsetvl_e8mf2 (size_t avl);
size_t __riscv_vsetvl_e8m1 (size_t avl);
size_t __riscv_vsetvl_e8m2 (size_t avl);
size_t __riscv_vsetvl_e8m4 (size_t avl);
size_t __riscv_vsetvl_e8m8 (size_t avl);
size_t __riscv_vsetvl_e16mf4 (size_t avl);
size_t __riscv_vsetvl_e16mf2 (size_t avl);
size_t __riscv_vsetvl_e16m1 (size_t avl);
size_t __riscv_vsetvl_e16m2 (size_t avl);
size_t __riscv_vsetvl_e16m4 (size_t avl);
size_t __riscv_vsetvl_e16m8 (size_t avl);
size_t __riscv_vsetvl_e32mf2 (size_t avl);
size_t __riscv_vsetvl_e32m1 (size_t avl);
size_t __riscv_vsetvl_e32m2 (size_t avl);
size_t __riscv_vsetvl_e32m4 (size_t avl);
size_t __riscv_vsetvl_e32m8 (size_t avl);
size_t __riscv_vsetvl_e64m1 (size_t avl);
size_t __riscv_vsetvl_e64m2 (size_t avl);
size_t __riscv_vsetvl_e64m4 (size_t avl);
size_t __riscv_vsetvl_e64m8 (size_t avl);

*/

template<typename T, int LMUL>
inline size_t SET_VECTOR_LENGTH(size_t avl) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == M1) return __riscv_vsetvl_e32m1(avl);
        else if constexpr (LMUL == M2) return __riscv_vsetvl_e32m2(avl);
        else if constexpr (LMUL == M4) return __riscv_vsetvl_e32m4(avl);
        else if constexpr (LMUL == M8) return __riscv_vsetvl_e32m8(avl);
    }
    else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vsetvl_e64m1(avl);
        else if constexpr (LMUL == M2) return __riscv_vsetvl_e64m2(avl);
        else if constexpr (LMUL == M4) return __riscv_vsetvl_e64m4(avl);
        else if constexpr (LMUL == M8) return __riscv_vsetvl_e64m8(avl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == M1) return __riscv_vsetvl_e16m1(avl);
        else if constexpr (LMUL == M2) return __riscv_vsetvl_e16m2(avl);
        else if constexpr (LMUL == M4) return __riscv_vsetvl_e16m4(avl);
        else if constexpr (LMUL == M8) return __riscv_vsetvl_e16m8(avl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == M1) return __riscv_vsetvl_e8m1(avl);
        else if constexpr (LMUL == M2) return __riscv_vsetvl_e8m2(avl);
        else if constexpr (LMUL == M4) return __riscv_vsetvl_e8m4(avl);
        else if constexpr (LMUL == M8) return __riscv_vsetvl_e8m8(avl);
    }
    return 0;
}

template<typename T, int LMUL>
inline size_t SET_VECTOR_LENGTH_MAX(){
	if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ) {
		if constexpr (LMUL == MF2) return __riscv_vsetvlmax_e32mf2();
        else if constexpr (LMUL == M1) return __riscv_vsetvlmax_e32m1();
        else if constexpr (LMUL == M2) return __riscv_vsetvlmax_e32m2();
        else if constexpr (LMUL == M4) return __riscv_vsetvlmax_e32m4();
        else if constexpr (LMUL == M8) return __riscv_vsetvlmax_e32m8();
    }
	else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t> ) {
        if constexpr (LMUL == M1) return __riscv_vsetvlmax_e64m1();
        else if constexpr (LMUL == M2) return __riscv_vsetvlmax_e64m2();
        else if constexpr (LMUL == M4) return __riscv_vsetvlmax_e64m4();
        else if constexpr (LMUL == M8) return __riscv_vsetvlmax_e64m8();
    }
	else if constexpr (std::is_same_v<T, _Float16> || std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> ) {
		if constexpr (LMUL == MF4) return __riscv_vsetvlmax_e16mf4();
		else if constexpr (LMUL == MF2) return __riscv_vsetvlmax_e16mf2();
        else if constexpr (LMUL == M1) return __riscv_vsetvlmax_e16m1();
        else if constexpr (LMUL == M2) return __riscv_vsetvlmax_e16m2();
        else if constexpr (LMUL == M4) return __riscv_vsetvlmax_e16m4();
        else if constexpr (LMUL == M8) return __riscv_vsetvlmax_e16m8();
    }
	else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> ) {
		if constexpr (LMUL == MF8) return __riscv_vsetvlmax_e8mf8();
		else if constexpr (LMUL == MF4) return __riscv_vsetvlmax_e8mf4();
		else if constexpr (LMUL == MF2) return __riscv_vsetvlmax_e8mf2();
        else if constexpr (LMUL == M1) return __riscv_vsetvlmax_e8m1();
        else if constexpr (LMUL == M2) return __riscv_vsetvlmax_e8m2();
        else if constexpr (LMUL == M4) return __riscv_vsetvlmax_e8m4();
        else if constexpr (LMUL == M8) return __riscv_vsetvlmax_e8m8();
    }
}

#endif // RVV_SETVECTOR_LENGTH_HPP