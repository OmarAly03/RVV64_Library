#ifndef RVV_ARITHMETIC_HPP
#define RVV_ARITHMETIC_HPP

#include <cstddef> 
#include <riscv_vector.h>
#include <type_traits>

/*
vint8mf8_t __riscv_vadd_vv_i8mf8 (vint8mf8_t op1, vint8mf8_t op2, size_t vl);
vint8mf8_t __riscv_vadd_vx_i8mf8 (vint8mf8_t op1, int8_t op2, size_t vl);
vint8mf4_t __riscv_vadd_vv_i8mf4 (vint8mf4_t op1, vint8mf4_t op2, size_t vl);
vint8mf4_t __riscv_vadd_vx_i8mf4 (vint8mf4_t op1, int8_t op2, size_t vl);
vint8mf2_t __riscv_vadd_vv_i8mf2 (vint8mf2_t op1, vint8mf2_t op2, size_t vl);
vint8mf2_t __riscv_vadd_vx_i8mf2 (vint8mf2_t op1, int8_t op2, size_t vl);
vint8m1_t __riscv_vadd_vv_i8m1 (vint8m1_t op1, vint8m1_t op2, size_t vl);
vint8m1_t __riscv_vadd_vx_i8m1 (vint8m1_t op1, int8_t op2, size_t vl);
vint8m2_t __riscv_vadd_vv_i8m2 (vint8m2_t op1, vint8m2_t op2, size_t vl);
vint8m2_t __riscv_vadd_vx_i8m2 (vint8m2_t op1, int8_t op2, size_t vl);
vint8m4_t __riscv_vadd_vv_i8m4 (vint8m4_t op1, vint8m4_t op2, size_t vl);
vint8m4_t __riscv_vadd_vx_i8m4 (vint8m4_t op1, int8_t op2, size_t vl);
vint8m8_t __riscv_vadd_vv_i8m8 (vint8m8_t op1, vint8m8_t op2, size_t vl);
vint8m8_t __riscv_vadd_vx_i8m8 (vint8m8_t op1, int8_t op2, size_t vl);
vint16mf4_t __riscv_vadd_vv_i16mf4 (vint16mf4_t op1, vint16mf4_t op2, size_t vl);
vint16mf4_t __riscv_vadd_vx_i16mf4 (vint16mf4_t op1, int16_t op2, size_t vl);
vint16mf2_t __riscv_vadd_vv_i16mf2 (vint16mf2_t op1, vint16mf2_t op2, size_t vl);
vint16mf2_t __riscv_vadd_vx_i16mf2 (vint16mf2_t op1, int16_t op2, size_t vl);
vint16m1_t __riscv_vadd_vv_i16m1 (vint16m1_t op1, vint16m1_t op2, size_t vl);
vint16m1_t __riscv_vadd_vx_i16m1 (vint16m1_t op1, int16_t op2, size_t vl);
vint16m2_t __riscv_vadd_vv_i16m2 (vint16m2_t op1, vint16m2_t op2, size_t vl);
vint16m2_t __riscv_vadd_vx_i16m2 (vint16m2_t op1, int16_t op2, size_t vl);
vint16m4_t __riscv_vadd_vv_i16m4 (vint16m4_t op1, vint16m4_t op2, size_t vl);
vint16m4_t __riscv_vadd_vx_i16m4 (vint16m4_t op1, int16_t op2, size_t vl);
vint16m8_t __riscv_vadd_vv_i16m8 (vint16m8_t op1, vint16m8_t op2, size_t vl);
vint16m8_t __riscv_vadd_vx_i16m8 (vint16m8_t op1, int16_t op2, size_t vl);
vint32mf2_t __riscv_vadd_vv_i32mf2 (vint32mf2_t op1, vint32mf2_t op2, size_t vl);
vint32mf2_t __riscv_vadd_vx_i32mf2 (vint32mf2_t op1, int32_t op2, size_t vl);
vint32m1_t __riscv_vadd_vv_i32m1 (vint32m1_t op1, vint32m1_t op2, size_t vl);
vint32m1_t __riscv_vadd_vx_i32m1 (vint32m1_t op1, int32_t op2, size_t vl);
vint32m2_t __riscv_vadd_vv_i32m2 (vint32m2_t op1, vint32m2_t op2, size_t vl);
vint32m2_t __riscv_vadd_vx_i32m2 (vint32m2_t op1, int32_t op2, size_t vl);
vint32m4_t __riscv_vadd_vv_i32m4 (vint32m4_t op1, vint32m4_t op2, size_t vl);
vint32m4_t __riscv_vadd_vx_i32m4 (vint32m4_t op1, int32_t op2, size_t vl);
vint32m8_t __riscv_vadd_vv_i32m8 (vint32m8_t op1, vint32m8_t op2, size_t vl);
vint32m8_t __riscv_vadd_vx_i32m8 (vint32m8_t op1, int32_t op2, size_t vl);
vint64m1_t __riscv_vadd_vv_i64m1 (vint64m1_t op1, vint64m1_t op2, size_t vl);
vint64m1_t __riscv_vadd_vx_i64m1 (vint64m1_t op1, int64_t op2, size_t vl);
vint64m2_t __riscv_vadd_vv_i64m2 (vint64m2_t op1, vint64m2_t op2, size_t vl);
vint64m2_t __riscv_vadd_vx_i64m2 (vint64m2_t op1, int64_t op2, size_t vl);
vint64m4_t __riscv_vadd_vv_i64m4 (vint64m4_t op1, vint64m4_t op2, size_t vl);
vint64m4_t __riscv_vadd_vx_i64m4 (vint64m4_t op1, int64_t op2, size_t vl);
vint64m8_t __riscv_vadd_vv_i64m8 (vint64m8_t op1, vint64m8_t op2, size_t vl);
vint64m8_t __riscv_vadd_vx_i64m8 (vint64m8_t op1, int64_t op2, size_t vl);

vfloat16mf4_t __riscv_vfsub_vv_f16mf4 (vfloat16mf4_t op1, vfloat16mf4_t op2, size_t vl);
vfloat16mf4_t __riscv_vfsub_vf_f16mf4 (vfloat16mf4_t op1, float16_t op2, size_t vl);
vfloat16mf2_t __riscv_vfsub_vv_f16mf2 (vfloat16mf2_t op1, vfloat16mf2_t op2, size_t vl);
vfloat16mf2_t __riscv_vfsub_vf_f16mf2 (vfloat16mf2_t op1, float16_t op2, size_t vl);
vfloat16m1_t __riscv_vfsub_vv_f16m1 (vfloat16m1_t op1, vfloat16m1_t op2, size_t vl);
vfloat16m1_t __riscv_vfsub_vf_f16m1 (vfloat16m1_t op1, float16_t op2, size_t vl);
vfloat16m2_t __riscv_vfsub_vv_f16m2 (vfloat16m2_t op1, vfloat16m2_t op2, size_t vl);
vfloat16m2_t __riscv_vfsub_vf_f16m2 (vfloat16m2_t op1, float16_t op2, size_t vl);
vfloat16m4_t __riscv_vfsub_vv_f16m4 (vfloat16m4_t op1, vfloat16m4_t op2, size_t vl);
vfloat16m4_t __riscv_vfsub_vf_f16m4 (vfloat16m4_t op1, float16_t op2, size_t vl);
vfloat16m8_t __riscv_vfsub_vv_f16m8 (vfloat16m8_t op1, vfloat16m8_t op2, size_t vl);
vfloat16m8_t __riscv_vfsub_vf_f16m8 (vfloat16m8_t op1, float16_t op2, size_t vl);
vfloat32mf2_t __riscv_vfsub_vv_f32mf2 (vfloat32mf2_t op1, vfloat32mf2_t op2, size_t vl);
vfloat32mf2_t __riscv_vfsub_vf_f32mf2 (vfloat32mf2_t op1, float32_t op2, size_t vl);
vfloat32m1_t __riscv_vfsub_vv_f32m1 (vfloat32m1_t op1, vfloat32m1_t op2, size_t vl);
vfloat32m1_t __riscv_vfsub_vf_f32m1 (vfloat32m1_t op1, float32_t op2, size_t vl);
vfloat32m2_t __riscv_vfsub_vv_f32m2 (vfloat32m2_t op1, vfloat32m2_t op2, size_t vl);
vfloat32m2_t __riscv_vfsub_vf_f32m2 (vfloat32m2_t op1, float32_t op2, size_t vl);
vfloat32m4_t __riscv_vfsub_vv_f32m4 (vfloat32m4_t op1, vfloat32m4_t op2, size_t vl);
vfloat32m4_t __riscv_vfsub_vf_f32m4 (vfloat32m4_t op1, float32_t op2, size_t vl);
vfloat32m8_t __riscv_vfsub_vv_f32m8 (vfloat32m8_t op1, vfloat32m8_t op2, size_t vl);
vfloat32m8_t __riscv_vfsub_vf_f32m8 (vfloat32m8_t op1, float32_t op2, size_t vl);
vfloat64m1_t __riscv_vfsub_vv_f64m1 (vfloat64m1_t op1, vfloat64m1_t op2, size_t vl);
vfloat64m1_t __riscv_vfsub_vf_f64m1 (vfloat64m1_t op1, float64_t op2, size_t vl);
vfloat64m2_t __riscv_vfsub_vv_f64m2 (vfloat64m2_t op1, vfloat64m2_t op2, size_t vl);
vfloat64m2_t __riscv_vfsub_vf_f64m2 (vfloat64m2_t op1, float64_t op2, size_t vl);
vfloat64m4_t __riscv_vfsub_vv_f64m4 (vfloat64m4_t op1, vfloat64m4_t op2, size_t vl);
vfloat64m4_t __riscv_vfsub_vf_f64m4 (vfloat64m4_t op1, float64_t op2, size_t vl);
vfloat64m8_t __riscv_vfsub_vv_f64m8 (vfloat64m8_t op1, vfloat64m8_t op2, size_t vl);
vfloat64m8_t __riscv_vfsub_vf_f64m8 (vfloat64m8_t op1, float64_t op2, size_t vl);

vint8mf8_t __riscv_vmul_vv_i8mf8 (vint8mf8_t op1, vint8mf8_t op2, size_t vl);
vint8mf8_t __riscv_vmul_vx_i8mf8 (vint8mf8_t op1, int8_t op2, size_t vl);
vint8mf4_t __riscv_vmul_vv_i8mf4 (vint8mf4_t op1, vint8mf4_t op2, size_t vl);
vint8mf4_t __riscv_vmul_vx_i8mf4 (vint8mf4_t op1, int8_t op2, size_t vl);
vint8mf2_t __riscv_vmul_vv_i8mf2 (vint8mf2_t op1, vint8mf2_t op2, size_t vl);
vint8mf2_t __riscv_vmul_vx_i8mf2 (vint8mf2_t op1, int8_t op2, size_t vl);
vint8m1_t __riscv_vmul_vv_i8m1 (vint8m1_t op1, vint8m1_t op2, size_t vl);
vint8m1_t __riscv_vmul_vx_i8m1 (vint8m1_t op1, int8_t op2, size_t vl);
vint8m2_t __riscv_vmul_vv_i8m2 (vint8m2_t op1, vint8m2_t op2, size_t vl);
vint8m2_t __riscv_vmul_vx_i8m2 (vint8m2_t op1, int8_t op2, size_t vl);
vint8m4_t __riscv_vmul_vv_i8m4 (vint8m4_t op1, vint8m4_t op2, size_t vl);
vint8m4_t __riscv_vmul_vx_i8m4 (vint8m4_t op1, int8_t op2, size_t vl);
vint8m8_t __riscv_vmul_vv_i8m8 (vint8m8_t op1, vint8m8_t op2, size_t vl);
vint8m8_t __riscv_vmul_vx_i8m8 (vint8m8_t op1, int8_t op2, size_t vl);
vint16mf4_t __riscv_vmul_vv_i16mf4 (vint16mf4_t op1, vint16mf4_t op2, size_t vl);
vint16mf4_t __riscv_vmul_vx_i16mf4 (vint16mf4_t op1, int16_t op2, size_t vl);
vint16mf2_t __riscv_vmul_vv_i16mf2 (vint16mf2_t op1, vint16mf2_t op2, size_t vl);
vint16mf2_t __riscv_vmul_vx_i16mf2 (vint16mf2_t op1, int16_t op2, size_t vl);
vint16m1_t __riscv_vmul_vv_i16m1 (vint16m1_t op1, vint16m1_t op2, size_t vl);
vint16m1_t __riscv_vmul_vx_i16m1 (vint16m1_t op1, int16_t op2, size_t vl);
vint16m2_t __riscv_vmul_vv_i16m2 (vint16m2_t op1, vint16m2_t op2, size_t vl);
vint16m2_t __riscv_vmul_vx_i16m2 (vint16m2_t op1, int16_t op2, size_t vl);
vint16m4_t __riscv_vmul_vv_i16m4 (vint16m4_t op1, vint16m4_t op2, size_t vl);
vint16m4_t __riscv_vmul_vx_i16m4 (vint16m4_t op1, int16_t op2, size_t vl);
vint16m8_t __riscv_vmul_vv_i16m8 (vint16m8_t op1, vint16m8_t op2, size_t vl);
vint16m8_t __riscv_vmul_vx_i16m8 (vint16m8_t op1, int16_t op2, size_t vl);
vint32mf2_t __riscv_vmul_vv_i32mf2 (vint32mf2_t op1, vint32mf2_t op2, size_t vl);
vint32mf2_t __riscv_vmul_vx_i32mf2 (vint32mf2_t op1, int32_t op2, size_t vl);
vint32m1_t __riscv_vmul_vv_i32m1 (vint32m1_t op1, vint32m1_t op2, size_t vl);
vint32m1_t __riscv_vmul_vx_i32m1 (vint32m1_t op1, int32_t op2, size_t vl);
vint32m2_t __riscv_vmul_vv_i32m2 (vint32m2_t op1, vint32m2_t op2, size_t vl);
vint32m2_t __riscv_vmul_vx_i32m2 (vint32m2_t op1, int32_t op2, size_t vl);
vint32m4_t __riscv_vmul_vv_i32m4 (vint32m4_t op1, vint32m4_t op2, size_t vl);
vint32m4_t __riscv_vmul_vx_i32m4 (vint32m4_t op1, int32_t op2, size_t vl);
vint32m8_t __riscv_vmul_vv_i32m8 (vint32m8_t op1, vint32m8_t op2, size_t vl);
vint32m8_t __riscv_vmul_vx_i32m8 (vint32m8_t op1, int32_t op2, size_t vl);
vint64m1_t __riscv_vmul_vv_i64m1 (vint64m1_t op1, vint64m1_t op2, size_t vl);
vint64m1_t __riscv_vmul_vx_i64m1 (vint64m1_t op1, int64_t op2, size_t vl);
vint64m2_t __riscv_vmul_vv_i64m2 (vint64m2_t op1, vint64m2_t op2, size_t vl);
vint64m2_t __riscv_vmul_vx_i64m2 (vint64m2_t op1, int64_t op2, size_t vl);
vint64m4_t __riscv_vmul_vv_i64m4 (vint64m4_t op1, vint64m4_t op2, size_t vl);
vint64m4_t __riscv_vmul_vx_i64m4 (vint64m4_t op1, int64_t op2, size_t vl);
vint64m8_t __riscv_vmul_vv_i64m8 (vint64m8_t op1, vint64m8_t op2, size_t vl);
vuint8mf8_t __riscv_vmul_vv_u8mf8 (vuint8mf8_t op1, vuint8mf8_t op2, size_t vl);
vuint8mf8_t __riscv_vmul_vx_u8mf8 (vuint8mf8_t op1, uint8_t op2, size_t vl);
vuint8mf4_t __riscv_vmul_vv_u8mf4 (vuint8mf4_t op1, vuint8mf4_t op2, size_t vl);
vuint8mf4_t __riscv_vmul_vx_u8mf4 (vuint8mf4_t op1, uint8_t op2, size_t vl);
vuint8mf2_t __riscv_vmul_vv_u8mf2 (vuint8mf2_t op1, vuint8mf2_t op2, size_t vl);
vuint8mf2_t __riscv_vmul_vx_u8mf2 (vuint8mf2_t op1, uint8_t op2, size_t vl);
vuint8m1_t __riscv_vmul_vv_u8m1 (vuint8m1_t op1, vuint8m1_t op2, size_t vl);
vuint8m1_t __riscv_vmul_vx_u8m1 (vuint8m1_t op1, uint8_t op2, size_t vl);
vuint8m2_t __riscv_vmul_vv_u8m2 (vuint8m2_t op1, vuint8m2_t op2, size_t vl);
vuint8m2_t __riscv_vmul_vx_u8m2 (vuint8m2_t op1, uint8_t op2, size_t vl);
vuint8m4_t __riscv_vmul_vv_u8m4 (vuint8m4_t op1, vuint8m4_t op2, size_t vl);
vuint8m4_t __riscv_vmul_vx_u8m4 (vuint8m4_t op1, uint8_t op2, size_t vl);
vuint8m8_t __riscv_vmul_vv_u8m8 (vuint8m8_t op1, vuint8m8_t op2, size_t vl);
vuint8m8_t __riscv_vmul_vx_u8m8 (vuint8m8_t op1, uint8_t op2, size_t vl);
vuint16mf4_t __riscv_vmul_vv_u16mf4 (vuint16mf4_t op1, vuint16mf4_t op2, size_t vl);
vuint16mf4_t __riscv_vmul_vx_u16mf4 (vuint16mf4_t op1, uint16_t op2, size_t vl);
vuint16mf2_t __riscv_vmul_vv_u16mf2 (vuint16mf2_t op1, vuint16mf2_t op2, size_t vl);
vuint16mf2_t __riscv_vmul_vx_u16mf2 (vuint16mf2_t op1, uint16_t op2, size_t vl);
vuint16m1_t __riscv_vmul_vv_u16m1 (vuint16m1_t op1, vuint16m1_t op2, size_t vl);
vuint16m1_t __riscv_vmul_vx_u16m1 (vuint16m1_t op1, uint16_t op2, size_t vl);
vuint16m2_t __riscv_vmul_vv_u16m2 (vuint16m2_t op1, vuint16m2_t op2, size_t vl);
vuint16m2_t __riscv_vmul_vx_u16m2 (vuint16m2_t op1, uint16_t op2, size_t vl);
vuint16m4_t __riscv_vmul_vv_u16m4 (vuint16m4_t op1, vuint16m4_t op2, size_t vl);
vuint16m4_t __riscv_vmul_vx_u16m4 (vuint16m4_t op1, uint16_t op2, size_t vl);
vuint16m8_t __riscv_vmul_vv_u16m8 (vuint16m8_t op1, vuint16m8_t op2, size_t vl);
vuint16m8_t __riscv_vmul_vx_u16m8 (vuint16m8_t op1, uint16_t op2, size_t vl);
vuint32mf2_t __riscv_vmul_vv_u32mf2 (vuint32mf2_t op1, vuint32mf2_t op2, size_t vl);
vuint32mf2_t __riscv_vmul_vx_u32mf2 (vuint32mf2_t op1, uint32_t op2, size_t vl);
vuint32m1_t __riscv_vmul_vv_u32m1 (vuint32m1_t op1, vuint32m1_t op2, size_t vl);
vuint32m1_t __riscv_vmul_vx_u32m1 (vuint32m1_t op1, uint32_t op2, size_t vl);
vuint32m2_t __riscv_vmul_vv_u32m2 (vuint32m2_t op1, vuint32m2_t op2, size_t vl);
vuint32m2_t __riscv_vmul_vx_u32m2 (vuint32m2_t op1, uint32_t op2, size_t vl);
vuint32m4_t __riscv_vmul_vv_u32m4 (vuint32m4_t op1, vuint32m4_t op2, size_t vl);
vuint32m4_t __riscv_vmul_vx_u32m4 (vuint32m4_t op1, uint32_t op2, size_t vl);
vuint32m8_t __riscv_vmul_vv_u32m8 (vuint32m8_t op1, vuint32m8_t op2, size_t vl);
vuint32m8_t __riscv_vmul_vx_u32m8 (vuint32m8_t op1, uint32_t op2, size_t vl);
vuint64m1_t __riscv_vmul_vv_u64m1 (vuint64m1_t op1, vuint64m1_t op2, size_t vl);
vuint64m1_t __riscv_vmul_vx_u64m1 (vuint64m1_t op1, uint64_t op2, size_t vl);
vuint64m2_t __riscv_vmul_vv_u64m2 (vuint64m2_t op1, vuint64m2_t op2, size_t vl);
vuint64m2_t __riscv_vmul_vx_u64m2 (vuint64m2_t op1, uint64_t op2, size_t vl);
vuint64m4_t __riscv_vmul_vv_u64m4 (vuint64m4_t op1, vuint64m4_t op2, size_t vl);
vuint64m4_t __riscv_vmul_vx_u64m4 (vuint64m4_t op1, uint64_t op2, size_t vl);
vuint64m8_t __riscv_vmul_vv_u64m8 (vuint64m8_t op1, vuint64m8_t op2, size_t vl);
vuint64m8_t __riscv_vmul_vx_u64m8 (vuint64m8_t op1, uint64_t op2, size_t vl);
*/

// Add these templates before your VECTOR_SUB templates:

// Vector-Vector Addition Template
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_ADD_VV(VecType op1, VecType op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfadd_vv_f32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfadd_vv_f32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfadd_vv_f32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfadd_vv_f32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfadd_vv_f32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfadd_vv_f64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfadd_vv_f64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfadd_vv_f64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfadd_vv_f64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vadd_vv_i8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vadd_vv_i8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vv_i8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vv_i8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_i8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_i8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_i8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vadd_vv_i16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vv_i16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vv_i16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_i16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_i16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_i16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vadd_vv_i32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vv_i32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_i32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_i32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_i32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vadd_vv_i64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_i64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_i64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_i64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vadd_vv_u8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vadd_vv_u8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vv_u8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vv_u8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_u8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_u8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_u8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vadd_vv_u16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vv_u16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vv_u16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_u16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_u16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_u16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vadd_vv_u32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vv_u32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_u32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_u32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_u32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vadd_vv_u64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vv_u64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vv_u64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vv_u64m8(op1, op2, vl);
    }
}

// Vector-Scalar Addition Template
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_ADD_VX(VecType op1, T op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfadd_vf_f32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfadd_vf_f32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfadd_vf_f32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfadd_vf_f32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfadd_vf_f32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfadd_vf_f64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfadd_vf_f64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfadd_vf_f64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfadd_vf_f64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vadd_vx_i8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vadd_vx_i8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vx_i8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vx_i8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_i8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_i8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_i8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vadd_vx_i16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vx_i16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vx_i16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_i16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_i16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_i16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vadd_vx_i32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vx_i32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_i32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_i32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_i32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vadd_vx_i64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_i64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_i64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_i64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vadd_vx_u8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vadd_vx_u8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vx_u8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vx_u8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_u8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_u8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_u8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vadd_vx_u16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vadd_vx_u16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vx_u16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_u16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_u16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_u16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vadd_vx_u32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vadd_vx_u32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_u32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_u32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_u32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vadd_vx_u64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vadd_vx_u64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vadd_vx_u64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vadd_vx_u64m8(op1, op2, vl);
    }
}

// Unified VECTOR_ADD template that auto-detects vector vs scalar
template<typename T, int LMUL, typename VecType, typename Op2Type>
inline auto VECTOR_ADD(VecType op1, Op2Type op2, size_t vl) {
    // If op2 is a scalar type (arithmetic type), use vector-scalar addition
    if constexpr (std::is_arithmetic_v<Op2Type>) {
        return VECTOR_ADD_VX<T, LMUL>(op1, static_cast<T>(op2), vl);
    }
    // Otherwise, use vector-vector addition
    else {
        return VECTOR_ADD_VV<T, LMUL>(op1, op2, vl);
    }
}

// Vector-Vector Subtraction Template
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_SUB_VV(VecType op1, VecType op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfsub_vv_f32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfsub_vv_f32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfsub_vv_f32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfsub_vv_f32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfsub_vv_f32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfsub_vv_f64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfsub_vv_f64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfsub_vv_f64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfsub_vv_f64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vsub_vv_i8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vsub_vv_i8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsub_vv_i8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vv_i8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vv_i8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vv_i8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vv_i8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vsub_vv_i16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsub_vv_i16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vv_i16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vv_i16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vv_i16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vv_i16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vsub_vv_i32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vv_i32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vv_i32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vv_i32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vv_i32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vsub_vv_i64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vv_i64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vv_i64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vv_i64m8(op1, op2, vl);
    }
}

// Vector-Scalar Subtraction Template
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_SUB_VX(VecType op1, T op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfsub_vf_f32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfsub_vf_f32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfsub_vf_f32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfsub_vf_f32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfsub_vf_f32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfsub_vf_f64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfsub_vf_f64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfsub_vf_f64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfsub_vf_f64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vsub_vx_i8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vsub_vx_i8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsub_vx_i8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vx_i8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vx_i8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vx_i8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vx_i8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vsub_vx_i16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vsub_vx_i16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vx_i16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vx_i16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vx_i16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vx_i16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vsub_vx_i32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vsub_vx_i32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vx_i32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vx_i32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vx_i32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vsub_vx_i64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vsub_vx_i64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vsub_vx_i64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vsub_vx_i64m8(op1, op2, vl);
    }
}

// Unified VECTOR_SUB template that auto-detects vector vs scalar
template<typename T, int LMUL, typename VecType, typename Op2Type>
inline auto VECTOR_SUB(VecType op1, Op2Type op2, size_t vl) {
    // If op2 is a scalar type, use vector-scalar subtraction
    if constexpr (std::is_same_v<Op2Type, T>) {
        return VECTOR_SUB_VX<T, LMUL>(op1, op2, vl);
    }
    // Otherwise, use vector-vector subtraction
    else {
        return VECTOR_SUB_VV<T, LMUL>(op1, op2, vl);
    }
}

// Add these templates after your VECTOR_SUB templates:

// Vector-Vector Multiply Template
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_MUL_VV(VecType op1, VecType op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfmul_vv_f32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfmul_vv_f32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmul_vv_f32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmul_vv_f32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmul_vv_f32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfmul_vv_f64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmul_vv_f64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmul_vv_f64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmul_vv_f64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vmul_vv_i8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vmul_vv_i8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vv_i8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vv_i8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_i8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_i8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_i8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vmul_vv_i16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vv_i16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vv_i16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_i16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_i16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_i16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vmul_vv_i32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vv_i32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_i32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_i32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_i32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmul_vv_i64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_i64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_i64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_i64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vmul_vv_u8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vmul_vv_u8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vv_u8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vv_u8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_u8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_u8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_u8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vmul_vv_u16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vv_u16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vv_u16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_u16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_u16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_u16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vmul_vv_u32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vv_u32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_u32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_u32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_u32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmul_vv_u64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vv_u64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vv_u64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vv_u64m8(op1, op2, vl);
    }
}

// Vector-Scalar Multiply Template
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_MUL_VX(VecType op1, T op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfmul_vf_f32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfmul_vf_f32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmul_vf_f32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmul_vf_f32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmul_vf_f32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfmul_vf_f64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmul_vf_f64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmul_vf_f64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmul_vf_f64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vmul_vx_i8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vmul_vx_i8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vx_i8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vx_i8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_i8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_i8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_i8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vmul_vx_i16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vx_i16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vx_i16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_i16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_i16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_i16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vmul_vx_i32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vx_i32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_i32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_i32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_i32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmul_vx_i64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_i64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_i64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_i64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vmul_vx_u8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vmul_vx_u8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vx_u8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vx_u8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_u8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_u8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_u8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vmul_vx_u16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmul_vx_u16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vx_u16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_u16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_u16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_u16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vmul_vx_u32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmul_vx_u32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_u32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_u32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_u32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmul_vx_u64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmul_vx_u64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmul_vx_u64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmul_vx_u64m8(op1, op2, vl);
    }
}

template<typename T, int LMUL, typename VecType, typename Op2Type>
inline auto VECTOR_MULTIPLY(VecType op1, Op2Type op2, size_t vl) {
    // If op2 is a scalar type (arithmetic type), use vector-scalar multiplication
    if constexpr (std::is_arithmetic_v<Op2Type>) {
        return VECTOR_MUL_VX<T, LMUL>(op1, static_cast<T>(op2), vl);
    }
    // Otherwise, use vector-vector multiplication
    else {
        return VECTOR_MUL_VV<T, LMUL>(op1, op2, vl);
    }
}

#endif // RVV_ARITHMETIC_HPP