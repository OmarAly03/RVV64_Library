#ifndef RVV_MINMAX_HPP
#define RVV_MINMAX_HPP

#include <cstddef>
#include <riscv_vector.h>
#include <type_traits>

/*
vint8mf8_t __riscv_vmax_vv_i8mf8 (vint8mf8_t op1, vint8mf8_t op2, size_t vl);
vint8mf8_t __riscv_vmax_vx_i8mf8 (vint8mf8_t op1, int8_t op2, size_t vl);
vint8mf4_t __riscv_vmax_vv_i8mf4 (vint8mf4_t op1, vint8mf4_t op2, size_t vl);
vint8mf4_t __riscv_vmax_vx_i8mf4 (vint8mf4_t op1, int8_t op2, size_t vl);
vint8mf2_t __riscv_vmax_vv_i8mf2 (vint8mf2_t op1, vint8mf2_t op2, size_t vl);
vint8mf2_t __riscv_vmax_vx_i8mf2 (vint8mf2_t op1, int8_t op2, size_t vl);
vint8m1_t __riscv_vmax_vv_i8m1 (vint8m1_t op1, vint8m1_t op2, size_t vl);
vint8m1_t __riscv_vmax_vx_i8m1 (vint8m1_t op1, int8_t op2, size_t vl);
vint8m2_t __riscv_vmax_vv_i8m2 (vint8m2_t op1, vint8m2_t op2, size_t vl);
vint8m2_t __riscv_vmax_vx_i8m2 (vint8m2_t op1, int8_t op2, size_t vl);
vint8m4_t __riscv_vmax_vv_i8m4 (vint8m4_t op1, vint8m4_t op2, size_t vl);
vint8m4_t __riscv_vmax_vx_i8m4 (vint8m4_t op1, int8_t op2, size_t vl);
vint8m8_t __riscv_vmax_vv_i8m8 (vint8m8_t op1, vint8m8_t op2, size_t vl);
vint8m8_t __riscv_vmax_vx_i8m8 (vint8m8_t op1, int8_t op2, size_t vl);
vint16mf4_t __riscv_vmax_vv_i16mf4 (vint16mf4_t op1, vint16mf4_t op2, size_t vl);
vint16mf4_t __riscv_vmax_vx_i16mf4 (vint16mf4_t op1, int16_t op2, size_t vl);
vint16mf2_t __riscv_vmax_vv_i16mf2 (vint16mf2_t op1, vint16mf2_t op2, size_t vl);
vint16mf2_t __riscv_vmax_vx_i16mf2 (vint16mf2_t op1, int16_t op2, size_t vl);
vint16m1_t __riscv_vmax_vv_i16m1 (vint16m1_t op1, vint16m1_t op2, size_t vl);
vint16m1_t __riscv_vmax_vx_i16m1 (vint16m1_t op1, int16_t op2, size_t vl);
vint16m2_t __riscv_vmax_vv_i16m2 (vint16m2_t op1, vint16m2_t op2, size_t vl);
vint16m2_t __riscv_vmax_vx_i16m2 (vint16m2_t op1, int16_t op2, size_t vl);
vint16m4_t __riscv_vmax_vv_i16m4 (vint16m4_t op1, vint16m4_t op2, size_t vl);
vint16m4_t __riscv_vmax_vx_i16m4 (vint16m4_t op1, int16_t op2, size_t vl);
vint16m8_t __riscv_vmax_vv_i16m8 (vint16m8_t op1, vint16m8_t op2, size_t vl);
vint16m8_t __riscv_vmax_vx_i16m8 (vint16m8_t op1, int16_t op2, size_t vl);
vint32mf2_t __riscv_vmax_vv_i32mf2 (vint32mf2_t op1, vint32mf2_t op2, size_t vl);
vint32mf2_t __riscv_vmax_vx_i32mf2 (vint32mf2_t op1, int32_t op2, size_t vl);
vint32m1_t __riscv_vmax_vv_i32m1 (vint32m1_t op1, vint32m1_t op2, size_t vl);
vint32m1_t __riscv_vmax_vx_i32m1 (vint32m1_t op1, int32_t op2, size_t vl);
vint32m2_t __riscv_vmax_vv_i32m2 (vint32m2_t op1, vint32m2_t op2, size_t vl);
vint32m2_t __riscv_vmax_vx_i32m2 (vint32m2_t op1, int32_t op2, size_t vl);
vint32m4_t __riscv_vmax_vv_i32m4 (vint32m4_t op1, vint32m4_t op2, size_t vl);
vint32m4_t __riscv_vmax_vx_i32m4 (vint32m4_t op1, int32_t op2, size_t vl);
vint32m8_t __riscv_vmax_vv_i32m8 (vint32m8_t op1, vint32m8_t op2, size_t vl);
vint32m8_t __riscv_vmax_vx_i32m8 (vint32m8_t op1, int32_t op2, size_t vl);
vint64m1_t __riscv_vmax_vv_i64m1 (vint64m1_t op1, vint64m1_t op2, size_t vl);
vint64m1_t __riscv_vmax_vx_i64m1 (vint64m1_t op1, int64_t op2, size_t vl);
vint64m2_t __riscv_vmax_vv_i64m2 (vint64m2_t op1, vint64m2_t op2, size_t vl);
vint64m2_t __riscv_vmax_vx_i64m2 (vint64m2_t op1, int64_t op2, size_t vl);
vint64m4_t __riscv_vmax_vv_i64m4 (vint64m4_t op1, vint64m4_t op2, size_t vl);
vint64m4_t __riscv_vmax_vx_i64m4 (vint64m4_t op1, int64_t op2, size_t vl);
vint64m8_t __riscv_vmax_vv_i64m8 (vint64m8_t op1, vint64m8_t op2, size_t vl);
vint64m8_t __riscv_vmax_vx_i64m8 (vint64m8_t op1, int64_t op2, size_t vl);

vfloat16mf4_t __riscv_vfmax_vv_f16mf4 (vfloat16mf4_t op1, vfloat16mf4_t op2, size_t vl);
vfloat16mf4_t __riscv_vfmax_vf_f16mf4 (vfloat16mf4_t op1, float16_t op2, size_t vl);
vfloat16mf2_t __riscv_vfmax_vv_f16mf2 (vfloat16mf2_t op1, vfloat16mf2_t op2, size_t vl);
vfloat16mf2_t __riscv_vfmax_vf_f16mf2 (vfloat16mf2_t op1, float16_t op2, size_t vl);
vfloat16m1_t __riscv_vfmax_vv_f16m1 (vfloat16m1_t op1, vfloat16m1_t op2, size_t vl);
vfloat16m1_t __riscv_vfmax_vf_f16m1 (vfloat16m1_t op1, float16_t op2, size_t vl);
vfloat16m2_t __riscv_vfmax_vv_f16m2 (vfloat16m2_t op1, vfloat16m2_t op2, size_t vl);
vfloat16m2_t __riscv_vfmax_vf_f16m2 (vfloat16m2_t op1, float16_t op2, size_t vl);
vfloat16m4_t __riscv_vfmax_vv_f16m4 (vfloat16m4_t op1, vfloat16m4_t op2, size_t vl);
vfloat16m4_t __riscv_vfmax_vf_f16m4 (vfloat16m4_t op1, float16_t op2, size_t vl);
vfloat16m8_t __riscv_vfmax_vv_f16m8 (vfloat16m8_t op1, vfloat16m8_t op2, size_t vl);
vfloat16m8_t __riscv_vfmax_vf_f16m8 (vfloat16m8_t op1, float16_t op2, size_t vl);
vfloat32mf2_t __riscv_vfmax_vv_f32mf2 (vfloat32mf2_t op1, vfloat32mf2_t op2, size_t vl);
vfloat32mf2_t __riscv_vfmax_vf_f32mf2 (vfloat32mf2_t op1, float32_t op2, size_t vl);
vfloat32m1_t __riscv_vfmax_vv_f32m1 (vfloat32m1_t op1, vfloat32m1_t op2, size_t vl);
vfloat32m1_t __riscv_vfmax_vf_f32m1 (vfloat32m1_t op1, float32_t op2, size_t vl);
vfloat32m2_t __riscv_vfmax_vv_f32m2 (vfloat32m2_t op1, vfloat32m2_t op2, size_t vl);
vfloat32m2_t __riscv_vfmax_vf_f32m2 (vfloat32m2_t op1, float32_t op2, size_t vl);
vfloat32m4_t __riscv_vfmax_vv_f32m4 (vfloat32m4_t op1, vfloat32m4_t op2, size_t vl);
vfloat32m4_t __riscv_vfmax_vf_f32m4 (vfloat32m4_t op1, float32_t op2, size_t vl);
vfloat32m8_t __riscv_vfmax_vv_f32m8 (vfloat32m8_t op1, vfloat32m8_t op2, size_t vl);
vfloat32m8_t __riscv_vfmax_vf_f32m8 (vfloat32m8_t op1, float32_t op2, size_t vl);
vfloat64m1_t __riscv_vfmax_vv_f64m1 (vfloat64m1_t op1, vfloat64m1_t op2, size_t vl);
vfloat64m1_t __riscv_vfmax_vf_f64m1 (vfloat64m1_t op1, float64_t op2, size_t vl);
vfloat64m2_t __riscv_vfmax_vv_f64m2 (vfloat64m2_t op1, vfloat64m2_t op2, size_t vl);
vfloat64m2_t __riscv_vfmax_vf_f64m2 (vfloat64m2_t op1, float64_t op2, size_t vl);
vfloat64m4_t __riscv_vfmax_vv_f64m4 (vfloat64m4_t op1, vfloat64m4_t op2, size_t vl);
vfloat64m4_t __riscv_vfmax_vf_f64m4 (vfloat64m4_t op1, float64_t op2, size_t vl);
vfloat64m8_t __riscv_vfmax_vv_f64m8 (vfloat64m8_t op1, vfloat64m8_t op2, size_t vl);
vfloat64m8_t __riscv_vfmax_vf_f64m8 (vfloat64m8_t op1, float64_t op2, size_t vl);
*/

/*
vint8mf8_t __riscv_vmin_vv_i8mf8 (vint8mf8_t op1, vint8mf8_t op2, size_t vl);
vint8mf8_t __riscv_vmin_vx_i8mf8 (vint8mf8_t op1, int8_t op2, size_t vl);
vint8mf4_t __riscv_vmin_vv_i8mf4 (vint8mf4_t op1, vint8mf4_t op2, size_t vl);
vint8mf4_t __riscv_vmin_vx_i8mf4 (vint8mf4_t op1, int8_t op2, size_t vl);
vint8mf2_t __riscv_vmin_vv_i8mf2 (vint8mf2_t op1, vint8mf2_t op2, size_t vl);
vint8mf2_t __riscv_vmin_vx_i8mf2 (vint8mf2_t op1, int8_t op2, size_t vl);
vint8m1_t __riscv_vmin_vv_i8m1 (vint8m1_t op1, vint8m1_t op2, size_t vl);
vint8m1_t __riscv_vmin_vx_i8m1 (vint8m1_t op1, int8_t op2, size_t vl);
vint8m2_t __riscv_vmin_vv_i8m2 (vint8m2_t op1, vint8m2_t op2, size_t vl);
vint8m2_t __riscv_vmin_vx_i8m2 (vint8m2_t op1, int8_t op2, size_t vl);
vint8m4_t __riscv_vmin_vv_i8m4 (vint8m4_t op1, vint8m4_t op2, size_t vl);
vint8m4_t __riscv_vmin_vx_i8m4 (vint8m4_t op1, int8_t op2, size_t vl);
vint8m8_t __riscv_vmin_vv_i8m8 (vint8m8_t op1, vint8m8_t op2, size_t vl);
vint8m8_t __riscv_vmin_vx_i8m8 (vint8m8_t op1, int8_t op2, size_t vl);
vint16mf4_t __riscv_vmin_vv_i16mf4 (vint16mf4_t op1, vint16mf4_t op2, size_t vl);
vint16mf4_t __riscv_vmin_vx_i16mf4 (vint16mf4_t op1, int16_t op2, size_t vl);
vint16mf2_t __riscv_vmin_vv_i16mf2 (vint16mf2_t op1, vint16mf2_t op2, size_t vl);
vint16mf2_t __riscv_vmin_vx_i16mf2 (vint16mf2_t op1, int16_t op2, size_t vl);
vint16m1_t __riscv_vmin_vv_i16m1 (vint16m1_t op1, vint16m1_t op2, size_t vl);
vint16m1_t __riscv_vmin_vx_i16m1 (vint16m1_t op1, int16_t op2, size_t vl);
vint16m2_t __riscv_vmin_vv_i16m2 (vint16m2_t op1, vint16m2_t op2, size_t vl);
vint16m2_t __riscv_vmin_vx_i16m2 (vint16m2_t op1, int16_t op2, size_t vl);
vint16m4_t __riscv_vmin_vv_i16m4 (vint16m4_t op1, vint16m4_t op2, size_t vl);
vint16m4_t __riscv_vmin_vx_i16m4 (vint16m4_t op1, int16_t op2, size_t vl);
vint16m8_t __riscv_vmin_vv_i16m8 (vint16m8_t op1, vint16m8_t op2, size_t vl);
vint16m8_t __riscv_vmin_vx_i16m8 (vint16m8_t op1, int16_t op2, size_t vl);
vint32mf2_t __riscv_vmin_vv_i32mf2 (vint32mf2_t op1, vint32mf2_t op2, size_t vl);
vint32mf2_t __riscv_vmin_vx_i32mf2 (vint32mf2_t op1, int32_t op2, size_t vl);
vint32m1_t __riscv_vmin_vv_i32m1 (vint32m1_t op1, vint32m1_t op2, size_t vl);
vint32m1_t __riscv_vmin_vx_i32m1 (vint32m1_t op1, int32_t op2, size_t vl);
vint32m2_t __riscv_vmin_vv_i32m2 (vint32m2_t op1, vint32m2_t op2, size_t vl);
vint32m2_t __riscv_vmin_vx_i32m2 (vint32m2_t op1, int32_t op2, size_t vl);
vint32m4_t __riscv_vmin_vv_i32m4 (vint32m4_t op1, vint32m4_t op2, size_t vl);
vint32m4_t __riscv_vmin_vx_i32m4 (vint32m4_t op1, int32_t op2, size_t vl);
vint32m8_t __riscv_vmin_vv_i32m8 (vint32m8_t op1, vint32m8_t op2, size_t vl);
vint32m8_t __riscv_vmin_vx_i32m8 (vint32m8_t op1, int32_t op2, size_t vl);
vint64m1_t __riscv_vmin_vv_i64m1 (vint64m1_t op1, vint64m1_t op2, size_t vl);
vint64m1_t __riscv_vmin_vx_i64m1 (vint64m1_t op1, int64_t op2, size_t vl);
vint64m2_t __riscv_vmin_vv_i64m2 (vint64m2_t op1, vint64m2_t op2, size_t vl);
vint64m2_t __riscv_vmin_vx_i64m2 (vint64m2_t op1, int64_t op2, size_t vl);
vint64m4_t __riscv_vmin_vv_i64m4 (vint64m4_t op1, vint64m4_t op2, size_t vl);
vint64m4_t __riscv_vmin_vx_i64m4 (vint64m4_t op1, int64_t op2, size_t vl);
vint64m8_t __riscv_vmin_vv_i64m8 (vint64m8_t op1, vint64m8_t op2, size_t vl);
vint64m8_t __riscv_vmin_vx_i64m8 (vint64m8_t op1, int64_t op2, size_t vl);

vfloat16mf4_t __riscv_vfmin_vv_f16mf4 (vfloat16mf4_t op1, vfloat16mf4_t op2, size_t vl);
vfloat16mf4_t __riscv_vfmin_vf_f16mf4 (vfloat16mf4_t op1, float16_t op2, size_t vl);
vfloat16mf2_t __riscv_vfmin_vv_f16mf2 (vfloat16mf2_t op1, vfloat16mf2_t op2, size_t vl);
vfloat16mf2_t __riscv_vfmin_vf_f16mf2 (vfloat16mf2_t op1, float16_t op2, size_t vl);
vfloat16m1_t __riscv_vfmin_vv_f16m1 (vfloat16m1_t op1, vfloat16m1_t op2, size_t vl);
vfloat16m1_t __riscv_vfmin_vf_f16m1 (vfloat16m1_t op1, float16_t op2, size_t vl);
vfloat16m2_t __riscv_vfmin_vv_f16m2 (vfloat16m2_t op1, vfloat16m2_t op2, size_t vl);
vfloat16m2_t __riscv_vfmin_vf_f16m2 (vfloat16m2_t op1, float16_t op2, size_t vl);
vfloat16m4_t __riscv_vfmin_vv_f16m4 (vfloat16m4_t op1, vfloat16m4_t op2, size_t vl);
vfloat16m4_t __riscv_vfmin_vf_f16m4 (vfloat16m4_t op1, float16_t op2, size_t vl);
vfloat16m8_t __riscv_vfmin_vv_f16m8 (vfloat16m8_t op1, vfloat16m8_t op2, size_t vl);
vfloat16m8_t __riscv_vfmin_vf_f16m8 (vfloat16m8_t op1, float16_t op2, size_t vl);
vfloat32mf2_t __riscv_vfmin_vv_f32mf2 (vfloat32mf2_t op1, vfloat32mf2_t op2, size_t vl);
vfloat32mf2_t __riscv_vfmin_vf_f32mf2 (vfloat32mf2_t op1, float32_t op2, size_t vl);
vfloat32m1_t __riscv_vfmin_vv_f32m1 (vfloat32m1_t op1, vfloat32m1_t op2, size_t vl);
vfloat32m1_t __riscv_vfmin_vf_f32m1 (vfloat32m1_t op1, float32_t op2, size_t vl);
vfloat32m2_t __riscv_vfmin_vv_f32m2 (vfloat32m2_t op1, vfloat32m2_t op2, size_t vl);
vfloat32m2_t __riscv_vfmin_vf_f32m2 (vfloat32m2_t op1, float32_t op2, size_t vl);
vfloat32m4_t __riscv_vfmin_vv_f32m4 (vfloat32m4_t op1, vfloat32m4_t op2, size_t vl);
vfloat32m4_t __riscv_vfmin_vf_f32m4 (vfloat32m4_t op1, float32_t op2, size_t vl);
vfloat32m8_t __riscv_vfmin_vv_f32m8 (vfloat32m8_t op1, vfloat32m8_t op2, size_t vl);
vfloat32m8_t __riscv_vfmin_vf_f32m8 (vfloat32m8_t op1, float32_t op2, size_t vl);
vfloat64m1_t __riscv_vfmin_vv_f64m1 (vfloat64m1_t op1, vfloat64m1_t op2, size_t vl);
vfloat64m1_t __riscv_vfmin_vf_f64m1 (vfloat64m1_t op1, float64_t op2, size_t vl);
vfloat64m2_t __riscv_vfmin_vv_f64m2 (vfloat64m2_t op1, vfloat64m2_t op2, size_t vl);
vfloat64m2_t __riscv_vfmin_vf_f64m2 (vfloat64m2_t op1, float64_t op2, size_t vl);
vfloat64m4_t __riscv_vfmin_vv_f64m4 (vfloat64m4_t op1, vfloat64m4_t op2, size_t vl);
vfloat64m4_t __riscv_vfmin_vf_f64m4 (vfloat64m4_t op1, float64_t op2, size_t vl);
vfloat64m8_t __riscv_vfmin_vv_f64m8 (vfloat64m8_t op1, vfloat64m8_t op2, size_t vl);
vfloat64m8_t __riscv_vfmin_vf_f64m8 (vfloat64m8_t op1, float64_t op2, size_t vl);

vint8mf8_t __riscv_vmax_vv_i8mf8_m (vbool64_t mask, vint8mf8_t op1, vint8mf8_t op2, size_t vl);
vint8mf8_t __riscv_vmax_vx_i8mf8_m (vbool64_t mask, vint8mf8_t op1, int8_t op2, size_t vl);
vint8mf4_t __riscv_vmax_vv_i8mf4_m (vbool32_t mask, vint8mf4_t op1, vint8mf4_t op2, size_t vl);
vint8mf4_t __riscv_vmax_vx_i8mf4_m (vbool32_t mask, vint8mf4_t op1, int8_t op2, size_t vl);
vint8mf2_t __riscv_vmax_vv_i8mf2_m (vbool16_t mask, vint8mf2_t op1, vint8mf2_t op2, size_t vl);
vint8mf2_t __riscv_vmax_vx_i8mf2_m (vbool16_t mask, vint8mf2_t op1, int8_t op2, size_t vl);
vint8m1_t __riscv_vmax_vv_i8m1_m (vbool8_t mask, vint8m1_t op1, vint8m1_t op2, size_t vl);
vint8m1_t __riscv_vmax_vx_i8m1_m (vbool8_t mask, vint8m1_t op1, int8_t op2, size_t vl);
vint8m2_t __riscv_vmax_vv_i8m2_m (vbool4_t mask, vint8m2_t op1, vint8m2_t op2, size_t vl);
vint8m2_t __riscv_vmax_vx_i8m2_m (vbool4_t mask, vint8m2_t op1, int8_t op2, size_t vl);
vint8m4_t __riscv_vmax_vv_i8m4_m (vbool2_t mask, vint8m4_t op1, vint8m4_t op2, size_t vl);
vint8m4_t __riscv_vmax_vx_i8m4_m (vbool2_t mask, vint8m4_t op1, int8_t op2, size_t vl);
vint8m8_t __riscv_vmax_vv_i8m8_m (vbool1_t mask, vint8m8_t op1, vint8m8_t op2, size_t vl);
vint8m8_t __riscv_vmax_vx_i8m8_m (vbool1_t mask, vint8m8_t op1, int8_t op2, size_t vl);
vint16mf4_t __riscv_vmax_vv_i16mf4_m (vbool64_t mask, vint16mf4_t op1, vint16mf4_t op2, size_t vl);
vint16mf4_t __riscv_vmax_vx_i16mf4_m (vbool64_t mask, vint16mf4_t op1, int16_t op2, size_t vl);
vint16mf2_t __riscv_vmax_vv_i16mf2_m (vbool32_t mask, vint16mf2_t op1, vint16mf2_t op2, size_t vl);
vint16mf2_t __riscv_vmax_vx_i16mf2_m (vbool32_t mask, vint16mf2_t op1, int16_t op2, size_t vl);
vint16m1_t __riscv_vmax_vv_i16m1_m (vbool16_t mask, vint16m1_t op1, vint16m1_t op2, size_t vl);
vint16m1_t __riscv_vmax_vx_i16m1_m (vbool16_t mask, vint16m1_t op1, int16_t op2, size_t vl);
vint16m2_t __riscv_vmax_vv_i16m2_m (vbool8_t mask, vint16m2_t op1, vint16m2_t op2, size_t vl);
vint16m2_t __riscv_vmax_vx_i16m2_m (vbool8_t mask, vint16m2_t op1, int16_t op2, size_t vl);
vint16m4_t __riscv_vmax_vv_i16m4_m (vbool4_t mask, vint16m4_t op1, vint16m4_t op2, size_t vl);
vint16m4_t __riscv_vmax_vx_i16m4_m (vbool4_t mask, vint16m4_t op1, int16_t op2, size_t vl);
vint16m8_t __riscv_vmax_vv_i16m8_m (vbool2_t mask, vint16m8_t op1, vint16m8_t op2, size_t vl);
vint16m8_t __riscv_vmax_vx_i16m8_m (vbool2_t mask, vint16m8_t op1, int16_t op2, size_t vl);
vint32mf2_t __riscv_vmax_vv_i32mf2_m (vbool64_t mask, vint32mf2_t op1, vint32mf2_t op2, size_t vl);
vint32mf2_t __riscv_vmax_vx_i32mf2_m (vbool64_t mask, vint32mf2_t op1, int32_t op2, size_t vl);
vint32m1_t __riscv_vmax_vv_i32m1_m (vbool32_t mask, vint32m1_t op1, vint32m1_t op2, size_t vl);
vint32m1_t __riscv_vmax_vx_i32m1_m (vbool32_t mask, vint32m1_t op1, int32_t op2, size_t vl);
vint32m2_t __riscv_vmax_vv_i32m2_m (vbool16_t mask, vint32m2_t op1, vint32m2_t op2, size_t vl);
vint32m2_t __riscv_vmax_vx_i32m2_m (vbool16_t mask, vint32m2_t op1, int32_t op2, size_t vl);
vint32m4_t __riscv_vmax_vv_i32m4_m (vbool8_t mask, vint32m4_t op1, vint32m4_t op2, size_t vl);
vint32m4_t __riscv_vmax_vx_i32m4_m (vbool8_t mask, vint32m4_t op1, int32_t op2, size_t vl);
vint32m8_t __riscv_vmax_vv_i32m8_m (vbool4_t mask, vint32m8_t op1, vint32m8_t op2, size_t vl);
vint32m8_t __riscv_vmax_vx_i32m8_m (vbool4_t mask, vint32m8_t op1, int32_t op2, size_t vl);
vint64m1_t __riscv_vmax_vv_i64m1_m (vbool64_t mask, vint64m1_t op1, vint64m1_t op2, size_t vl);
vint64m1_t __riscv_vmax_vx_i64m1_m (vbool64_t mask, vint64m1_t op1, int64_t op2, size_t vl);
vint64m2_t __riscv_vmax_vv_i64m2_m (vbool32_t mask, vint64m2_t op1, vint64m2_t op2, size_t vl);
vint64m2_t __riscv_vmax_vx_i64m2_m (vbool32_t mask, vint64m2_t op1, int64_t op2, size_t vl);
vint64m4_t __riscv_vmax_vv_i64m4_m (vbool16_t mask, vint64m4_t op1, vint64m4_t op2, size_t vl);
vint64m4_t __riscv_vmax_vx_i64m4_m (vbool16_t mask, vint64m4_t op1, int64_t op2, size_t vl);
vint64m8_t __riscv_vmax_vv_i64m8_m (vbool8_t mask, vint64m8_t op1, vint64m8_t op2, size_t vl);
vint64m8_t __riscv_vmax_vx_i64m8_m (vbool8_t mask, vint64m8_t op1, int64_t op2, size_t vl);
*/
// Vector-Vector Max Template
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_MAX_VV(VecType op1, VecType op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfmax_vv_f32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfmax_vv_f32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmax_vv_f32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmax_vv_f32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmax_vv_f32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfmax_vv_f64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmax_vv_f64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmax_vv_f64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmax_vv_f64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vmax_vv_i8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vmax_vv_i8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmax_vv_i8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmax_vv_i8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmax_vv_i8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmax_vv_i8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmax_vv_i8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vmax_vv_i16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmax_vv_i16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmax_vv_i16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmax_vv_i16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmax_vv_i16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmax_vv_i16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vmax_vv_i32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmax_vv_i32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmax_vv_i32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmax_vv_i32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmax_vv_i32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmax_vv_i64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmax_vv_i64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmax_vv_i64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmax_vv_i64m8(op1, op2, vl);
    }
}

// Vector-Scalar Max Template
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_MAX_VX(VecType op1, T op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfmax_vf_f32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfmax_vf_f32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmax_vf_f32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmax_vf_f32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmax_vf_f32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfmax_vf_f64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmax_vf_f64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmax_vf_f64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmax_vf_f64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vmax_vx_i8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vmax_vx_i8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmax_vx_i8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmax_vx_i8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmax_vx_i8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmax_vx_i8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmax_vx_i8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vmax_vx_i16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmax_vx_i16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmax_vx_i16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmax_vx_i16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmax_vx_i16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmax_vx_i16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vmax_vx_i32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmax_vx_i32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmax_vx_i32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmax_vx_i32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmax_vx_i32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmax_vx_i64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmax_vx_i64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmax_vx_i64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmax_vx_i64m8(op1, op2, vl);
    }
}

// Unified VECTOR_MAX template that auto-detects vector vs scalar
template<typename T, int LMUL, typename VecType, typename Op2Type>
inline auto VECTOR_MAX(VecType op1, Op2Type op2, size_t vl) {
    // If op2 is a scalar type, use vector-scalar max
    if constexpr (std::is_same_v<Op2Type, T>) {
        return VECTOR_MAX_VX<T, LMUL>(op1, op2, vl);
    }
    // Otherwise, use vector-vector max
    else {
        return VECTOR_MAX_VV<T, LMUL>(op1, op2, vl);
    }
}

// Vector-Vector Min Template
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_MIN_VV(VecType op1, VecType op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfmin_vv_f32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfmin_vv_f32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmin_vv_f32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmin_vv_f32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmin_vv_f32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfmin_vv_f64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmin_vv_f64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmin_vv_f64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmin_vv_f64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vmin_vv_i8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vmin_vv_i8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmin_vv_i8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmin_vv_i8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmin_vv_i8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmin_vv_i8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmin_vv_i8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vmin_vv_i16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmin_vv_i16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmin_vv_i16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmin_vv_i16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmin_vv_i16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmin_vv_i16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vmin_vv_i32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmin_vv_i32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmin_vv_i32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmin_vv_i32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmin_vv_i32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmin_vv_i64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmin_vv_i64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmin_vv_i64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmin_vv_i64m8(op1, op2, vl);
    }
}

// Vector-Scalar Min Template
template<typename T, int LMUL, typename VecType>
inline auto VECTOR_MIN_VX(VecType op1, T op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfmin_vf_f32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfmin_vf_f32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmin_vf_f32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmin_vf_f32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmin_vf_f32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfmin_vf_f64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmin_vf_f64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmin_vf_f64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmin_vf_f64m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vmin_vx_i8mf8(op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vmin_vx_i8mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmin_vx_i8mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmin_vx_i8m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmin_vx_i8m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmin_vx_i8m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmin_vx_i8m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vmin_vx_i16mf4(op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmin_vx_i16mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmin_vx_i16m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmin_vx_i16m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmin_vx_i16m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmin_vx_i16m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vmin_vx_i32mf2(op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmin_vx_i32m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmin_vx_i32m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmin_vx_i32m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmin_vx_i32m8(op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmin_vx_i64m1(op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmin_vx_i64m2(op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmin_vx_i64m4(op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmin_vx_i64m8(op1, op2, vl);
    }
}

// Unified VECTOR_MIN template that auto-detects vector vs scalar
template<typename T, int LMUL, typename VecType, typename Op2Type>
inline auto VECTOR_MIN(VecType op1, Op2Type op2, size_t vl) {
    // If op2 is a scalar type, use vector-scalar min
    if constexpr (std::is_same_v<Op2Type, T>) {
        return VECTOR_MIN_VX<T, LMUL>(op1, op2, vl);
    }
    // Otherwise, use vector-vector min
    else {
        return VECTOR_MIN_VV<T, LMUL>(op1, op2, vl);
    }
}


// Vector-Vector Max Masked Template
template<typename T, int LMUL, typename MaskType, typename VecType>
inline auto VECTOR_MAX_VV_MASKED(MaskType mask, VecType op1, VecType op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfmax_vv_f32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfmax_vv_f32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmax_vv_f32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmax_vv_f32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmax_vv_f32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfmax_vv_f64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmax_vv_f64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmax_vv_f64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmax_vv_f64m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vmax_vv_i8mf8_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vmax_vv_i8mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmax_vv_i8mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmax_vv_i8m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmax_vv_i8m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmax_vv_i8m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmax_vv_i8m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vmax_vv_i16mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmax_vv_i16mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmax_vv_i16m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmax_vv_i16m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmax_vv_i16m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmax_vv_i16m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vmax_vv_i32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmax_vv_i32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmax_vv_i32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmax_vv_i32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmax_vv_i32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmax_vv_i64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmax_vv_i64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmax_vv_i64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmax_vv_i64m8_m(mask, op1, op2, vl);
    }
}

// Vector-Scalar Max Masked Template
template<typename T, int LMUL, typename MaskType, typename VecType>
inline auto VECTOR_MAX_VX_MASKED(MaskType mask, VecType op1, T op2, size_t vl) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (LMUL == MF2) return __riscv_vfmax_vf_f32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vfmax_vf_f32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmax_vf_f32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmax_vf_f32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmax_vf_f32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, double>) {
        if constexpr (LMUL == M1) return __riscv_vfmax_vf_f64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vfmax_vf_f64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vfmax_vf_f64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vfmax_vf_f64m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        if constexpr (LMUL == MF8) return __riscv_vmax_vx_i8mf8_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF4) return __riscv_vmax_vx_i8mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmax_vx_i8mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmax_vx_i8m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmax_vx_i8m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmax_vx_i8m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmax_vx_i8m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        if constexpr (LMUL == MF4) return __riscv_vmax_vx_i16mf4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == MF2) return __riscv_vmax_vx_i16mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmax_vx_i16m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmax_vx_i16m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmax_vx_i16m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmax_vx_i16m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        if constexpr (LMUL == MF2) return __riscv_vmax_vx_i32mf2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M1) return __riscv_vmax_vx_i32m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmax_vx_i32m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmax_vx_i32m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmax_vx_i32m8_m(mask, op1, op2, vl);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        if constexpr (LMUL == M1) return __riscv_vmax_vx_i64m1_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M2) return __riscv_vmax_vx_i64m2_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M4) return __riscv_vmax_vx_i64m4_m(mask, op1, op2, vl);
        else if constexpr (LMUL == M8) return __riscv_vmax_vx_i64m8_m(mask, op1, op2, vl);
    }
}

// Unified VECTOR_MAX_MASKED template that auto-detects vector vs scalar
template<typename T, int LMUL, typename MaskType, typename VecType, typename Op2Type>
inline auto VECTOR_MAX_MASKED(MaskType mask, VecType op1, Op2Type op2, size_t vl) {
    // If op2 is a scalar type, use vector-scalar masked max
    if constexpr (std::is_same_v<Op2Type, T>) {
        return VECTOR_MAX_VX_MASKED<T, LMUL>(mask, op1, op2, vl);
    }
    // Otherwise, use vector-vector masked max
    else {
        return VECTOR_MAX_VV_MASKED<T, LMUL>(mask, op1, op2, vl);
    }
}

#endif // RVV_MINMAX_HPP