# Arithmetic Operations

## VECTOR_SUB

### vfsub_vv
**Vector-Vector Floating-Point Subtraction**

Performs element-wise subtraction of two floating-point vectors: `result[i] = op1[i] - op2[i]`

```c
vfloat16mf4_t __riscv_vfsub_vv_f16mf4 (vfloat16mf4_t op1, vfloat16mf4_t op2, size_t vl);
vfloat16mf2_t __riscv_vfsub_vv_f16mf2 (vfloat16mf2_t op1, vfloat16mf2_t op2, size_t vl);
vfloat16m1_t __riscv_vfsub_vv_f16m1 (vfloat16m1_t op1, vfloat16m1_t op2, size_t vl);
vfloat16m2_t __riscv_vfsub_vv_f16m2 (vfloat16m2_t op1, vfloat16m2_t op2, size_t vl);
vfloat16m4_t __riscv_vfsub_vv_f16m4 (vfloat16m4_t op1, vfloat16m4_t op2, size_t vl);
vfloat16m8_t __riscv_vfsub_vv_f16m8 (vfloat16m8_t op1, vfloat16m8_t op2, size_t vl);
vfloat32mf2_t __riscv_vfsub_vv_f32mf2 (vfloat32mf2_t op1, vfloat32mf2_t op2, size_t vl);
vfloat32m1_t __riscv_vfsub_vv_f32m1 (vfloat32m1_t op1, vfloat32m1_t op2, size_t vl);
vfloat32m2_t __riscv_vfsub_vv_f32m2 (vfloat32m2_t op1, vfloat32m2_t op2, size_t vl);
vfloat32m4_t __riscv_vfsub_vv_f32m4 (vfloat32m4_t op1, vfloat32m4_t op2, size_t vl);
vfloat32m8_t __riscv_vfsub_vv_f32m8 (vfloat32m8_t op1, vfloat32m8_t op2, size_t vl);
vfloat64m1_t __riscv_vfsub_vv_f64m1 (vfloat64m1_t op1, vfloat64m1_t op2, size_t vl);
vfloat64m2_t __riscv_vfsub_vv_f64m2 (vfloat64m2_t op1, vfloat64m2_t op2, size_t vl);
vfloat64m4_t __riscv_vfsub_vv_f64m4 (vfloat64m4_t op1, vfloat64m4_t op2, size_t vl);
vfloat64m8_t __riscv_vfsub_vv_f64m8 (vfloat64m8_t op1, vfloat64m8_t op2, size_t vl);
```

**Parameters:**
- `op1`: First floating-point vector (minuend)
- `op2`: Second floating-point vector (subtrahend)  
- `vl`: Vector length (number of elements to process)

**Returns:** Vector containing the element-wise difference `op1 - op2`

**Supported Types:** FP16, FP32, FP64 with LMUL from mf4 to m8

### vfsub_vf
**Vector-Scalar Floating-Point Subtraction**

Performs element-wise subtraction of a scalar from each element of a floating-point vector: `result[i] = op1[i] - op2`

```c
vfloat16mf4_t __riscv_vfsub_vf_f16mf4 (vfloat16mf4_t op1, float16_t op2, size_t vl);
vfloat16mf2_t __riscv_vfsub_vf_f16mf2 (vfloat16mf2_t op1, float16_t op2, size_t vl);
vfloat16m1_t __riscv_vfsub_vf_f16m1 (vfloat16m1_t op1, float16_t op2, size_t vl);
vfloat16m2_t __riscv_vfsub_vf_f16m2 (vfloat16m2_t op1, float16_t op2, size_t vl);
vfloat16m4_t __riscv_vfsub_vf_f16m4 (vfloat16m4_t op1, float16_t op2, size_t vl);
vfloat16m8_t __riscv_vfsub_vf_f16m8 (vfloat16m8_t op1, float16_t op2, size_t vl);
vfloat32mf2_t __riscv_vfsub_vf_f32mf2 (vfloat32mf2_t op1, float32_t op2, size_t vl);
vfloat32m1_t __riscv_vfsub_vf_f32m1 (vfloat32m1_t op1, float32_t op2, size_t vl);
vfloat32m2_t __riscv_vfsub_vf_f32m2 (vfloat32m2_t op1, float32_t op2, size_t vl);
vfloat32m4_t __riscv_vfsub_vf_f32m4 (vfloat32m4_t op1, float32_t op2, size_t vl);
vfloat32m8_t __riscv_vfsub_vf_f32m8 (vfloat32m8_t op1, float32_t op2, size_t vl);
vfloat64m1_t __riscv_vfsub_vf_f64m1 (vfloat64m1_t op1, float64_t op2, size_t vl);
vfloat64m2_t __riscv_vfsub_vf_f64m2 (vfloat64m2_t op1, float64_t op2, size_t vl);
vfloat64m4_t __riscv_vfsub_vf_f64m4 (vfloat64m4_t op1, float64_t op2, size_t vl);
vfloat64m8_t __riscv_vfsub_vf_f64m8 (vfloat64m8_t op1, float64_t op2, size_t vl);
```

**Parameters:**
- `op1`: Floating-point vector (minuend)
- `op2`: Scalar floating-point value (subtrahend)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector containing the element-wise difference `op1[i] - op2`

**Supported Types:** FP16, FP32, FP64 with LMUL from mf4 to m8

### vsub_vv
**Vector-Vector Integer Subtraction**

Performs element-wise subtraction of two integer vectors: `result[i] = op1[i] - op2[i]`

```c
vint8mf8_t __riscv_vsub_vv_i8mf8 (vint8mf8_t op1, vint8mf8_t op2, size_t vl);
vint8mf4_t __riscv_vsub_vv_i8mf4 (vint8mf4_t op1, vint8mf4_t op2, size_t vl);
vint8mf2_t __riscv_vsub_vv_i8mf2 (vint8mf2_t op1, vint8mf2_t op2, size_t vl);
vint8m1_t __riscv_vsub_vv_i8m1 (vint8m1_t op1, vint8m1_t op2, size_t vl);
vint8m2_t __riscv_vsub_vv_i8m2 (vint8m2_t op1, vint8m2_t op2, size_t vl);
vint8m4_t __riscv_vsub_vv_i8m4 (vint8m4_t op1, vint8m4_t op2, size_t vl);
vint8m8_t __riscv_vsub_vv_i8m8 (vint8m8_t op1, vint8m8_t op2, size_t vl);
vint16mf4_t __riscv_vsub_vv_i16mf4 (vint16mf4_t op1, vint16mf4_t op2, size_t vl);
vint16mf2_t __riscv_vsub_vv_i16mf2 (vint16mf2_t op1, vint16mf2_t op2, size_t vl);
vint16m1_t __riscv_vsub_vv_i16m1 (vint16m1_t op1, vint16m1_t op2, size_t vl);
vint16m2_t __riscv_vsub_vv_i16m2 (vint16m2_t op1, vint16m2_t op2, size_t vl);
vint16m4_t __riscv_vsub_vv_i16m4 (vint16m4_t op1, vint16m4_t op2, size_t vl);
vint16m8_t __riscv_vsub_vv_i16m8 (vint16m8_t op1, vint16m8_t op2, size_t vl);
vint32mf2_t __riscv_vsub_vv_i32mf2 (vint32mf2_t op1, vint32mf2_t op2, size_t vl);
vint32m1_t __riscv_vsub_vv_i32m1 (vint32m1_t op1, vint32m1_t op2, size_t vl);
vint32m2_t __riscv_vsub_vv_i32m2 (vint32m2_t op1, vint32m2_t op2, size_t vl);
vint32m4_t __riscv_vsub_vv_i32m4 (vint32m4_t op1, vint32m4_t op2, size_t vl);
vint32m8_t __riscv_vsub_vv_i32m8 (vint32m8_t op1, vint32m8_t op2, size_t vl);
vint64m1_t __riscv_vsub_vv_i64m1 (vint64m1_t op1, vint64m1_t op2, size_t vl);
vint64m2_t __riscv_vsub_vv_i64m2 (vint64m2_t op1, vint64m2_t op2, size_t vl);
vint64m4_t __riscv_vsub_vv_i64m4 (vint64m4_t op1, vint64m4_t op2, size_t vl);
vint64m8_t __riscv_vsub_vv_i64m8 (vint64m8_t op1, vint64m8_t op2, size_t vl);
vuint8mf8_t __riscv_vsub_vv_u8mf8 (vuint8mf8_t op1, vuint8mf8_t op2, size_t vl);
vuint8mf4_t __riscv_vsub_vv_u8mf4 (vuint8mf4_t op1, vuint8mf4_t op2, size_t vl);
vuint8mf2_t __riscv_vsub_vv_u8mf2 (vuint8mf2_t op1, vuint8mf2_t op2, size_t vl);
vuint8m1_t __riscv_vsub_vv_u8m1 (vuint8m1_t op1, vuint8m1_t op2, size_t vl);
vuint8m2_t __riscv_vsub_vv_u8m2 (vuint8m2_t op1, vuint8m2_t op2, size_t vl);
vuint8m4_t __riscv_vsub_vv_u8m4 (vuint8m4_t op1, vuint8m4_t op2, size_t vl);
vuint8m8_t __riscv_vsub_vv_u8m8 (vuint8m8_t op1, vuint8m8_t op2, size_t vl);
vuint16mf4_t __riscv_vsub_vv_u16mf4 (vuint16mf4_t op1, vuint16mf4_t op2, size_t vl);
vuint16mf2_t __riscv_vsub_vv_u16mf2 (vuint16mf2_t op1, vuint16mf2_t op2, size_t vl);
vuint16m1_t __riscv_vsub_vv_u16m1 (vuint16m1_t op1, vuint16m1_t op2, size_t vl);
vuint16m2_t __riscv_vsub_vv_u16m2 (vuint16m2_t op1, vuint16m2_t op2, size_t vl);
vuint16m4_t __riscv_vsub_vv_u16m4 (vuint16m4_t op1, vuint16m4_t op2, size_t vl);
vuint16m8_t __riscv_vsub_vv_u16m8 (vuint16m8_t op1, vuint16m8_t op2, size_t vl);
vuint32mf2_t __riscv_vsub_vv_u32mf2 (vuint32mf2_t op1, vuint32mf2_t op2, size_t vl);
vuint32m1_t __riscv_vsub_vv_u32m1 (vuint32m1_t op1, vuint32m1_t op2, size_t vl);
vuint32m2_t __riscv_vsub_vv_u32m2 (vuint32m2_t op1, vuint32m2_t op2, size_t vl);
vuint32m4_t __riscv_vsub_vv_u32m4 (vuint32m4_t op1, vuint32m4_t op2, size_t vl);
vuint32m8_t __riscv_vsub_vv_u32m8 (vuint32m8_t op1, vuint32m8_t op2, size_t vl);
vuint64m1_t __riscv_vsub_vv_u64m1 (vuint64m1_t op1, vuint64m1_t op2, size_t vl);
vuint64m2_t __riscv_vsub_vv_u64m2 (vuint64m2_t op1, vuint64m2_t op2, size_t vl);
vuint64m4_t __riscv_vsub_vv_u64m4 (vuint64m4_t op1, vuint64m4_t op2, size_t vl);
vuint64m8_t __riscv_vsub_vv_u64m8 (vuint64m8_t op1, vuint64m8_t op2, size_t vl);
```

**Parameters:**
- `op1`: First integer vector (minuend)
- `op2`: Second integer vector (subtrahend)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector containing the element-wise difference `op1 - op2`

**Supported Types:** Signed and unsigned integers (8, 16, 32, 64-bit) with LMUL from mf8 to m8



### vsub_vx
**Vector-Scalar Integer Subtraction**

Performs element-wise subtraction of a scalar from each element of an integer vector: `result[i] = op1[i] - op2`

```c
vint8mf8_t __riscv_vsub_vx_i8mf8 (vint8mf8_t op1, int8_t op2, size_t vl);
vint8mf4_t __riscv_vsub_vx_i8mf4 (vint8mf4_t op1, int8_t op2, size_t vl);
vint8mf2_t __riscv_vsub_vx_i8mf2 (vint8mf2_t op1, int8_t op2, size_t vl);
vint8m1_t __riscv_vsub_vx_i8m1 (vint8m1_t op1, int8_t op2, size_t vl);
vint8m2_t __riscv_vsub_vx_i8m2 (vint8m2_t op1, int8_t op2, size_t vl);
vint8m4_t __riscv_vsub_vx_i8m4 (vint8m4_t op1, int8_t op2, size_t vl);
vint8m8_t __riscv_vsub_vx_i8m8 (vint8m8_t op1, int8_t op2, size_t vl);
vint16mf4_t __riscv_vsub_vx_i16mf4 (vint16mf4_t op1, int16_t op2, size_t vl);
vint16mf2_t __riscv_vsub_vx_i16mf2 (vint16mf2_t op1, int16_t op2, size_t vl);
vint16m1_t __riscv_vsub_vx_i16m1 (vint16m1_t op1, int16_t op2, size_t vl);
vint16m2_t __riscv_vsub_vx_i16m2 (vint16m2_t op1, int16_t op2, size_t vl);
vint16m4_t __riscv_vsub_vx_i16m4 (vint16m4_t op1, int16_t op2, size_t vl);
vint16m8_t __riscv_vsub_vx_i16m8 (vint16m8_t op1, int16_t op2, size_t vl);
vint32mf2_t __riscv_vsub_vx_i32mf2 (vint32mf2_t op1, int32_t op2, size_t vl);
vint32m1_t __riscv_vsub_vx_i32m1 (vint32m1_t op1, int32_t op2, size_t vl);
vint32m2_t __riscv_vsub_vx_i32m2 (vint32m2_t op1, int32_t op2, size_t vl);
vint32m4_t __riscv_vsub_vx_i32m4 (vint32m4_t op1, int32_t op2, size_t vl);
vint32m8_t __riscv_vsub_vx_i32m8 (vint32m8_t op1, int32_t op2, size_t vl);
vint64m1_t __riscv_vsub_vx_i64m1 (vint64m1_t op1, int64_t op2, size_t vl);
vint64m2_t __riscv_vsub_vx_i64m2 (vint64m2_t op1, int64_t op2, size_t vl);
vint64m4_t __riscv_vsub_vx_i64m4 (vint64m4_t op1, int64_t op2, size_t vl);
vint64m8_t __riscv_vsub_vx_i64m8 (vint64m8_t op1, int64_t op2, size_t vl);
vuint8mf8_t __riscv_vsub_vx_u8mf8 (vuint8mf8_t op1, uint8_t op2, size_t vl);
vuint8mf4_t __riscv_vsub_vx_u8mf4 (vuint8mf4_t op1, uint8_t op2, size_t vl);
vuint8mf2_t __riscv_vsub_vx_u8mf2 (vuint8mf2_t op1, uint8_t op2, size_t vl);
vuint8m1_t __riscv_vsub_vx_u8m1 (vuint8m1_t op1, uint8_t op2, size_t vl);
vuint8m2_t __riscv_vsub_vx_u8m2 (vuint8m2_t op1, uint8_t op2, size_t vl);
vuint8m4_t __riscv_vsub_vx_u8m4 (vuint8m4_t op1, uint8_t op2, size_t vl);
vuint8m8_t __riscv_vsub_vx_u8m8 (vuint8m8_t op1, uint8_t op2, size_t vl);
vuint16mf4_t __riscv_vsub_vx_u16mf4 (vuint16mf4_t op1, uint16_t op2, size_t vl);
vuint16mf2_t __riscv_vsub_vx_u16mf2 (vuint16mf2_t op1, uint16_t op2, size_t vl);
vuint16m1_t __riscv_vsub_vx_u16m1 (vuint16m1_t op1, uint16_t op2, size_t vl);
vuint16m2_t __riscv_vsub_vx_u16m2 (vuint16m2_t op1, uint16_t op2, size_t vl);
vuint16m4_t __riscv_vsub_vx_u16m4 (vuint16m4_t op1, uint16_t op2, size_t vl);
vuint16m8_t __riscv_vsub_vx_u16m8 (vuint16m8_t op1, uint16_t op2, size_t vl);
vuint32mf2_t __riscv_vsub_vx_u32mf2 (vuint32mf2_t op1, uint32_t op2, size_t vl);
vuint32m1_t __riscv_vsub_vx_u32m1 (vuint32m1_t op1, uint32_t op2, size_t vl);
vuint32m2_t __riscv_vsub_vx_u32m2 (vuint32m2_t op1, uint32_t op2, size_t vl);
vuint32m4_t __riscv_vsub_vx_u32m4 (vuint32m4_t op1, uint32_t op2, size_t vl);
vuint32m8_t __riscv_vsub_vx_u32m8 (vuint32m8_t op1, uint32_t op2, size_t vl);
vuint64m1_t __riscv_vsub_vx_u64m1 (vuint64m1_t op1, uint64_t op2, size_t vl);
vuint64m2_t __riscv_vsub_vx_u64m2 (vuint64m2_t op1, uint64_t op2, size_t vl);
vuint64m4_t __riscv_vsub_vx_u64m4 (vuint64m4_t op1, uint64_t op2, size_t vl);
vuint64m8_t __riscv_vsub_vx_u64m8 (vuint64m8_t op1, uint64_t op2, size_t vl);
```

**Parameters:**
- `op1`: Integer vector (minuend)
- `op2`: Scalar integer value (subtrahend)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector containing the element-wise difference `op1[i] - op2`

**Supported Types:** Signed and unsigned integers (8, 16, 32, 64-bit) with LMUL from mf8 to m8

## VECTOR_SUB_MASKED

### vsub_vx_m
**Masked Vector-Scalar Integer Subtraction**

Performs element-wise subtraction of a scalar from each element of an integer vector with mask control: `result[i] = mask[i] ? (op1[i] - op2) : op1[i]`

```c
vint8mf8_t __riscv_vsub_vx_i8mf8_m (vbool64_t mask, vint8mf8_t op1, int8_t op2, size_t vl);
vint8mf4_t __riscv_vsub_vx_i8mf4_m (vbool32_t mask, vint8mf4_t op1, int8_t op2, size_t vl);
vint8mf2_t __riscv_vsub_vx_i8mf2_m (vbool16_t mask, vint8mf2_t op1, int8_t op2, size_t vl);
vint8m1_t __riscv_vsub_vx_i8m1_m (vbool8_t mask, vint8m1_t op1, int8_t op2, size_t vl);
vint8m2_t __riscv_vsub_vx_i8m2_m (vbool4_t mask, vint8m2_t op1, int8_t op2, size_t vl);
vint8m4_t __riscv_vsub_vx_i8m4_m (vbool2_t mask, vint8m4_t op1, int8_t op2, size_t vl);
vint8m8_t __riscv_vsub_vx_i8m8_m (vbool1_t mask, vint8m8_t op1, int8_t op2, size_t vl);
vint16mf4_t __riscv_vsub_vx_i16mf4_m (vbool64_t mask, vint16mf4_t op1, int16_t op2, size_t vl);
vint16mf2_t __riscv_vsub_vx_i16mf2_m (vbool32_t mask, vint16mf2_t op1, int16_t op2, size_t vl);
vint16m1_t __riscv_vsub_vx_i16m1_m (vbool16_t mask, vint16m1_t op1, int16_t op2, size_t vl);
vint16m2_t __riscv_vsub_vx_i16m2_m (vbool8_t mask, vint16m2_t op1, int16_t op2, size_t vl);
vint16m4_t __riscv_vsub_vx_i16m4_m (vbool4_t mask, vint16m4_t op1, int16_t op2, size_t vl);
vint16m8_t __riscv_vsub_vx_i16m8_m (vbool2_t mask, vint16m8_t op1, int16_t op2, size_t vl);
vint32mf2_t __riscv_vsub_vx_i32mf2_m (vbool64_t mask, vint32mf2_t op1, int32_t op2, size_t vl);
vint32m1_t __riscv_vsub_vx_i32m1_m (vbool32_t mask, vint32m1_t op1, int32_t op2, size_t vl);
vint32m2_t __riscv_vsub_vx_i32m2_m (vbool16_t mask, vint32m2_t op1, int32_t op2, size_t vl);
vint32m4_t __riscv_vsub_vx_i32m4_m (vbool8_t mask, vint32m4_t op1, int32_t op2, size_t vl);
vint32m8_t __riscv_vsub_vx_i32m8_m (vbool4_t mask, vint32m8_t op1, int32_t op2, size_t vl);
vint64m1_t __riscv_vsub_vx_i64m1_m (vbool64_t mask, vint64m1_t op1, int64_t op2, size_t vl);
vint64m2_t __riscv_vsub_vx_i64m2_m (vbool32_t mask, vint64m2_t op1, int64_t op2, size_t vl);
vint64m4_t __riscv_vsub_vx_i64m4_m (vbool16_t mask, vint64m4_t op1, int64_t op2, size_t vl);
vint64m8_t __riscv_vsub_vx_i64m8_m (vbool8_t mask, vint64m8_t op1, int64_t op2, size_t vl);
vuint8mf8_t __riscv_vsub_vx_u8mf8_m (vbool64_t mask, vuint8mf8_t op1, uint8_t op2, size_t vl);
vuint8mf4_t __riscv_vsub_vx_u8mf4_m (vbool32_t mask, vuint8mf4_t op1, uint8_t op2, size_t vl);
vuint8mf2_t __riscv_vsub_vx_u8mf2_m (vbool16_t mask, vuint8mf2_t op1, uint8_t op2, size_t vl);
vuint8m1_t __riscv_vsub_vx_u8m1_m (vbool8_t mask, vuint8m1_t op1, uint8_t op2, size_t vl);
vuint8m2_t __riscv_vsub_vx_u8m2_m (vbool4_t mask, vuint8m2_t op1, uint8_t op2, size_t vl);
vuint8m4_t __riscv_vsub_vx_u8m4_m (vbool2_t mask, vuint8m4_t op1, uint8_t op2, size_t vl);
vuint8m8_t __riscv_vsub_vx_u8m8_m (vbool1_t mask, vuint8m8_t op1, uint8_t op2, size_t vl);
vuint16mf4_t __riscv_vsub_vx_u16mf4_m (vbool64_t mask, vuint16mf4_t op1, uint16_t op2, size_t vl);
vuint16mf2_t __riscv_vsub_vx_u16mf2_m (vbool32_t mask, vuint16mf2_t op1, uint16_t op2, size_t vl);
vuint16m1_t __riscv_vsub_vx_u16m1_m (vbool16_t mask, vuint16m1_t op1, uint16_t op2, size_t vl);
vuint16m2_t __riscv_vsub_vx_u16m2_m (vbool8_t mask, vuint16m2_t op1, uint16_t op2, size_t vl);
vuint16m4_t __riscv_vsub_vx_u16m4_m (vbool4_t mask, vuint16m4_t op1, uint16_t op2, size_t vl);
vuint16m8_t __riscv_vsub_vx_u16m8_m (vbool2_t mask, vuint16m8_t op1, uint16_t op2, size_t vl);
vuint32mf2_t __riscv_vsub_vx_u32mf2_m (vbool64_t mask, vuint32mf2_t op1, uint32_t op2, size_t vl);
vuint32m1_t __riscv_vsub_vx_u32m1_m (vbool32_t mask, vuint32m1_t op1, uint32_t op2, size_t vl);
vuint32m2_t __riscv_vsub_vx_u32m2_m (vbool16_t mask, vuint32m2_t op1, uint32_t op2, size_t vl);
vuint32m4_t __riscv_vsub_vx_u32m4_m (vbool8_t mask, vuint32m4_t op1, uint32_t op2, size_t vl);
vuint32m8_t __riscv_vsub_vx_u32m8_m (vbool4_t mask, vuint32m8_t op1, uint32_t op2, size_t vl);
vuint64m1_t __riscv_vsub_vx_u64m1_m (vbool64_t mask, vuint64m1_t op1, uint64_t op2, size_t vl);
vuint64m2_t __riscv_vsub_vx_u64m2_m (vbool32_t mask, vuint64m2_t op1, uint64_t op2, size_t vl);
vuint64m4_t __riscv_vsub_vx_u64m4_m (vbool16_t mask, vuint64m4_t op1, uint64_t op2, size_t vl);
vuint64m8_t __riscv_vsub_vx_u64m8_m (vbool8_t mask, vuint64m8_t op1, uint64_t op2, size_t vl);
```

**Parameters:**
- `mask`: Boolean mask vector controlling which elements are processed
- `op1`: Integer vector (minuend)
- `op2`: Scalar integer value (subtrahend)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector with subtraction applied only to masked elements; unmasked elements retain values from `op1`

**Supported Types:** Signed and unsigned integers (8, 16, 32, 64-bit) with LMUL from mf8 to m8

### vfsub_vv_m
**Masked Vector-Vector Floating-Point Subtraction**

Performs element-wise subtraction of two floating-point vectors with mask control: `result[i] = mask[i] ? (op1[i] - op2[i]) : op1[i]`

```c
vfloat16mf4_t __riscv_vfsub_vv_f16mf4_m (vbool64_t mask, vfloat16mf4_t op1, vfloat16mf4_t op2, size_t vl);
vfloat16mf2_t __riscv_vfsub_vv_f16mf2_m (vbool32_t mask, vfloat16mf2_t op1, vfloat16mf2_t op2, size_t vl);
vfloat16m1_t __riscv_vfsub_vv_f16m1_m (vbool16_t mask, vfloat16m1_t op1, vfloat16m1_t op2, size_t vl);
vfloat16m2_t __riscv_vfsub_vv_f16m2_m (vbool8_t mask, vfloat16m2_t op1, vfloat16m2_t op2, size_t vl);
vfloat16m4_t __riscv_vfsub_vv_f16m4_m (vbool4_t mask, vfloat16m4_t op1, vfloat16m4_t op2, size_t vl);
vfloat16m8_t __riscv_vfsub_vv_f16m8_m (vbool2_t mask, vfloat16m8_t op1, vfloat16m8_t op2, size_t vl);
vfloat32mf2_t __riscv_vfsub_vv_f32mf2_m (vbool64_t mask, vfloat32mf2_t op1, vfloat32mf2_t op2, size_t vl);
vfloat32m1_t __riscv_vfsub_vv_f32m1_m (vbool32_t mask, vfloat32m1_t op1, vfloat32m1_t op2, size_t vl);
vfloat32m2_t __riscv_vfsub_vv_f32m2_m (vbool16_t mask, vfloat32m2_t op1, vfloat32m2_t op2, size_t vl);
vfloat32m4_t __riscv_vfsub_vv_f32m4_m (vbool8_t mask, vfloat32m4_t op1, vfloat32m4_t op2, size_t vl);
vfloat32m8_t __riscv_vfsub_vv_f32m8_m (vbool4_t mask, vfloat32m8_t op1, vfloat32m8_t op2, size_t vl);
vfloat64m1_t __riscv_vfsub_vv_f64m1_m (vbool64_t mask, vfloat64m1_t op1, vfloat64m1_t op2, size_t vl);
vfloat64m2_t __riscv_vfsub_vv_f64m2_m (vbool32_t mask, vfloat64m2_t op1, vfloat64m2_t op2, size_t vl);
vfloat64m4_t __riscv_vfsub_vv_f64m4_m (vbool16_t mask, vfloat64m4_t op1, vfloat64m4_t op2, size_t vl);
vfloat64m8_t __riscv_vfsub_vv_f64m8_m (vbool8_t mask, vfloat64m8_t op1, vfloat64m8_t op2, size_t vl);
```

**Parameters:**
- `mask`: Boolean mask vector controlling which elements are processed
- `op1`: First floating-point vector (minuend)
- `op2`: Second floating-point vector (subtrahend)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector with subtraction applied only to masked elements; unmasked elements retain values from `op1`

**Supported Types:** FP16, FP32, FP64 with LMUL from mf4 to m8

### vfsub_vf_m
**Masked Vector-Scalar Floating-Point Subtraction**

Performs element-wise subtraction of a scalar from each element of a floating-point vector with mask control: `result[i] = mask[i] ? (op1[i] - op2) : op1[i]`

```c
vfloat16mf4_t __riscv_vfsub_vf_f16mf4_m (vbool64_t mask, vfloat16mf4_t op1, float16_t op2, size_t vl);
vfloat16mf2_t __riscv_vfsub_vf_f16mf2_m (vbool32_t mask, vfloat16mf2_t op1, float16_t op2, size_t vl);
vfloat16m1_t __riscv_vfsub_vf_f16m1_m (vbool16_t mask, vfloat16m1_t op1, float16_t op2, size_t vl);
vfloat16m2_t __riscv_vfsub_vf_f16m2_m (vbool8_t mask, vfloat16m2_t op1, float16_t op2, size_t vl);
vfloat16m4_t __riscv_vfsub_vf_f16m4_m (vbool4_t mask, vfloat16m4_t op1, float16_t op2, size_t vl);
vfloat16m8_t __riscv_vfsub_vf_f16m8_m (vbool2_t mask, vfloat16m8_t op1, float16_t op2, size_t vl);
vfloat32mf2_t __riscv_vfsub_vf_f32mf2_m (vbool64_t mask, vfloat32mf2_t op1, float32_t op2, size_t vl);
vfloat32m1_t __riscv_vfsub_vf_f32m1_m (vbool32_t mask, vfloat32m1_t op1, float32_t op2, size_t vl);
vfloat32m2_t __riscv_vfsub_vf_f32m2_m (vbool16_t mask, vfloat32m2_t op1, float32_t op2, size_t vl);
vfloat32m4_t __riscv_vfsub_vf_f32m4_m (vbool8_t mask, vfloat32m4_t op1, float32_t op2, size_t vl);
vfloat32m8_t __riscv_vfsub_vf_f32m8_m (vbool4_t mask, vfloat32m8_t op1, float32_t op2, size_t vl);
vfloat64m1_t __riscv_vfsub_vf_f64m1_m (vbool64_t mask, vfloat64m1_t op1, float64_t op2, size_t vl);
vfloat64m2_t __riscv_vfsub_vf_f64m2_m (vbool32_t mask, vfloat64m2_t op1, float64_t op2, size_t vl);
vfloat64m4_t __riscv_vfsub_vf_f64m4_m (vbool16_t mask, vfloat64m4_t op1, float64_t op2, size_t vl);
vfloat64m8_t __riscv_vfsub_vf_f64m8_m (vbool8_t mask, vfloat64m8_t op1, float64_t op2, size_t vl);
```

**Parameters:**
- `mask`: Boolean mask vector controlling which elements are processed
- `op1`: Floating-point vector (minuend)
- `op2`: Scalar floating-point value (subtrahend)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector with subtraction applied only to masked elements; unmasked elements retain values from `op1`

**Supported Types:** FP16, FP32, FP64 with LMUL from mf4 to m8

### vsub_vv_m
**Masked Vector-Vector Integer Subtraction**

Performs element-wise subtraction of two integer vectors with mask control: `result[i] = mask[i] ? (op1[i] - op2[i]) : op1[i]`

```c
vint8mf8_t __riscv_vsub_vv_i8mf8_m (vbool64_t mask, vint8mf8_t op1, vint8mf8_t op2, size_t vl);
vint8mf4_t __riscv_vsub_vv_i8mf4_m (vbool32_t mask, vint8mf4_t op1, vint8mf4_t op2, size_t vl);
vint8mf2_t __riscv_vsub_vv_i8mf2_m (vbool16_t mask, vint8mf2_t op1, vint8mf2_t op2, size_t vl);
vint8m1_t __riscv_vsub_vv_i8m1_m (vbool8_t mask, vint8m1_t op1, vint8m1_t op2, size_t vl);
vint8m2_t __riscv_vsub_vv_i8m2_m (vbool4_t mask, vint8m2_t op1, vint8m2_t op2, size_t vl);
vint8m4_t __riscv_vsub_vv_i8m4_m (vbool2_t mask, vint8m4_t op1, vint8m4_t op2, size_t vl);
vint8m8_t __riscv_vsub_vv_i8m8_m (vbool1_t mask, vint8m8_t op1, vint8m8_t op2, size_t vl);
vint16mf4_t __riscv_vsub_vv_i16mf4_m (vbool64_t mask, vint16mf4_t op1, vint16mf4_t op2, size_t vl);
vint16mf2_t __riscv_vsub_vv_i16mf2_m (vbool32_t mask, vint16mf2_t op1, vint16mf2_t op2, size_t vl);
vint16m1_t __riscv_vsub_vv_i16m1_m (vbool16_t mask, vint16m1_t op1, vint16m1_t op2, size_t vl);
vint16m2_t __riscv_vsub_vv_i16m2_m (vbool8_t mask, vint16m2_t op1, vint16m2_t op2, size_t vl);
vint16m4_t __riscv_vsub_vv_i16m4_m (vbool4_t mask, vint16m4_t op1, vint16m4_t op2, size_t vl);
vint16m8_t __riscv_vsub_vv_i16m8_m (vbool2_t mask, vint16m8_t op1, vint16m8_t op2, size_t vl);
vint32mf2_t __riscv_vsub_vv_i32mf2_m (vbool64_t mask, vint32mf2_t op1, vint32mf2_t op2, size_t vl);
vint32m1_t __riscv_vsub_vv_i32m1_m (vbool32_t mask, vint32m1_t op1, vint32m1_t op2, size_t vl);
vint32m2_t __riscv_vsub_vv_i32m2_m (vbool16_t mask, vint32m2_t op1, vint32m2_t op2, size_t vl);
vint32m4_t __riscv_vsub_vv_i32m4_m (vbool8_t mask, vint32m4_t op1, vint32m4_t op2, size_t vl);
vint32m8_t __riscv_vsub_vv_i32m8_m (vbool4_t mask, vint32m8_t op1, vint32m8_t op2, size_t vl);
vint64m1_t __riscv_vsub_vv_i64m1_m (vbool64_t mask, vint64m1_t op1, vint64m1_t op2, size_t vl);
vint64m2_t __riscv_vsub_vv_i64m2_m (vbool32_t mask, vint64m2_t op1, vint64m2_t op2, size_t vl);
vint64m4_t __riscv_vsub_vv_i64m4_m (vbool16_t mask, vint64m4_t op1, vint64m4_t op2, size_t vl);
vint64m8_t __riscv_vsub_vv_i64m8_m (vbool8_t mask, vint64m8_t op1, vint64m8_t op2, size_t vl);
vuint8mf8_t __riscv_vsub_vv_u8mf8_m (vbool64_t mask, vuint8mf8_t op1, vuint8mf8_t op2, size_t vl);
vuint8mf4_t __riscv_vsub_vv_u8mf4_m (vbool32_t mask, vuint8mf4_t op1, vuint8mf4_t op2, size_t vl);
vuint8mf2_t __riscv_vsub_vv_u8mf2_m (vbool16_t mask, vuint8mf2_t op1, vuint8mf2_t op2, size_t vl);
vuint8m1_t __riscv_vsub_vv_u8m1_m (vbool8_t mask, vuint8m1_t op1, vuint8m1_t op2, size_t vl);
vuint8m2_t __riscv_vsub_vv_u8m2_m (vbool4_t mask, vuint8m2_t op1, vuint8m2_t op2, size_t vl);
vuint8m4_t __riscv_vsub_vv_u8m4_m (vbool2_t mask, vuint8m4_t op1, vuint8m4_t op2, size_t vl);
vuint8m8_t __riscv_vsub_vv_u8m8_m (vbool1_t mask, vuint8m8_t op1, vuint8m8_t op2, size_t vl);
vuint16mf4_t __riscv_vsub_vv_u16mf4_m (vbool64_t mask, vuint16mf4_t op1, vuint16mf4_t op2, size_t vl);
vuint16mf2_t __riscv_vsub_vv_u16mf2_m (vbool32_t mask, vuint16mf2_t op1, vuint16mf2_t op2, size_t vl);
vuint16m1_t __riscv_vsub_vv_u16m1_m (vbool16_t mask, vuint16m1_t op1, vuint16m1_t op2, size_t vl);
vuint16m2_t __riscv_vsub_vv_u16m2_m (vbool8_t mask, vuint16m2_t op1, vuint16m2_t op2, size_t vl);
vuint16m4_t __riscv_vsub_vv_u16m4_m (vbool4_t mask, vuint16m4_t op1, vuint16m4_t op2, size_t vl);
vuint16m8_t __riscv_vsub_vv_u16m8_m (vbool2_t mask, vuint16m8_t op1, vuint16m8_t op2, size_t vl);
vuint32mf2_t __riscv_vsub_vv_u32mf2_m (vbool64_t mask, vuint32mf2_t op1, vuint32mf2_t op2, size_t vl);
vuint32m1_t __riscv_vsub_vv_u32m1_m (vbool32_t mask, vuint32m1_t op1, vuint32m1_t op2, size_t vl);
vuint32m2_t __riscv_vsub_vv_u32m2_m (vbool16_t mask, vuint32m2_t op1, vuint32m2_t op2, size_t vl);
vuint32m4_t __riscv_vsub_vv_u32m4_m (vbool8_t mask, vuint32m4_t op1, vuint32m4_t op2, size_t vl);
vuint32m8_t __riscv_vsub_vv_u32m8_m (vbool4_t mask, vuint32m8_t op1, vuint32m8_t op2, size_t vl);
vuint64m1_t __riscv_vsub_vv_u64m1_m (vbool64_t mask, vuint64m1_t op1, vuint64m1_t op2, size_t vl);
vuint64m2_t __riscv_vsub_vv_u64m2_m (vbool32_t mask, vuint64m2_t op1, vuint64m2_t op2, size_t vl);
vuint64m4_t __riscv_vsub_vv_u64m4_m (vbool16_t mask, vuint64m4_t op1, vuint64m4_t op2, size_t vl);
vuint64m8_t __riscv_vsub_vv_u64m8_m (vbool8_t mask, vuint64m8_t op1, vuint64m8_t op2, size_t vl);
```

**Parameters:**
- `mask`: Boolean mask vector controlling which elements are processed
- `op1`: First integer vector (minuend)
- `op2`: Second integer vector (subtrahend)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector with subtraction applied only to masked elements; unmasked elements retain values from `op1`

**Supported Types:** Signed and unsigned integers (8, 16, 32, 64-bit) with LMUL from mf8 to m8

## VECTOR_ADD

### vadd_vv
**Vector-Vector Integer Addition**

Performs element-wise addition of two integer vectors: `result[i] = op1[i] + op2[i]`

```c
vint8mf8_t __riscv_vadd_vv_i8mf8 (vint8mf8_t op1, vint8mf8_t op2, size_t vl);
vint8mf4_t __riscv_vadd_vv_i8mf4 (vint8mf4_t op1, vint8mf4_t op2, size_t vl);
vint8mf2_t __riscv_vadd_vv_i8mf2 (vint8mf2_t op1, vint8mf2_t op2, size_t vl);
vint8m1_t __riscv_vadd_vv_i8m1 (vint8m1_t op1, vint8m1_t op2, size_t vl);
vint8m2_t __riscv_vadd_vv_i8m2 (vint8m2_t op1, vint8m2_t op2, size_t vl);
vint8m4_t __riscv_vadd_vv_i8m4 (vint8m4_t op1, vint8m4_t op2, size_t vl);
vint8m8_t __riscv_vadd_vv_i8m8 (vint8m8_t op1, vint8m8_t op2, size_t vl);
vint16mf4_t __riscv_vadd_vv_i16mf4 (vint16mf4_t op1, vint16mf4_t op2, size_t vl);
vint16mf2_t __riscv_vadd_vv_i16mf2 (vint16mf2_t op1, vint16mf2_t op2, size_t vl);
vint16m1_t __riscv_vadd_vv_i16m1 (vint16m1_t op1, vint16m1_t op2, size_t vl);
vint16m2_t __riscv_vadd_vv_i16m2 (vint16m2_t op1, vint16m2_t op2, size_t vl);
vint16m4_t __riscv_vadd_vv_i16m4 (vint16m4_t op1, vint16m4_t op2, size_t vl);
vint16m8_t __riscv_vadd_vv_i16m8 (vint16m8_t op1, vint16m8_t op2, size_t vl);
vint32mf2_t __riscv_vadd_vv_i32mf2 (vint32mf2_t op1, vint32mf2_t op2, size_t vl);
vint32m1_t __riscv_vadd_vv_i32m1 (vint32m1_t op1, vint32m1_t op2, size_t vl);
vint32m2_t __riscv_vadd_vv_i32m2 (vint32m2_t op1, vint32m2_t op2, size_t vl);
vint32m4_t __riscv_vadd_vv_i32m4 (vint32m4_t op1, vint32m4_t op2, size_t vl);
vint32m8_t __riscv_vadd_vv_i32m8 (vint32m8_t op1, vint32m8_t op2, size_t vl);
vint64m1_t __riscv_vadd_vv_i64m1 (vint64m1_t op1, vint64m1_t op2, size_t vl);
vint64m2_t __riscv_vadd_vv_i64m2 (vint64m2_t op1, vint64m2_t op2, size_t vl);
vint64m4_t __riscv_vadd_vv_i64m4 (vint64m4_t op1, vint64m4_t op2, size_t vl);
vint64m8_t __riscv_vadd_vv_i64m8 (vint64m8_t op1, vint64m8_t op2, size_t vl);
vuint8mf8_t __riscv_vadd_vv_u8mf8 (vuint8mf8_t op1, vuint8mf8_t op2, size_t vl);
vuint8mf4_t __riscv_vadd_vv_u8mf4 (vuint8mf4_t op1, vuint8mf4_t op2, size_t vl);
vuint8mf2_t __riscv_vadd_vv_u8mf2 (vuint8mf2_t op1, vuint8mf2_t op2, size_t vl);
vuint8m1_t __riscv_vadd_vv_u8m1 (vuint8m1_t op1, vuint8m1_t op2, size_t vl);
vuint8m2_t __riscv_vadd_vv_u8m2 (vuint8m2_t op1, vuint8m2_t op2, size_t vl);
vuint8m4_t __riscv_vadd_vv_u8m4 (vuint8m4_t op1, vuint8m4_t op2, size_t vl);
vuint8m8_t __riscv_vadd_vv_u8m8 (vuint8m8_t op1, vuint8m8_t op2, size_t vl);
vuint16mf4_t __riscv_vadd_vv_u16mf4 (vuint16mf4_t op1, vuint16mf4_t op2, size_t vl);
vuint16mf2_t __riscv_vadd_vv_u16mf2 (vuint16mf2_t op1, vuint16mf2_t op2, size_t vl);
vuint16m1_t __riscv_vadd_vv_u16m1 (vuint16m1_t op1, vuint16m1_t op2, size_t vl);
vuint16m2_t __riscv_vadd_vv_u16m2 (vuint16m2_t op1, vuint16m2_t op2, size_t vl);
vuint16m4_t __riscv_vadd_vv_u16m4 (vuint16m4_t op1, vuint16m4_t op2, size_t vl);
vuint16m8_t __riscv_vadd_vv_u16m8 (vuint16m8_t op1, vuint16m8_t op2, size_t vl);
vuint32mf2_t __riscv_vadd_vv_u32mf2 (vuint32mf2_t op1, vuint32mf2_t op2, size_t vl);
vuint32m1_t __riscv_vadd_vv_u32m1 (vuint32m1_t op1, vuint32m1_t op2, size_t vl);
vuint32m2_t __riscv_vadd_vv_u32m2 (vuint32m2_t op1, vuint32m2_t op2, size_t vl);
vuint32m4_t __riscv_vadd_vv_u32m4 (vuint32m4_t op1, vuint32m4_t op2, size_t vl);
vuint32m8_t __riscv_vadd_vv_u32m8 (vuint32m8_t op1, vuint32m8_t op2, size_t vl);
vuint64m1_t __riscv_vadd_vv_u64m1 (vuint64m1_t op1, vuint64m1_t op2, size_t vl);
vuint64m2_t __riscv_vadd_vv_u64m2 (vuint64m2_t op1, vuint64m2_t op2, size_t vl);
vuint64m4_t __riscv_vadd_vv_u64m4 (vuint64m4_t op1, vuint64m4_t op2, size_t vl);
vuint64m8_t __riscv_vadd_vv_u64m8 (vuint64m8_t op1, vuint64m8_t op2, size_t vl);
```

**Parameters:**
- `op1`: First integer vector (addend)
- `op2`: Second integer vector (addend)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector containing the element-wise sum `op1 + op2`

**Supported Types:** Signed and unsigned integers (8, 16, 32, 64-bit) with LMUL from mf8 to m8

### vadd_vx
**Vector-Scalar Integer Addition**

Performs element-wise addition of a scalar to each element of an integer vector: `result[i] = op1[i] + op2`

```c
vint8mf8_t __riscv_vadd_vx_i8mf8 (vint8mf8_t op1, int8_t op2, size_t vl);
vint8mf4_t __riscv_vadd_vx_i8mf4 (vint8mf4_t op1, int8_t op2, size_t vl);
vint8mf2_t __riscv_vadd_vx_i8mf2 (vint8mf2_t op1, int8_t op2, size_t vl);
vint8m1_t __riscv_vadd_vx_i8m1 (vint8m1_t op1, int8_t op2, size_t vl);
vint8m2_t __riscv_vadd_vx_i8m2 (vint8m2_t op1, int8_t op2, size_t vl);
vint8m4_t __riscv_vadd_vx_i8m4 (vint8m4_t op1, int8_t op2, size_t vl);
vint8m8_t __riscv_vadd_vx_i8m8 (vint8m8_t op1, int8_t op2, size_t vl);
vint16mf4_t __riscv_vadd_vx_i16mf4 (vint16mf4_t op1, int16_t op2, size_t vl);
vint16mf2_t __riscv_vadd_vx_i16mf2 (vint16mf2_t op1, int16_t op2, size_t vl);
vint16m1_t __riscv_vadd_vx_i16m1 (vint16m1_t op1, int16_t op2, size_t vl);
vint16m2_t __riscv_vadd_vx_i16m2 (vint16m2_t op1, int16_t op2, size_t vl);
vint16m4_t __riscv_vadd_vx_i16m4 (vint16m4_t op1, int16_t op2, size_t vl);
vint16m8_t __riscv_vadd_vx_i16m8 (vint16m8_t op1, int16_t op2, size_t vl);
vint32mf2_t __riscv_vadd_vx_i32mf2 (vint32mf2_t op1, int32_t op2, size_t vl);
vint32m1_t __riscv_vadd_vx_i32m1 (vint32m1_t op1, int32_t op2, size_t vl);
vint32m2_t __riscv_vadd_vx_i32m2 (vint32m2_t op1, int32_t op2, size_t vl);
vint32m4_t __riscv_vadd_vx_i32m4 (vint32m4_t op1, int32_t op2, size_t vl);
vint32m8_t __riscv_vadd_vx_i32m8 (vint32m8_t op1, int32_t op2, size_t vl);
vint64m1_t __riscv_vadd_vx_i64m1 (vint64m1_t op1, int64_t op2, size_t vl);
vint64m2_t __riscv_vadd_vx_i64m2 (vint64m2_t op1, int64_t op2, size_t vl);
vint64m4_t __riscv_vadd_vx_i64m4 (vint64m4_t op1, int64_t op2, size_t vl);
vint64m8_t __riscv_vadd_vx_i64m8 (vint64m8_t op1, int64_t op2, size_t vl);
vuint8mf8_t __riscv_vadd_vx_u8mf8 (vuint8mf8_t op1, uint8_t op2, size_t vl);
vuint8mf4_t __riscv_vadd_vx_u8mf4 (vuint8mf4_t op1, uint8_t op2, size_t vl);
vuint8mf2_t __riscv_vadd_vx_u8mf2 (vuint8mf2_t op1, uint8_t op2, size_t vl);
vuint8m1_t __riscv_vadd_vx_u8m1 (vuint8m1_t op1, uint8_t op2, size_t vl);
vuint8m2_t __riscv_vadd_vx_u8m2 (vuint8m2_t op1, uint8_t op2, size_t vl);
vuint8m4_t __riscv_vadd_vx_u8m4 (vuint8m4_t op1, uint8_t op2, size_t vl);
vuint8m8_t __riscv_vadd_vx_u8m8 (vuint8m8_t op1, uint8_t op2, size_t vl);
vuint16mf4_t __riscv_vadd_vx_u16mf4 (vuint16mf4_t op1, uint16_t op2, size_t vl);
vuint16mf2_t __riscv_vadd_vx_u16mf2 (vuint16mf2_t op1, uint16_t op2, size_t vl);
vuint16m1_t __riscv_vadd_vx_u16m1 (vuint16m1_t op1, uint16_t op2, size_t vl);
vuint16m2_t __riscv_vadd_vx_u16m2 (vuint16m2_t op1, uint16_t op2, size_t vl);
vuint16m4_t __riscv_vadd_vx_u16m4 (vuint16m4_t op1, uint16_t op2, size_t vl);
vuint16m8_t __riscv_vadd_vx_u16m8 (vuint16m8_t op1, uint16_t op2, size_t vl);
vuint32mf2_t __riscv_vadd_vx_u32mf2 (vuint32mf2_t op1, uint32_t op2, size_t vl);
vuint32m1_t __riscv_vadd_vx_u32m1 (vuint32m1_t op1, uint32_t op2, size_t vl);
vuint32m2_t __riscv_vadd_vx_u32m2 (vuint32m2_t op1, uint32_t op2, size_t vl);
vuint32m4_t __riscv_vadd_vx_u32m4 (vuint32m4_t op1, uint32_t op2, size_t vl);
vuint32m8_t __riscv_vadd_vx_u32m8 (vuint32m8_t op1, uint32_t op2, size_t vl);
vuint64m1_t __riscv_vadd_vx_u64m1 (vuint64m1_t op1, uint64_t op2, size_t vl);
vuint64m2_t __riscv_vadd_vx_u64m2 (vuint64m2_t op1, uint64_t op2, size_t vl);
vuint64m4_t __riscv_vadd_vx_u64m4 (vuint64m4_t op1, uint64_t op2, size_t vl);
vuint64m8_t __riscv_vadd_vx_u64m8 (vuint64m8_t op1, uint64_t op2, size_t vl);
```

**Parameters:**
- `op1`: Integer vector (addend)
- `op2`: Scalar integer value (addend)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector containing the element-wise sum `op1[i] + op2`

**Supported Types:** Signed and unsigned integers (8, 16, 32, 64-bit) with LMUL from mf8 to m8

### vfadd_vv
**Vector-Vector Floating-Point Addition**

Performs element-wise addition of two floating-point vectors: `result[i] = op1[i] + op2[i]`

```c
vfloat16mf4_t __riscv_vfadd_vv_f16mf4 (vfloat16mf4_t op1, vfloat16mf4_t op2, size_t vl);
vfloat16mf2_t __riscv_vfadd_vv_f16mf2 (vfloat16mf2_t op1, vfloat16mf2_t op2, size_t vl);
vfloat16m1_t __riscv_vfadd_vv_f16m1 (vfloat16m1_t op1, vfloat16m1_t op2, size_t vl);
vfloat16m2_t __riscv_vfadd_vv_f16m2 (vfloat16m2_t op1, vfloat16m2_t op2, size_t vl);
vfloat16m4_t __riscv_vfadd_vv_f16m4 (vfloat16m4_t op1, vfloat16m4_t op2, size_t vl);
vfloat16m8_t __riscv_vfadd_vv_f16m8 (vfloat16m8_t op1, vfloat16m8_t op2, size_t vl);
vfloat32mf2_t __riscv_vfadd_vv_f32mf2 (vfloat32mf2_t op1, vfloat32mf2_t op2, size_t vl);
vfloat32m1_t __riscv_vfadd_vv_f32m1 (vfloat32m1_t op1, vfloat32m1_t op2, size_t vl);
vfloat32m2_t __riscv_vfadd_vv_f32m2 (vfloat32m2_t op1, vfloat32m2_t op2, size_t vl);
vfloat32m4_t __riscv_vfadd_vv_f32m4 (vfloat32m4_t op1, vfloat32m4_t op2, size_t vl);
vfloat32m8_t __riscv_vfadd_vv_f32m8 (vfloat32m8_t op1, vfloat32m8_t op2, size_t vl);
vfloat64m1_t __riscv_vfadd_vv_f64m1 (vfloat64m1_t op1, vfloat64m1_t op2, size_t vl);
vfloat64m2_t __riscv_vfadd_vv_f64m2 (vfloat64m2_t op1, vfloat64m2_t op2, size_t vl);
vfloat64m4_t __riscv_vfadd_vv_f64m4 (vfloat64m4_t op1, vfloat64m4_t op2, size_t vl);
vfloat64m8_t __riscv_vfadd_vv_f64m8 (vfloat64m8_t op1, vfloat64m8_t op2, size_t vl);
```

**Parameters:**
- `op1`: First floating-point vector (addend)
- `op2`: Second floating-point vector (addend)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector containing the element-wise sum `op1 + op2`

**Supported Types:** FP16, FP32, FP64 with LMUL from mf4 to m8

### vfadd_vf
**Vector-Scalar Floating-Point Addition**

Performs element-wise addition of a scalar to each element of a floating-point vector: `result[i] = op1[i] + op2`

```c
vfloat16mf4_t __riscv_vfadd_vf_f16mf4 (vfloat16mf4_t op1, float16_t op2, size_t vl);
vfloat16mf2_t __riscv_vfadd_vf_f16mf2 (vfloat16mf2_t op1, float16_t op2, size_t vl);
vfloat16m1_t __riscv_vfadd_vf_f16m1 (vfloat16m1_t op1, float16_t op2, size_t vl);
vfloat16m2_t __riscv_vfadd_vf_f16m2 (vfloat16m2_t op1, float16_t op2, size_t vl);
vfloat16m4_t __riscv_vfadd_vf_f16m4 (vfloat16m4_t op1, float16_t op2, size_t vl);
vfloat16m8_t __riscv_vfadd_vf_f16m8 (vfloat16m8_t op1, float16_t op2, size_t vl);
vfloat32mf2_t __riscv_vfadd_vf_f32mf2 (vfloat32mf2_t op1, float32_t op2, size_t vl);
vfloat32m1_t __riscv_vfadd_vf_f32m1 (vfloat32m1_t op1, float32_t op2, size_t vl);
vfloat32m2_t __riscv_vfadd_vf_f32m2 (vfloat32m2_t op1, float32_t op2, size_t vl);
vfloat32m4_t __riscv_vfadd_vf_f32m4 (vfloat32m4_t op1, float32_t op2, size_t vl);
vfloat32m8_t __riscv_vfadd_vf_f32m8 (vfloat32m8_t op1, float32_t op2, size_t vl);
vfloat64m1_t __riscv_vfadd_vf_f64m1 (vfloat64m1_t op1, float64_t op2, size_t vl);
vfloat64m2_t __riscv_vfadd_vf_f64m2 (vfloat64m2_t op1, float64_t op2, size_t vl);
vfloat64m4_t __riscv_vfadd_vf_f64m4 (vfloat64m4_t op1, float64_t op2, size_t vl);
vfloat64m8_t __riscv_vfadd_vf_f64m8 (vfloat64m8_t op1, float64_t op2, size_t vl);
```

**Parameters:**
- `op1`: Floating-point vector (addend)
- `op2`: Scalar floating-point value (addend)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector containing the element-wise sum `op1[i] + op2`

**Supported Types:** FP16, FP32, FP64 with LMUL from mf4 to m8

## VECTOR_ADD_MASKED

### vadd_vv_m
**Masked Vector-Vector Integer Addition**

Performs element-wise addition of two integer vectors with mask control: `result[i] = mask[i] ? (op1[i] + op2[i]) : op1[i]`

```c
vint8mf8_t __riscv_vadd_vv_i8mf8_m (vbool64_t mask, vint8mf8_t op1, vint8mf8_t op2, size_t vl);
vint8mf4_t __riscv_vadd_vv_i8mf4_m (vbool32_t mask, vint8mf4_t op1, vint8mf4_t op2, size_t vl);
vint8mf2_t __riscv_vadd_vv_i8mf2_m (vbool16_t mask, vint8mf2_t op1, vint8mf2_t op2, size_t vl);
vint8m1_t __riscv_vadd_vv_i8m1_m (vbool8_t mask, vint8m1_t op1, vint8m1_t op2, size_t vl);
vint8m2_t __riscv_vadd_vv_i8m2_m (vbool4_t mask, vint8m2_t op1, vint8m2_t op2, size_t vl);
vint8m4_t __riscv_vadd_vv_i8m4_m (vbool2_t mask, vint8m4_t op1, vint8m4_t op2, size_t vl);
vint8m8_t __riscv_vadd_vv_i8m8_m (vbool1_t mask, vint8m8_t op1, vint8m8_t op2, size_t vl);
vint16mf4_t __riscv_vadd_vv_i16mf4_m (vbool64_t mask, vint16mf4_t op1, vint16mf4_t op2, size_t vl);
vint16mf2_t __riscv_vadd_vv_i16mf2_m (vbool32_t mask, vint16mf2_t op1, vint16mf2_t op2, size_t vl);
vint16m1_t __riscv_vadd_vv_i16m1_m (vbool16_t mask, vint16m1_t op1, vint16m1_t op2, size_t vl);
vint16m2_t __riscv_vadd_vv_i16m2_m (vbool8_t mask, vint16m2_t op1, vint16m2_t op2, size_t vl);
vint16m4_t __riscv_vadd_vv_i16m4_m (vbool4_t mask, vint16m4_t op1, vint16m4_t op2, size_t vl);
vint16m8_t __riscv_vadd_vv_i16m8_m (vbool2_t mask, vint16m8_t op1, vint16m8_t op2, size_t vl);
vint32mf2_t __riscv_vadd_vv_i32mf2_m (vbool64_t mask, vint32mf2_t op1, vint32mf2_t op2, size_t vl);
vint32m1_t __riscv_vadd_vv_i32m1_m (vbool32_t mask, vint32m1_t op1, vint32m1_t op2, size_t vl);
vint32m2_t __riscv_vadd_vv_i32m2_m (vbool16_t mask, vint32m2_t op1, vint32m2_t op2, size_t vl);
vint32m4_t __riscv_vadd_vv_i32m4_m (vbool8_t mask, vint32m4_t op1, vint32m4_t op2, size_t vl);
vint32m8_t __riscv_vadd_vv_i32m8_m (vbool4_t mask, vint32m8_t op1, vint32m8_t op2, size_t vl);
vint64m1_t __riscv_vadd_vv_i64m1_m (vbool64_t mask, vint64m1_t op1, vint64m1_t op2, size_t vl);
vint64m2_t __riscv_vadd_vv_i64m2_m (vbool32_t mask, vint64m2_t op1, vint64m2_t op2, size_t vl);
vint64m4_t __riscv_vadd_vv_i64m4_m (vbool16_t mask, vint64m4_t op1, vint64m4_t op2, size_t vl);
vint64m8_t __riscv_vadd_vv_i64m8_m (vbool8_t mask, vint64m8_t op1, vint64m8_t op2, size_t vl);
vuint8mf8_t __riscv_vadd_vv_u8mf8_m (vbool64_t mask, vuint8mf8_t op1, vuint8mf8_t op2, size_t vl);
vuint8mf4_t __riscv_vadd_vv_u8mf4_m (vbool32_t mask, vuint8mf4_t op1, vuint8mf4_t op2, size_t vl);
vuint8mf2_t __riscv_vadd_vv_u8mf2_m (vbool16_t mask, vuint8mf2_t op1, vuint8mf2_t op2, size_t vl);
vuint8m1_t __riscv_vadd_vv_u8m1_m (vbool8_t mask, vuint8m1_t op1, vuint8m1_t op2, size_t vl);
vuint8m2_t __riscv_vadd_vv_u8m2_m (vbool4_t mask, vuint8m2_t op1, vuint8m2_t op2, size_t vl);
vuint8m4_t __riscv_vadd_vv_u8m4_m (vbool2_t mask, vuint8m4_t op1, vuint8m4_t op2, size_t vl);
vuint8m8_t __riscv_vadd_vv_u8m8_m (vbool1_t mask, vuint8m8_t op1, vuint8m8_t op2, size_t vl);
vuint16mf4_t __riscv_vadd_vv_u16mf4_m (vbool64_t mask, vuint16mf4_t op1, vuint16mf4_t op2, size_t vl);
vuint16mf2_t __riscv_vadd_vv_u16mf2_m (vbool32_t mask, vuint16mf2_t op1, vuint16mf2_t op2, size_t vl);
vuint16m1_t __riscv_vadd_vv_u16m1_m (vbool16_t mask, vuint16m1_t op1, vuint16m1_t op2, size_t vl);
vuint16m2_t __riscv_vadd_vv_u16m2_m (vbool8_t mask, vuint16m2_t op1, vuint16m2_t op2, size_t vl);
vuint16m4_t __riscv_vadd_vv_u16m4_m (vbool4_t mask, vuint16m4_t op1, vuint16m4_t op2, size_t vl);
vuint16m8_t __riscv_vadd_vv_u16m8_m (vbool2_t mask, vuint16m8_t op1, vuint16m8_t op2, size_t vl);
vuint32mf2_t __riscv_vadd_vv_u32mf2_m (vbool64_t mask, vuint32mf2_t op1, vuint32mf2_t op2, size_t vl);
vuint32m1_t __riscv_vadd_vv_u32m1_m (vbool32_t mask, vuint32m1_t op1, vuint32m2_t op2, size_t vl);
vuint32m2_t __riscv_vadd_vv_u32m2_m (vbool16_t mask, vuint32m2_t op1, vuint32m2_t op2, size_t vl);
vuint32m4_t __riscv_vadd_vv_u32m4_m (vbool8_t mask, vuint32m4_t op1, vuint32m4_t op2, size_t vl);
vuint32m8_t __riscv_vadd_vv_u32m8_m (vbool4_t mask, vuint32m8_t op1, vuint32m8_t op2, size_t vl);
vuint64m1_t __riscv_vadd_vv_u64m1_m (vbool64_t mask, vuint64m1_t op1, vuint64m1_t op2, size_t vl);
vuint64m2_t __riscv_vadd_vv_u64m2_m (vbool32_t mask, vuint64m2_t op1, vuint64m2_t op2, size_t vl);
vuint64m4_t __riscv_vadd_vv_u64m4_m (vbool16_t mask, vuint64m4_t op1, vuint64m4_t op2, size_t vl);
vuint64m8_t __riscv_vadd_vv_u64m8_m (vbool8_t mask, vuint64m8_t op1, vuint64m8_t op2, size_t vl);
```

**Parameters:**
- `mask`: Boolean mask vector controlling which elements are processed
- `op1`: First integer vector (addend)
- `op2`: Second integer vector (addend)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector with addition applied only to masked elements; unmasked elements retain values from `op1`

**Supported Types:** Signed and unsigned integers (8, 16, 32, 64-bit) with LMUL from mf8 to m8

### vadd_vx_m
**Masked Vector-Scalar Integer Addition**

Performs element-wise addition of a scalar to each element of an integer vector with mask control: `result[i] = mask[i] ? (op1[i] + op2) : op1[i]`

```c
vint8mf8_t __riscv_vadd_vx_i8mf8_m (vbool64_t mask, vint8mf8_t op1, int8_t op2, size_t vl);
vint8mf4_t __riscv_vadd_vx_i8mf4_m (vbool32_t mask, vint8mf4_t op1, int8_t op2, size_t vl);
vint8mf2_t __riscv_vadd_vx_i8mf2_m (vbool16_t mask, vint8mf2_t op1, int8_t op2, size_t vl);
vint8m1_t __riscv_vadd_vx_i8m1_m (vbool8_t mask, vint8m1_t op1, int8_t op2, size_t vl);
vint8m2_t __riscv_vadd_vx_i8m2_m (vbool4_t mask, vint8m2_t op1, int8_t op2, size_t vl);
vint8m4_t __riscv_vadd_vx_i8m4_m (vbool2_t mask, vint8m4_t op1, int8_t op2, size_t vl);
vint8m8_t __riscv_vadd_vx_i8m8_m (vbool1_t mask, vint8m8_t op1, int8_t op2, size_t vl);
vint16mf4_t __riscv_vadd_vx_i16mf4_m (vbool64_t mask, vint16mf4_t op1, int16_t op2, size_t vl);
vint16mf2_t __riscv_vadd_vx_i16mf2_m (vbool32_t mask, vint16mf2_t op1, int16_t op2, size_t vl);
vint16m1_t __riscv_vadd_vx_i16m1_m (vbool16_t mask, vint16m1_t op1, int16_t op2, size_t vl);
vint16m2_t __riscv_vadd_vx_i16m2_m (vbool8_t mask, vint16m2_t op1, int16_t op2, size_t vl);
vint16m4_t __riscv_vadd_vx_i16m4_m (vbool4_t mask, vint16m4_t op1, int16_t op2, size_t vl);
vint16m8_t __riscv_vadd_vx_i16m8_m (vbool2_t mask, vint16m8_t op1, int16_t op2, size_t vl);
vint32mf2_t __riscv_vadd_vx_i32mf2_m (vbool64_t mask, vint32mf2_t op1, int32_t op2, size_t vl);
vint32m1_t __riscv_vadd_vx_i32m1_m (vbool32_t mask, vint32m1_t op1, int32_t op2, size_t vl);
vint32m2_t __riscv_vadd_vx_i32m2_m (vbool16_t mask, vint32m2_t op1, int32_t op2, size_t vl);
vint32m4_t __riscv_vadd_vx_i32m4_m (vbool8_t mask, vint32m4_t op1, int32_t op2, size_t vl);
vint32m8_t __riscv_vadd_vx_i32m8_m (vbool4_t mask, vint32m8_t op1, int32_t op2, size_t vl);
vint64m1_t __riscv_vadd_vx_i64m1_m (vbool64_t mask, vint64m1_t op1, int64_t op2, size_t vl);
vint64m2_t __riscv_vadd_vx_i64m2_m (vbool32_t mask, vint64m2_t op1, int64_t op2, size_t vl);
vint64m4_t __riscv_vadd_vx_i64m4_m (vbool16_t mask, vint64m4_t op1, int64_t op2, size_t vl);
vint64m8_t __riscv_vadd_vx_i64m8_m (vbool8_t mask, vint64m8_t op1, int64_t op2, size_t vl);
vuint8mf8_t __riscv_vadd_vx_u8mf8_m (vbool64_t mask, vuint8mf8_t op1, uint8_t op2, size_t vl);
vuint8mf4_t __riscv_vadd_vx_u8mf4_m (vbool32_t mask, vuint8mf4_t op1, uint8_t op2, size_t vl);
vuint8mf2_t __riscv_vadd_vx_u8mf2_m (vbool16_t mask, vuint8mf2_t op1, uint8_t op2, size_t vl);
vuint8m1_t __riscv_vadd_vx_u8m1_m (vbool8_t mask, vuint8m1_t op1, uint8_t op2, size_t vl);
vuint8m2_t __riscv_vadd_vx_u8m2_m (vbool4_t mask, vuint8m2_t op1, uint8_t op2, size_t vl);
vuint8m4_t __riscv_vadd_vx_u8m4_m (vbool2_t mask, vuint8m4_t op1, uint8_t op2, size_t vl);
vuint8m8_t __riscv_vadd_vx_u8m8_m (vbool1_t mask, vuint8m8_t op1, uint8_t op2, size_t vl);
vuint16mf4_t __riscv_vadd_vx_u16mf4_m (vbool64_t mask, vuint16mf4_t op1, uint16_t op2, size_t vl);
vuint16mf2_t __riscv_vadd_vx_u16mf2_m (vbool32_t mask, vuint16mf2_t op1, uint16_t op2, size_t vl);
vuint16m1_t __riscv_vadd_vx_u16m1_m (vbool16_t mask, vuint16m1_t op1, uint16_t op2, size_t vl);
vuint16m2_t __riscv_vadd_vx_u16m2_m (vbool8_t mask, vuint16m2_t op1, uint16_t op2, size_t vl);
vuint16m4_t __riscv_vadd_vx_u16m4_m (vbool4_t mask, vuint16m4_t op1, uint16_t op2, size_t vl);
vuint16m8_t __riscv_vadd_vx_u16m8_m (vbool2_t mask, vuint16m8_t op1, uint16_t op2, size_t vl);
vuint32mf2_t __riscv_vadd_vx_u32mf2_m (vbool64_t mask, vuint32mf2_t op1, uint32_t op2, size_t vl);
vuint32m1_t __riscv_vadd_vx_u32m1_m (vbool32_t mask, vuint32m1_t op1, uint32_t op2, size_t vl);
vuint32m2_t __riscv_vadd_vx_u32m2_m (vbool16_t mask, vuint32m2_t op1, uint32_t op2, size_t vl);
vuint32m4_t __riscv_vadd_vx_u32m4_m (vbool8_t mask, vuint32m4_t op1, uint32_t op2, size_t vl);
vuint32m8_t __riscv_vadd_vx_u32m8_m (vbool4_t mask, vuint32m8_t op1, uint32_t op2, size_t vl);
vuint64m1_t __riscv_vadd_vx_u64m1_m (vbool64_t mask, vuint64m1_t op1, uint64_t op2, size_t vl);
vuint64m2_t __riscv_vadd_vx_u64m2_m (vbool32_t mask, vuint64m2_t op1, uint64_t op2, size_t vl);
vuint64m4_t __riscv_vadd_vx_u64m4_m (vbool16_t mask, vuint64m4_t op1, uint64_t op2, size_t vl);
vuint64m8_t __riscv_vadd_vx_u64m8_m (vbool8_t mask, vuint64m8_t op1, uint64_t op2, size_t vl);
```

**Parameters:**
- `mask`: Boolean mask vector controlling which elements are processed
- `op1`: Integer vector (addend)
- `op2`: Scalar integer value (addend)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector with addition applied only to masked elements; unmasked elements retain values from `op1`

**Supported Types:** Signed and unsigned integers (8, 16, 32, 64-bit) with LMUL from mf8 to m8

### vfadd_vv_m
**Masked Vector-Vector Floating-Point Addition**

Performs element-wise addition of two floating-point vectors with mask control: `result[i] = mask[i] ? (op1[i] + op2[i]) : op1[i]`

```c
vfloat16mf4_t __riscv_vfadd_vv_f16mf4_m (vbool64_t mask, vfloat16mf4_t op1, vfloat16mf4_t op2, size_t vl);
vfloat16mf2_t __riscv_vfadd_vv_f16mf2_m (vbool32_t mask, vfloat16mf2_t op1, vfloat16mf2_t op2, size_t vl);
vfloat16m1_t __riscv_vfadd_vv_f16m1_m (vbool16_t mask, vfloat16m1_t op1, vfloat16m1_t op2, size_t vl);
vfloat16m2_t __riscv_vfadd_vv_f16m2_m (vbool8_t mask, vfloat16m2_t op1, vfloat16m2_t op2, size_t vl);
vfloat16m4_t __riscv_vfadd_vv_f16m4_m (vbool4_t mask, vfloat16m4_t op1, vfloat16m4_t op2, size_t vl);
vfloat16m8_t __riscv_vfadd_vv_f16m8_m (vbool2_t mask, vfloat16m8_t op1, vfloat16m8_t op2, size_t vl);
vfloat32mf2_t __riscv_vfadd_vv_f32mf2_m (vbool64_t mask, vfloat32mf2_t op1, vfloat32mf2_t op2, size_t vl);
vfloat32m1_t __riscv_vfadd_vv_f32m1_m (vbool32_t mask, vfloat32m1_t op1, vfloat32m1_t op2, size_t vl);
vfloat32m2_t __riscv_vfadd_vv_f32m2_m (vbool16_t mask, vfloat32m2_t op1, vfloat32m2_t op2, size_t vl);
vfloat32m4_t __riscv_vfadd_vv_f32m4_m (vbool8_t mask, vfloat32m4_t op1, vfloat32m4_t op2, size_t vl);
vfloat32m8_t __riscv_vfadd_vv_f32m8_m (vbool4_t mask, vfloat32m8_t op1, vfloat32m8_t op2, size_t vl);
vfloat64m1_t __riscv_vfadd_vv_f64m1_m (vbool64_t mask, vfloat64m1_t op1, vfloat64m1_t op2, size_t vl);
vfloat64m2_t __riscv_vfadd_vv_f64m2_m (vbool32_t mask, vfloat64m2_t op1, vfloat64m2_t op2, size_t vl);
vfloat64m4_t __riscv_vfadd_vv_f64m4_m (vbool16_t mask, vfloat64m4_t op1, vfloat64m4_t op2, size_t vl);
vfloat64m8_t __riscv_vfadd_vv_f64m8_m (vbool8_t mask, vfloat64m8_t op1, vfloat64m8_t op2, size_t vl);
```

**Parameters:**
- `mask`: Boolean mask vector controlling which elements are processed
- `op1`: First floating-point vector (addend)
- `op2`: Second floating-point vector (addend)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector with addition applied only to masked elements; unmasked elements retain values from `op1`

**Supported Types:** FP16, FP32, FP64 with LMUL from mf4 to m8

### vfadd_vf_m
**Masked Vector-Scalar Floating-Point Addition**

Performs element-wise addition of a scalar to each element of a floating-point vector with mask control: `result[i] = mask[i] ? (op1[i] + op2) : op1[i]`

```c
vfloat16mf4_t __riscv_vfadd_vf_f16mf4_m (vbool64_t mask, vfloat16mf4_t op1, float16_t op2, size_t vl);
vfloat16mf2_t __riscv_vfadd_vf_f16mf2_m (vbool32_t mask, vfloat16mf2_t op1, float16_t op2, size_t vl);
vfloat16m1_t __riscv_vfadd_vf_f16m1_m (vbool16_t mask, vfloat16m1_t op1, float16_t op2, size_t vl);
vfloat16m2_t __riscv_vfadd_vf_f16m2_m (vbool8_t mask, vfloat16m2_t op1, float16_t op2, size_t vl);
vfloat16m4_t __riscv_vfadd_vf_f16m4_m (vbool4_t mask, vfloat16m4_t op1, float16_t op2, size_t vl);
vfloat16m8_t __riscv_vfadd_vf_f16m8_m (vbool2_t mask, vfloat16m8_t op1, float16_t op2, size_t vl);
vfloat32mf2_t __riscv_vfadd_vf_f32mf2_m (vbool64_t mask, vfloat32mf2_t op1, float32_t op2, size_t vl);
vfloat32m1_t __riscv_vfadd_vf_f32m1_m (vbool32_t mask, vfloat32m1_t op1, float32_t op2, size_t vl);
vfloat32m2_t __riscv_vfadd_vf_f32m2_m (vbool16_t mask, vfloat32m2_t op1, float32_t op2, size_t vl);
vfloat32m4_t __riscv_vfadd_vf_f32m4_m (vbool8_t mask, vfloat32m4_t op1, float32_t op2, size_t vl);
vfloat32m8_t __riscv_vfadd_vf_f32m8_m (vbool4_t mask, vfloat32m8_t op1, float32_t op2, size_t vl);
vfloat64m1_t __riscv_vfadd_vf_f64m1_m (vbool64_t mask, vfloat64m1_t op1, float64_t op2, size_t vl);
vfloat64m2_t __riscv_vfadd_vf_f64m2_m (vbool32_t mask, vfloat64m2_t op1, float64_t op2, size_t vl);
vfloat64m4_t __riscv_vfadd_vf_f64m4_m (vbool16_t mask, vfloat64m4_t op1, float64_t op2, size_t vl);
vfloat64m8_t __riscv_vfadd_vf_f64m8_m (vbool8_t mask, vfloat64m8_t op1, float64_t op2, size_t vl);
```

**Parameters:**
- `mask`: Boolean mask vector controlling which elements are processed
- `op1`: Floating-point vector (addend)
- `op2`: Scalar floating-point value (addend)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector with addition applied only to masked elements; unmasked elements retain values from `op1`

**Supported Types:** FP16, FP32, FP64 with LMUL from mf4 to m8

## VECTOR_MUL

### vmul_vv
**Vector-Vector Integer Multiplication**

Performs element-wise multiplication of two integer vectors: `result[i] = op1[i] * op2[i]`

```c
vint8mf8_t __riscv_vmul_vv_i8mf8 (vint8mf8_t op1, vint8mf8_t op2, size_t vl);
vint8mf4_t __riscv_vmul_vv_i8mf4 (vint8mf4_t op1, vint8mf4_t op2, size_t vl);
vint8mf2_t __riscv_vmul_vv_i8mf2 (vint8mf2_t op1, vint8mf2_t op2, size_t vl);
vint8m1_t __riscv_vmul_vv_i8m1 (vint8m1_t op1, vint8m1_t op2, size_t vl);
vint8m2_t __riscv_vmul_vv_i8m2 (vint8m2_t op1, vint8m2_t op2, size_t vl);
vint8m4_t __riscv_vmul_vv_i8m4 (vint8m4_t op1, vint8m4_t op2, size_t vl);
vint8m8_t __riscv_vmul_vv_i8m8 (vint8m8_t op1, vint8m8_t op2, size_t vl);
vint16mf4_t __riscv_vmul_vv_i16mf4 (vint16mf4_t op1, vint16mf4_t op2, size_t vl);
vint16mf2_t __riscv_vmul_vv_i16mf2 (vint16mf2_t op1, vint16mf2_t op2, size_t vl);
vint16m1_t __riscv_vmul_vv_i16m1 (vint16m1_t op1, vint16m1_t op2, size_t vl);
vint16m2_t __riscv_vmul_vv_i16m2 (vint16m2_t op1, vint16m2_t op2, size_t vl);
vint16m4_t __riscv_vmul_vv_i16m4 (vint16m4_t op1, vint16m4_t op2, size_t vl);
vint16m8_t __riscv_vmul_vv_i16m8 (vint16m8_t op1, vint16m8_t op2, size_t vl);
vint32mf2_t __riscv_vmul_vv_i32mf2 (vint32mf2_t op1, vint32mf2_t op2, size_t vl);
vint32m1_t __riscv_vmul_vv_i32m1 (vint32m1_t op1, vint32m1_t op2, size_t vl);
vint32m2_t __riscv_vmul_vv_i32m2 (vint32m2_t op1, vint32m2_t op2, size_t vl);
vint32m4_t __riscv_vmul_vv_i32m4 (vint32m4_t op1, vint32m4_t op2, size_t vl);
vint32m8_t __riscv_vmul_vv_i32m8 (vint32m8_t op1, vint32m8_t op2, size_t vl);
vint64m1_t __riscv_vmul_vv_i64m1 (vint64m1_t op1, vint64m1_t op2, size_t vl);
vint64m2_t __riscv_vmul_vv_i64m2 (vint64m2_t op1, vint64m2_t op2, size_t vl);
vint64m4_t __riscv_vmul_vv_i64m4 (vint64m4_t op1, vint64m4_t op2, size_t vl);
vint64m8_t __riscv_vmul_vv_i64m8 (vint64m8_t op1, vint64m8_t op2, size_t vl);
vuint8mf8_t __riscv_vmul_vv_u8mf8 (vuint8mf8_t op1, vuint8mf8_t op2, size_t vl);
vuint8mf4_t __riscv_vmul_vv_u8mf4 (vuint8mf4_t op1, vuint8mf4_t op2, size_t vl);
vuint8mf2_t __riscv_vmul_vv_u8mf2 (vuint8mf2_t op1, vuint8mf2_t op2, size_t vl);
vuint8m1_t __riscv_vmul_vv_u8m1 (vuint8m1_t op1, vuint8m1_t op2, size_t vl);
vuint8m2_t __riscv_vmul_vv_u8m2 (vuint8m2_t op1, vuint8m2_t op2, size_t vl);
vuint8m4_t __riscv_vmul_vv_u8m4 (vuint8m4_t op1, vuint8m4_t op2, size_t vl);
vuint8m8_t __riscv_vmul_vv_u8m8 (vuint8m8_t op1, vuint8m8_t op2, size_t vl);
vuint16mf4_t __riscv_vmul_vv_u16mf4 (vuint16mf4_t op1, vuint16mf4_t op2, size_t vl);
vuint16mf2_t __riscv_vmul_vv_u16mf2 (vuint16mf2_t op1, vuint16mf2_t op2, size_t vl);
vuint16m1_t __riscv_vmul_vv_u16m1 (vuint16m1_t op1, vuint16m1_t op2, size_t vl);
vuint16m2_t __riscv_vmul_vv_u16m2 (vuint16m2_t op1, vuint16m2_t op2, size_t vl);
vuint16m4_t __riscv_vmul_vv_u16m4 (vuint16m4_t op1, vuint16m4_t op2, size_t vl);
vuint16m8_t __riscv_vmul_vv_u16m8 (vuint16m8_t op1, vuint16m8_t op2, size_t vl);
vuint32mf2_t __riscv_vmul_vv_u32mf2 (vuint32mf2_t op1, vuint32mf2_t op2, size_t vl);
vuint32m1_t __riscv_vmul_vv_u32m1 (vuint32m1_t op1, vuint32m1_t op2, size_t vl);
vuint32m2_t __riscv_vmul_vv_u32m2 (vuint32m2_t op1, vuint32m2_t op2, size_t vl);
vuint32m4_t __riscv_vmul_vv_u32m4 (vuint32m4_t op1, vuint32m4_t op2, size_t vl);
vuint32m8_t __riscv_vmul_vv_u32m8 (vuint32m8_t op1, vuint32m8_t op2, size_t vl);
vuint64m1_t __riscv_vmul_vv_u64m1 (vuint64m1_t op1, vuint64m1_t op2, size_t vl);
vuint64m2_t __riscv_vmul_vv_u64m2 (vuint64m2_t op1, vuint64m2_t op2, size_t vl);
vuint64m4_t __riscv_vmul_vv_u64m4 (vuint64m4_t op1, vuint64m4_t op2, size_t vl);
vuint64m8_t __riscv_vmul_vv_u64m8 (vuint64m8_t op1, vuint64m8_t op2, size_t vl);
```

**Parameters:**
- `op1`: First integer vector (multiplicand)
- `op2`: Second integer vector (multiplier)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector containing the element-wise product `op1 * op2`

**Supported Types:** Signed and unsigned integers (8, 16, 32, 64-bit) with LMUL from mf8 to m8

### vmul_vx
**Vector-Scalar Integer Multiplication**

Performs element-wise multiplication of each element of an integer vector by a scalar: `result[i] = op1[i] * op2`

```c
vint8mf8_t __riscv_vmul_vx_i8mf8 (vint8mf8_t op1, int8_t op2, size_t vl);
vint8mf4_t __riscv_vmul_vx_i8mf4 (vint8mf4_t op1, int8_t op2, size_t vl);
vint8mf2_t __riscv_vmul_vx_i8mf2 (vint8mf2_t op1, int8_t op2, size_t vl);
vint8m1_t __riscv_vmul_vx_i8m1 (vint8m1_t op1, int8_t op2, size_t vl);
vint8m2_t __riscv_vmul_vx_i8m2 (vint8m2_t op1, int8_t op2, size_t vl);
vint8m4_t __riscv_vmul_vx_i8m4 (vint8m4_t op1, int8_t op2, size_t vl);
vint8m8_t __riscv_vmul_vx_i8m8 (vint8m8_t op1, int8_t op2, size_t vl);
vint16mf4_t __riscv_vmul_vx_i16mf4 (vint16mf4_t op1, int16_t op2, size_t vl);
vint16mf2_t __riscv_vmul_vx_i16mf2 (vint16mf2_t op1, int16_t op2, size_t vl);
vint16m1_t __riscv_vmul_vx_i16m1 (vint16m1_t op1, int16_t op2, size_t vl);
vint16m2_t __riscv_vmul_vx_i16m2 (vint16m2_t op1, int16_t op2, size_t vl);
vint16m4_t __riscv_vmul_vx_i16m4 (vint16m4_t op1, int16_t op2, size_t vl);
vint16m8_t __riscv_vmul_vx_i16m8 (vint16m8_t op1, int16_t op2, size_t vl);
vint32mf2_t __riscv_vmul_vx_i32mf2 (vint32mf2_t op1, int32_t op2, size_t vl);
vint32m1_t __riscv_vmul_vx_i32m1 (vint32m1_t op1, int32_t op2, size_t vl);
vint32m2_t __riscv_vmul_vx_i32m2 (vint32m2_t op1, int32_t op2, size_t vl);
vint32m4_t __riscv_vmul_vx_i32m4 (vint32m4_t op1, int32_t op2, size_t vl);
vint32m8_t __riscv_vmul_vx_i32m8 (vint32m8_t op1, int32_t op2, size_t vl);
vint64m1_t __riscv_vmul_vx_i64m1 (vint64m1_t op1, int64_t op2, size_t vl);
vint64m2_t __riscv_vmul_vx_i64m2 (vint64m2_t op1, int64_t op2, size_t vl);
vint64m4_t __riscv_vmul_vx_i64m4 (vint64m4_t op1, int64_t op2, size_t vl);
vint64m8_t __riscv_vmul_vx_i64m8 (vint64m8_t op1, int64_t op2, size_t vl);
vuint8mf8_t __riscv_vmul_vx_u8mf8 (vuint8mf8_t op1, uint8_t op2, size_t vl);
vuint8mf4_t __riscv_vmul_vx_u8mf4 (vuint8mf4_t op1, uint8_t op2, size_t vl);
vuint8mf2_t __riscv_vmul_vx_u8mf2 (vuint8mf2_t op1, uint8_t op2, size_t vl);
vuint8m1_t __riscv_vmul_vx_u8m1 (vuint8m1_t op1, uint8_t op2, size_t vl);
vuint8m2_t __riscv_vmul_vx_u8m2 (vuint8m2_t op1, uint8_t op2, size_t vl);
vuint8m4_t __riscv_vmul_vx_u8m4 (vuint8m4_t op1, uint8_t op2, size_t vl);
vuint8m8_t __riscv_vmul_vx_u8m8 (vuint8m8_t op1, uint8_t op2, size_t vl);
vuint16mf4_t __riscv_vmul_vx_u16mf4 (vuint16mf4_t op1, uint16_t op2, size_t vl);
vuint16mf2_t __riscv_vmul_vx_u16mf2 (vuint16mf2_t op1, uint16_t op2, size_t vl);
vuint16m1_t __riscv_vmul_vx_u16m1 (vuint16m1_t op1, uint16_t op2, size_t vl);
vuint16m2_t __riscv_vmul_vx_u16m2 (vuint16m2_t op1, uint16_t op2, size_t vl);
vuint16m4_t __riscv_vmul_vx_u16m4 (vuint16m4_t op1, uint16_t op2, size_t vl);
vuint16m8_t __riscv_vmul_vx_u16m8 (vuint16m8_t op1, uint16_t op2, size_t vl);
vuint32mf2_t __riscv_vmul_vx_u32mf2 (vuint32mf2_t op1, uint32_t op2, size_t vl);
vuint32m1_t __riscv_vmul_vx_u32m1 (vuint32m1_t op1, uint32_t op2, size_t vl);
vuint32m2_t __riscv_vmul_vx_u32m2 (vuint32m2_t op1, uint32_t op2, size_t vl);
vuint32m4_t __riscv_vmul_vx_u32m4 (vuint32m4_t op1, uint32_t op2, size_t vl);
vuint32m8_t __riscv_vmul_vx_u32m8 (vuint32m8_t op1, uint32_t op2, size_t vl);
vuint64m1_t __riscv_vmul_vx_u64m1 (vuint64m1_t op1, uint64_t op2, size_t vl);
vuint64m2_t __riscv_vmul_vx_u64m2 (vuint64m2_t op1, uint64_t op2, size_t vl);
vuint64m4_t __riscv_vmul_vx_u64m4 (vuint64m4_t op1, uint64_t op2, size_t vl);
vuint64m8_t __riscv_vmul_vx_u64m8 (vuint64m8_t op1, uint64_t op2, size_t vl);
```

**Parameters:**
- `op1`: Integer vector (multiplicand)
- `op2`: Scalar integer value (multiplier)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector containing the element-wise product `op1[i] * op2`

**Supported Types:** Signed and unsigned integers (8, 16, 32, 64-bit) with LMUL from mf8 to m8

### vfmul_vv
**Vector-Vector Floating-Point Multiplication**

Performs element-wise multiplication of two floating-point vectors: `result[i] = op1[i] * op2[i]`

```c
vfloat16mf4_t __riscv_vfmul_vv_f16mf4 (vfloat16mf4_t op1, vfloat16mf4_t op2, size_t vl);
vfloat16mf2_t __riscv_vfmul_vv_f16mf2 (vfloat16mf2_t op1, vfloat16mf2_t op2, size_t vl);
vfloat16m1_t __riscv_vfmul_vv_f16m1 (vfloat16m1_t op1, vfloat16m1_t op2, size_t vl);
vfloat16m2_t __riscv_vfmul_vv_f16m2 (vfloat16m2_t op1, vfloat16m2_t op2, size_t vl);
vfloat16m4_t __riscv_vfmul_vv_f16m4 (vfloat16m4_t op1, vfloat16m4_t op2, size_t vl);
vfloat16m8_t __riscv_vfmul_vv_f16m8 (vfloat16m8_t op1, vfloat16m8_t op2, size_t vl);
vfloat32mf2_t __riscv_vfmul_vv_f32mf2 (vfloat32mf2_t op1, vfloat32mf2_t op2, size_t vl);
vfloat32m1_t __riscv_vfmul_vv_f32m1 (vfloat32m1_t op1, vfloat32m1_t op2, size_t vl);
vfloat32m2_t __riscv_vfmul_vv_f32m2 (vfloat32m2_t op1, vfloat32m2_t op2, size_t vl);
vfloat32m4_t __riscv_vfmul_vv_f32m4 (vfloat32m4_t op1, vfloat32m4_t op2, size_t vl);
vfloat32m8_t __riscv_vfmul_vv_f32m8 (vfloat32m8_t op1, vfloat32m8_t op2, size_t vl);
vfloat64m1_t __riscv_vfmul_vv_f64m1 (vfloat64m1_t op1, vfloat64m1_t op2, size_t vl);
vfloat64m2_t __riscv_vfmul_vv_f64m2 (vfloat64m2_t op1, vfloat64m2_t op2, size_t vl);
vfloat64m4_t __riscv_vfmul_vv_f64m4 (vfloat64m4_t op1, vfloat64m4_t op2, size_t vl);
vfloat64m8_t __riscv_vfmul_vv_f64m8 (vfloat64m8_t op1, vfloat64m8_t op2, size_t vl);
```

**Parameters:**
- `op1`: First floating-point vector (multiplicand)
- `op2`: Second floating-point vector (multiplier)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector containing the element-wise product `op1 * op2`

**Supported Types:** FP16, FP32, FP64 with LMUL from mf4 to m8

### vfmul_vf
**Vector-Scalar Floating-Point Multiplication**

Performs element-wise multiplication of each element of a floating-point vector by a scalar: `result[i] = op1[i] * op2`

```c
vfloat16mf4_t __riscv_vfmul_vf_f16mf4 (vfloat16mf4_t op1, float16_t op2, size_t vl);
vfloat16mf2_t __riscv_vfmul_vf_f16mf2 (vfloat16mf2_t op1, float16_t op2, size_t vl);
vfloat16m1_t __riscv_vfmul_vf_f16m1 (vfloat16m1_t op1, float16_t op2, size_t vl);
vfloat16m2_t __riscv_vfmul_vf_f16m2 (vfloat16m2_t op1, float16_t op2, size_t vl);
vfloat16m4_t __riscv_vfmul_vf_f16m4 (vfloat16m4_t op1, float16_t op2, size_t vl);
vfloat16m8_t __riscv_vfmul_vf_f16m8 (vfloat16m8_t op1, float16_t op2, size_t vl);
vfloat32mf2_t __riscv_vfmul_vf_f32mf2 (vfloat32mf2_t op1, float32_t op2, size_t vl);
vfloat32m1_t __riscv_vfmul_vf_f32m1 (vfloat32m1_t op1, float32_t op2, size_t vl);
vfloat32m2_t __riscv_vfmul_vf_f32m2 (vfloat32m2_t op1, float32_t op2, size_t vl);
vfloat32m4_t __riscv_vfmul_vf_f32m4 (vfloat32m4_t op1, float32_t op2, size_t vl);
vfloat32m8_t __riscv_vfmul_vf_f32m8 (vfloat32m8_t op1, float32_t op2, size_t vl);
vfloat64m1_t __riscv_vfmul_vf_f64m1 (vfloat64m1_t op1, float64_t op2, size_t vl);
vfloat64m2_t __riscv_vfmul_vf_f64m2 (vfloat64m2_t op1, float64_t op2, size_t vl);
vfloat64m4_t __riscv_vfmul_vf_f64m4 (vfloat64m4_t op1, float64_t op2, size_t vl);
vfloat64m8_t __riscv_vfmul_vf_f64m8 (vfloat64m8_t op1, float64_t op2, size_t vl);
```

**Parameters:**
- `op1`: Floating-point vector (multiplicand)
- `op2`: Scalar floating-point value (multiplier)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector containing the element-wise product `op1[i] * op2`

**Supported Types:** FP16, FP32, FP64 with LMUL from mf4 to m8

## VECTOR_MUL_MASKED

### vmul_vv_m
**Masked Vector-Vector Integer Multiplication**

Performs element-wise multiplication of two integer vectors with mask control: `result[i] = mask[i] ? (op1[i] * op2[i]) : op1[i]`

```c
vint8mf8_t __riscv_vmul_vv_i8mf8_m (vbool64_t mask, vint8mf8_t op1, vint8mf8_t op2, size_t vl);
vint8mf4_t __riscv_vmul_vv_i8mf4_m (vbool32_t mask, vint8mf4_t op1, vint8mf4_t op2, size_t vl);
vint8mf2_t __riscv_vmul_vv_i8mf2_m (vbool16_t mask, vint8mf2_t op1, vint8mf2_t op2, size_t vl);
vint8m1_t __riscv_vmul_vv_i8m1_m (vbool8_t mask, vint8m1_t op1, vint8m1_t op2, size_t vl);
vint8m2_t __riscv_vmul_vv_i8m2_m (vbool4_t mask, vint8m2_t op1, vint8m2_t op2, size_t vl);
vint8m4_t __riscv_vmul_vv_i8m4_m (vbool2_t mask, vint8m4_t op1, vint8m4_t op2, size_t vl);
vint8m8_t __riscv_vmul_vv_i8m8_m (vbool1_t mask, vint8m8_t op1, vint8m8_t op2, size_t vl);
vint16mf4_t __riscv_vmul_vv_i16mf4_m (vbool64_t mask, vint16mf4_t op1, vint16mf4_t op2, size_t vl);
vint16mf2_t __riscv_vmul_vv_i16mf2_m (vbool32_t mask, vint16mf2_t op1, vint16mf2_t op2, size_t vl);
vint16m1_t __riscv_vmul_vv_i16m1_m (vbool16_t mask, vint16m1_t op1, vint16m1_t op2, size_t vl);
vint16m2_t __riscv_vmul_vv_i16m2_m (vbool8_t mask, vint16m2_t op1, vint16m2_t op2, size_t vl);
vint16m4_t __riscv_vmul_vv_i16m4_m (vbool4_t mask, vint16m4_t op1, vint16m4_t op2, size_t vl);
vint16m8_t __riscv_vmul_vv_i16m8_m (vbool2_t mask, vint16m8_t op1, vint16m8_t op2, size_t vl);
vint32mf2_t __riscv_vmul_vv_i32mf2_m (vbool64_t mask, vint32mf2_t op1, vint32mf2_t op2, size_t vl);
vint32m1_t __riscv_vmul_vv_i32m1_m (vbool32_t mask, vint32m1_t op1, vint32m1_t op2, size_t vl);
vint32m2_t __riscv_vmul_vv_i32m2_m (vbool16_t mask, vint32m2_t op1, vint32m2_t op2, size_t vl);
vint32m4_t __riscv_vmul_vv_i32m4_m (vbool8_t mask, vint32m4_t op1, vint32m4_t op2, size_t vl);
vint32m8_t __riscv_vmul_vv_i32m8_m (vbool4_t mask, vint32m8_t op1, vint32m8_t op2, size_t vl);
vint64m1_t __riscv_vmul_vv_i64m1_m (vbool64_t mask, vint64m1_t op1, vint64m1_t op2, size_t vl);
vint64m2_t __riscv_vmul_vv_i64m2_m (vbool32_t mask, vint64m2_t op1, vint64m2_t op2, size_t vl);
vint64m4_t __riscv_vmul_vv_i64m4_m (vbool16_t mask, vint64m4_t op1, vint64m4_t op2, size_t vl);
vint64m8_t __riscv_vmul_vv_i64m8_m (vbool8_t mask, vint64m8_t op1, vint64m8_t op2, size_t vl);
vuint8mf8_t __riscv_vmul_vv_u8mf8_m (vbool64_t mask, vuint8mf8_t op1, vuint8mf8_t op2, size_t vl);
vuint8mf4_t __riscv_vmul_vv_u8mf4_m (vbool32_t mask, vuint8mf4_t op1, vuint8mf4_t op2, size_t vl);
vuint8mf2_t __riscv_vmul_vv_u8mf2_m (vbool16_t mask, vuint8mf2_t op1, vuint8mf2_t op2, size_t vl);
vuint8m1_t __riscv_vmul_vv_u8m1_m (vbool8_t mask, vuint8m1_t op1, vuint8m1_t op2, size_t vl);
vuint8m2_t __riscv_vmul_vv_u8m2_m (vbool4_t mask, vuint8m2_t op1, vuint8m2_t op2, size_t vl);
vuint8m4_t __riscv_vmul_vv_u8m4_m (vbool2_t mask, vuint8m4_t op1, vuint8m4_t op2, size_t vl);
vuint8m8_t __riscv_vmul_vv_u8m8_m (vbool1_t mask, vuint8m8_t op1, vuint8m8_t op2, size_t vl);
vuint16mf4_t __riscv_vmul_vv_u16mf4_m (vbool64_t mask, vuint16mf4_t op1, vuint16mf4_t op2, size_t vl);
vuint16mf2_t __riscv_vmul_vv_u16mf2_m (vbool32_t mask, vuint16mf2_t op1, vuint16mf2_t op2, size_t vl);
vuint16m1_t __riscv_vmul_vv_u16m1_m (vbool16_t mask, vuint16m1_t op1, vuint16m1_t op2, size_t vl);
vuint16m2_t __riscv_vmul_vv_u16m2_m (vbool8_t mask, vuint16m2_t op1, vuint16m2_t op2, size_t vl);
vuint16m4_t __riscv_vmul_vv_u16m4_m (vbool4_t mask, vuint16m4_t op1, vuint16m4_t op2, size_t vl);
vuint16m8_t __riscv_vmul_vv_u16m8_m (vbool2_t mask, vuint16m8_t op1, vuint16m8_t op2, size_t vl);
vuint32mf2_t __riscv_vmul_vv_u32mf2_m (vbool64_t mask, vuint32mf2_t op1, vuint32mf2_t op2, size_t vl);
vuint32m1_t __riscv_vmul_vv_u32m1_m (vbool32_t mask, vuint32m1_t op1, vuint32m1_t op2, size_t vl);
vuint32m2_t __riscv_vmul_vv_u32m2_m (vbool16_t mask, vuint32m2_t op1, vuint32m2_t op2, size_t vl);
vuint32m4_t __riscv_vmul_vv_u32m4_m (vbool8_t mask, vuint32m4_t op1, vuint32m4_t op2, size_t vl);
vuint32m8_t __riscv_vmul_vv_u32m8_m (vbool4_t mask, vuint32m8_t op1, vuint32m8_t op2, size_t vl);
vuint64m1_t __riscv_vmul_vv_u64m1_m (vbool64_t mask, vuint64m1_t op1, vuint64m1_t op2, size_t vl);
vuint64m2_t __riscv_vmul_vv_u64m2_m (vbool32_t mask, vuint64m2_t op1, vuint64m2_t op2, size_t vl);
vuint64m4_t __riscv_vmul_vv_u64m4_m (vbool16_t mask, vuint64m4_t op1, vuint64m4_t op2, size_t vl);
vuint64m8_t __riscv_vmul_vv_u64m8_m (vbool8_t mask, vuint64m8_t op1, vuint64m8_t op2, size_t vl);
```

**Parameters:**
- `mask`: Boolean mask vector controlling which elements are processed
- `op1`: First integer vector (multiplicand)
- `op2`: Second integer vector (multiplier)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector with multiplication applied only to masked elements; unmasked elements retain values from `op1`

**Supported Types:** Signed and unsigned integers (8, 16, 32, 64-bit) with LMUL from mf8 to m8

### vmul_vx_m
**Masked Vector-Scalar Integer Multiplication**

Performs element-wise multiplication of each element of an integer vector by a scalar with mask control: `result[i] = mask[i] ? (op1[i] * op2) : op1[i]`

```c
vint8mf8_t __riscv_vmul_vx_i8mf8_m (vbool64_t mask, vint8mf8_t op1, int8_t op2, size_t vl);
vint8mf4_t __riscv_vmul_vx_i8mf4_m (vbool32_t mask, vint8mf4_t op1, int8_t op2, size_t vl);
vint8mf2_t __riscv_vmul_vx_i8mf2_m (vbool16_t mask, vint8mf2_t op1, int8_t op2, size_t vl);
vint8m1_t __riscv_vmul_vx_i8m1_m (vbool8_t mask, vint8m1_t op1, int8_t op2, size_t vl);
vint8m2_t __riscv_vmul_vx_i8m2_m (vbool4_t mask, vint8m2_t op1, int8_t op2, size_t vl);
vint8m4_t __riscv_vmul_vx_i8m4_m (vbool2_t mask, vint8m4_t op1, int8_t op2, size_t vl);
vint8m8_t __riscv_vmul_vx_i8m8_m (vbool1_t mask, vint8m8_t op1, int8_t op2, size_t vl);
vint16mf4_t __riscv_vmul_vx_i16mf4_m (vbool64_t mask, vint16mf4_t op1, int16_t op2, size_t vl);
vint16mf2_t __riscv_vmul_vx_i16mf2_m (vbool32_t mask, vint16mf2_t op1, int16_t op2, size_t vl);
vint16m1_t __riscv_vmul_vx_i16m1_m (vbool16_t mask, vint16m1_t op1, int16_t op2, size_t vl);
vint16m2_t __riscv_vmul_vx_i16m2_m (vbool8_t mask, vint16m2_t op1, int16_t op2, size_t vl);
vint16m4_t __riscv_vmul_vx_i16m4_m (vbool4_t mask, vint16m4_t op1, int16_t op2, size_t vl);
vint16m8_t __riscv_vmul_vx_i16m8_m (vbool2_t mask, vint16m8_t op1, int16_t op2, size_t vl);
vint32mf2_t __riscv_vmul_vx_i32mf2_m (vbool64_t mask, vint32mf2_t op1, int32_t op2, size_t vl);
vint32m1_t __riscv_vmul_vx_i32m1_m (vbool32_t mask, vint32m1_t op1, int32_t op2, size_t vl);
vint32m2_t __riscv_vmul_vx_i32m2_m (vbool16_t mask, vint32m2_t op1, int32_t op2, size_t vl);
vint32m4_t __riscv_vmul_vx_i32m4_m (vbool8_t mask, vint32m4_t op1, int32_t op2, size_t vl);
vint32m8_t __riscv_vmul_vx_i32m8_m (vbool4_t mask, vint32m8_t op1, int32_t op2, size_t vl);
vint64m1_t __riscv_vmul_vx_i64m1_m (vbool64_t mask, vint64m1_t op1, int64_t op2, size_t vl);
vint64m2_t __riscv_vmul_vx_i64m2_m (vbool32_t mask, vint64m2_t op1, int64_t op2, size_t vl);
vint64m4_t __riscv_vmul_vx_i64m4_m (vbool16_t mask, vint64m4_t op1, int64_t op2, size_t vl);
vint64m8_t __riscv_vmul_vx_i64m8_m (vbool8_t mask, vint64m8_t op1, int64_t op2, size_t vl);
vuint8mf8_t __riscv_vmul_vx_u8mf8_m (vbool64_t mask, vuint8mf8_t op1, uint8_t op2, size_t vl);
vuint8mf4_t __riscv_vmul_vx_u8mf4_m (vbool32_t mask, vuint8mf4_t op1, uint8_t op2, size_t vl);
vuint8mf2_t __riscv_vmul_vx_u8mf2_m (vbool16_t mask, vuint8mf2_t op1, uint8_t op2, size_t vl);
vuint8m1_t __riscv_vmul_vx_u8m1_m (vbool8_t mask, vuint8m1_t op1, uint8_t op2, size_t vl);
vuint8m2_t __riscv_vmul_vx_u8m2_m (vbool4_t mask, vuint8m2_t op1, uint8_t op2, size_t vl);
vuint8m4_t __riscv_vmul_vx_u8m4_m (vbool2_t mask, vuint8m4_t op1, uint8_t op2, size_t vl);
vuint8m8_t __riscv_vmul_vx_u8m8_m (vbool1_t mask, vuint8m8_t op1, uint8_t op2, size_t vl);
vuint16mf4_t __riscv_vmul_vx_u16mf4_m (vbool64_t mask, vuint16mf4_t op1, uint16_t op2, size_t vl);
vuint16mf2_t __riscv_vmul_vx_u16mf2_m (vbool32_t mask, vuint16mf2_t op1, uint16_t op2, size_t vl);
vuint16m1_t __riscv_vmul_vx_u16m1_m (vbool16_t mask, vuint16m1_t op1, uint16_t op2, size_t vl);
vuint16m2_t __riscv_vmul_vx_u16m2_m (vbool8_t mask, vuint16m2_t op1, uint16_t op2, size_t vl);
vuint16m4_t __riscv_vmul_vx_u16m4_m (vbool4_t mask, vuint16m4_t op1, uint16_t op2, size_t vl);
vuint16m8_t __riscv_vmul_vx_u16m8_m (vbool2_t mask, vuint16m8_t op1, uint16_t op2, size_t vl);
vuint32mf2_t __riscv_vmul_vx_u32mf2_m (vbool64_t mask, vuint32mf2_t op1, uint32_t op2, size_t vl);
vuint32m1_t __riscv_vmul_vx_u32m1_m (vbool32_t mask, vuint32m1_t op1, uint32_t op2, size_t vl);
vuint32m2_t __riscv_vmul_vx_u32m2_m (vbool16_t mask, vuint32m2_t op1, uint32_t op2, size_t vl);
vuint32m4_t __riscv_vmul_vx_u32m4_m (vbool8_t mask, vuint32m4_t op1, uint32_t op2, size_t vl);
vuint32m8_t __riscv_vmul_vx_u32m8_m (vbool4_t mask, vuint32m8_t op1, uint32_t op2, size_t vl);
vuint64m1_t __riscv_vmul_vx_u64m1_m (vbool64_t mask, vuint64m1_t op1, uint64_t op2, size_t vl);
vuint64m2_t __riscv_vmul_vx_u64m2_m (vbool32_t mask, vuint64m2_t op1, uint64_t op2, size_t vl);
vuint64m4_t __riscv_vmul_vx_u64m4_m (vbool16_t mask, vuint64m4_t op1, uint64_t op2, size_t vl);
vuint64m8_t __riscv_vmul_vx_u64m8_m (vbool8_t mask, vuint64m8_t op1, uint64_t op2, size_t vl);
```

**Parameters:**
- `mask`: Boolean mask vector controlling which elements are processed
- `op1`: Integer vector (multiplicand)
- `op2`: Scalar integer value (multiplier)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector with multiplication applied only to masked elements; unmasked elements retain values from `op1`

**Supported Types:** Signed and unsigned integers (8, 16, 32, 64-bit) with LMUL from mf8 to m8

### vfmul_vv_m
**Masked Vector-Vector Floating-Point Multiplication**

Performs element-wise multiplication of two floating-point vectors with mask control: `result[i] = mask[i] ? (op1[i] * op2[i]) : op1[i]`

```c
vfloat16mf4_t __riscv_vfmul_vv_f16mf4_m (vbool64_t mask, vfloat16mf4_t op1, vfloat16mf4_t op2, size_t vl);
vfloat16mf2_t __riscv_vfmul_vv_f16mf2_m (vbool32_t mask, vfloat16mf2_t op1, vfloat16mf2_t op2, size_t vl);
vfloat16m1_t __riscv_vfmul_vv_f16m1_m (vbool16_t mask, vfloat16m1_t op1, vfloat16m1_t op2, size_t vl);
vfloat16m2_t __riscv_vfmul_vv_f16m2_m (vbool8_t mask, vfloat16m2_t op1, vfloat16m2_t op2, size_t vl);
vfloat16m4_t __riscv_vfmul_vv_f16m4_m (vbool4_t mask, vfloat16m4_t op1, vfloat16m4_t op2, size_t vl);
vfloat16m8_t __riscv_vfmul_vv_f16m8_m (vbool2_t mask, vfloat16m8_t op1, vfloat16m8_t op2, size_t vl);
vfloat32mf2_t __riscv_vfmul_vv_f32mf2_m (vbool64_t mask, vfloat32mf2_t op1, vfloat32mf2_t op2, size_t vl);
vfloat32m1_t __riscv_vfmul_vv_f32m1_m (vbool32_t mask, vfloat32m1_t op1, vfloat32m1_t op2, size_t vl);
vfloat32m2_t __riscv_vfmul_vv_f32m2_m (vbool16_t mask, vfloat32m2_t op1, vfloat32m2_t op2, size_t vl);
vfloat32m4_t __riscv_vfmul_vv_f32m4_m (vbool8_t mask, vfloat32m4_t op1, vfloat32m4_t op2, size_t vl);
vfloat32m8_t __riscv_vfmul_vv_f32m8_m (vbool4_t mask, vfloat32m8_t op1, vfloat32m8_t op2, size_t vl);
vfloat64m1_t __riscv_vfmul_vv_f64m1_m (vbool64_t mask, vfloat64m1_t op1, vfloat64m1_t op2, size_t vl);
vfloat64m2_t __riscv_vfmul_vv_f64m2_m (vbool32_t mask, vfloat64m2_t op1, vfloat64m2_t op2, size_t vl);
vfloat64m4_t __riscv_vfmul_vv_f64m4_m (vbool16_t mask, vfloat64m4_t op1, vfloat64m4_t op2, size_t vl);
vfloat64m8_t __riscv_vfmul_vv_f64m8_m (vbool8_t mask, vfloat64m8_t op1, vfloat64m8_t op2, size_t vl);
```

**Parameters:**
- `mask`: Boolean mask vector controlling which elements are processed
- `op1`: First floating-point vector (multiplicand)
- `op2`: Second floating-point vector (multiplier)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector with multiplication applied only to masked elements; unmasked elements retain values from `op1`

**Supported Types:** FP16, FP32, FP64 with LMUL from mf4 to m8

### vfmul_vf_m
**Masked Vector-Scalar Floating-Point Multiplication**

Performs element-wise multiplication of each element of a floating-point vector by a scalar with mask control: `result[i] = mask[i] ? (op1[i] * op2) : op1[i]`

```c
vfloat16mf4_t __riscv_vfmul_vf_f16mf4_m (vbool64_t mask, vfloat16mf4_t op1, float16_t op2, size_t vl);
vfloat16mf2_t __riscv_vfmul_vf_f16mf2_m (vbool32_t mask, vfloat16mf2_t op1, float16_t op2, size_t vl);
vfloat16m1_t __riscv_vfmul_vf_f16m1_m (vbool16_t mask, vfloat16m1_t op1, float16_t op2, size_t vl);
vfloat16m2_t __riscv_vfmul_vf_f16m2_m (vbool8_t mask, vfloat16m2_t op1, float16_t op2, size_t vl);
vfloat16m4_t __riscv_vfmul_vf_f16m4_m (vbool4_t mask, vfloat16m4_t op1, float16_t op2, size_t vl);
vfloat16m8_t __riscv_vfmul_vf_f16m8_m (vbool2_t mask, vfloat16m8_t op1, float16_t op2, size_t vl);
vfloat32mf2_t __riscv_vfmul_vf_f32mf2_m (vbool64_t mask, vfloat32mf2_t op1, float32_t op2, size_t vl);
vfloat32m1_t __riscv_vfmul_vf_f32m1_m (vbool32_t mask, vfloat32m1_t op1, float32_t op2, size_t vl);
vfloat32m2_t __riscv_vfmul_vf_f32m2_m (vbool16_t mask, vfloat32m2_t op1, float32_t op2, size_t vl);
vfloat32m4_t __riscv_vfmul_vf_f32m4_m (vbool8_t mask, vfloat32m4_t op1, float32_t op2, size_t vl);
vfloat32m8_t __riscv_vfmul_vf_f32m8_m (vbool4_t mask, vfloat32m8_t op1, float32_t op2, size_t vl);
vfloat64m1_t __riscv_vfmul_vf_f64m1_m (vbool64_t mask, vfloat64m1_t op1, float64_t op2, size_t vl);
vfloat64m2_t __riscv_vfmul_vf_f64m2_m (vbool32_t mask, vfloat64m2_t op1, float64_t op2, size_t vl);
vfloat64m4_t __riscv_vfmul_vf_f64m4_m (vbool16_t mask, vfloat64m4_t op1, float64_t op2, size_t vl);
vfloat64m8_t __riscv_vfmul_vf_f64m8_m (vbool8_t mask, vfloat64m8_t op1, float64_t op2, size_t vl);
```

**Parameters:**
- `mask`: Boolean mask vector controlling which elements are processed
- `op1`: Floating-point vector (multiplicand)
- `op2`: Scalar floating-point value (multiplier)
- `vl`: Vector length (number of elements to process)

**Returns:** Vector with multiplication applied only to masked elements; unmasked elements retain values from `op1`

**Supported Types:** FP16, FP32, FP64 with LMUL from mf4 to m8