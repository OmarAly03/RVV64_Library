# RVV Wrapper Overview
This document describes the high‑level C++ wrappers around the RISC‑V Vector (RVV) 1.0 intrinsics used in this project.

Each wrapper family provides:

- A **type‑safe**, **LMUL‑aware** interface over the raw RVV intrinsics.
- A single “unified” entry point (e.g. `VECTOR_ADD`, `VECTOR_MUL`, `VECTOR_SLL`) that:
  - Dispatches between vector–vector vs. vector–scalar forms.
  - Dispatches between masked vs. unmasked forms (where applicable).
  - Selects the appropriate intrinsic variant based on element type `T` and `LMUL`.

The following table summarizes the categories covered in this file and the **core RVV instruction families** they wrap (names follow the RVV spec’s base mnemonics, not concrete function signatures):

| Category                                                  | Core Instruction Families (Mnemonic Roots)                            |
| --------------------------------------------------------- | --------------------------------------------------------------------- |
| Vector Single-Width Floating-Point Add/Subtract          | `vfadd`, `vfsub`                                                      |
| Vector Single-Width Integer Add/Subtract                 | `vadd`, `vsub`                                                        |
| Vector Single-Width Floating-Point Multiply              | `vfmul`                                                               |
| Vector Single-Width Integer Multiply                     | `vmul`                                                                |
| Vector Single-Width Logical Shift                        | `vsll`                                                                |
| Vector Single-Width Integer Bitwise Logical AND          | `vand`                                                                |
| Vector Single-Width Integer Logical Right Shift          | `vsrl`                                                                |
| Vector Compress                                          | `vcompress`                                                           |
| Vector Move and Broadcast                                | `vmv`, `vfmv`                                                         |
| Vector Min/Max                                           | `vmax`, `vmin`, `vfmax`, `vfmin`                                      |
| Vector Mask Logical Operations                           | `vmand`, `vmor`, `vmxor`                                             |
| Vector Fused Multiply-Accumulate / Multiply-Add          | `vfmacc`, `vmacc`, `vfmsac`, `vfnmacc`, `vfmadd`                      |
| Vector Integer and Floating-Point Comparison             | families such as `vmslt`, `vmsltu`, `vmseq`                           |
| Vector Indexed Load (Gather)                             | `vluxei`, `vloxei`                                                    |
| Vector Population Count                                  | `vcpop`                                                               |
| Vector Index Generation                                  | `vid`                                                                 |
| Vector Floating-Point Reduction Sum                      | `vfredsum`                                                            |
| Vector Load                                              | `vle`                                                                 |
| Vector Store and Indexed Store                           | `vse`, `vsuxei`, `vsoxei`                                             |
| Vector Narrowing Shift-Right                             | `vnsra`, `vnsrl`                                                      |
| Vector Slide Down                                        | `vslidedown`                                                          |
| Vector Length Configuration                              | `vsetvl`, `vsetvli`, `vsetvlmax`                                      |
| Vector Type Reinterpretation                             | Bit reinterpret via vector type casts (no direct RVV mnemonic)        |

The sections below describe, for each category, the **public wrapper APIs** and the element types they support.

---

## Vector Single-Width Floating-Point Add/Subtract Functions

Wrappers:

- `VECTOR_ADD<T, LMUL, VecType, Op2Type>`
  - Unified add wrapper (covers):
    - Vector–vector add
    - Vector–scalar add
    - All maskless float / integer types supported in `rvv_arithmetic.hpp`

- `VECTOR_ADD_MASKED<T, LMUL, MaskType, VecType, Op2Type>`
  - Unified masked add wrapper (covers):
    - Masked vector–vector add
    - Masked vector–scalar add
    - All maskable float / integer types supported in `rvv_arithmetic.hpp`

- `VECTOR_SUB<T, LMUL, VecType, Op2Type>`
  - Unified subtract wrapper (covers):
    - Vector–vector sub
    - Vector–scalar sub
    - All maskless float / integer types supported in `rvv_arithmetic.hpp`

- `VECTOR_SUB_MASKED<T, LMUL, MaskType, VecType, Op2Type>`
  - Unified masked subtract wrapper (covers):
    - Masked vector–vector sub
    - Masked vector–scalar sub
    - All maskable float / integer types supported in `rvv_arithmetic.hpp`

---

## Vector Single-Width Integer Add/Subtract Functions

Wrappers:

- Same as above (`VECTOR_ADD*`, `VECTOR_SUB*`) for integer element types:
  - `int8_t`, `int16_t`, `int32_t`, `int64_t`
  - `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`

---

## Vector Single-Width Floating-Point Multiply Functions

Wrappers:

- `VECTOR_MUL<T, LMUL, VecType, Op2Type>`
  - Unified multiply wrapper (covers):
    - Vector–vector multiply
    - Vector–scalar multiply
    - All maskless float / integer types supported in `rvv_arithmetic.hpp`

- `VECTOR_MUL_MASKED<T, LMUL, MaskType, VecType, Op2Type>`
  - Unified masked multiply wrapper (covers):
    - Masked vector–vector multiply
    - Masked vector–scalar multiply
    - All maskable float / integer types supported in `rvv_arithmetic.hpp`

---

## Vector Single-Width Integer Multiply Functions

Wrappers:

- Same as above (`VECTOR_MUL*`) for integer element types:
  - `int8_t`, `int16_t`, `int32_t`, `int64_t`
  - `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`

---

## Vector Single-Width Logical Shift Instructions

Wrappers:

- `VECTOR_SLL<T, LMUL, VecType, ShiftType>`
  - Unified left shift wrapper (covers):
    - Vector–vector left shift
    - Vector–scalar left shift
    - All supported signed / unsigned integer element types

- `VECTOR_SLL_MASKED<T, LMUL, MaskType, VecType, ShiftType>`
  - Unified masked left shift wrapper (covers):
    - Masked vector–vector left shift
    - Masked vector–scalar left shift
    - All supported signed / unsigned integer element types

---

## Vector Single-Width Integer Bitwise Logical AND Functions

Wrappers:

- `VECTOR_AND_VX<T, LMUL, VecType>`
  - Vector–scalar bitwise AND (`v & x`) for unsigned integer element types:
    - `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`

---

## Vector Single-Width Integer Logical Right Shift Instructions

Wrappers:

- `VECTOR_SRL_VX<T, LMUL, VecType>`
  - Vector–scalar logical right shift (`v >> x`) for unsigned integer element types:
    - `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`

---

## Vector Compress Instructions

Wrappers:

- `VECTOR_COMPRESS<T, LMUL, VecType, BoolType>`
  - Compresses elements from `src` under `mask` into a compacted vector.
  - Supported element types `T`:
    - Floating-point: `_Float16`, `float`, `double`
    - Signed integers: `int8_t`, `int16_t`, `int32_t`, `int64_t`
    - Unsigned integers: `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`

## Vector Move and Broadcast Instructions

Wrappers:

- `VECTOR_MOVE<T, LMUL, SrcType>`
  - Unified move/broadcast wrapper (covers):
    - Scalar → vector broadcast (full vector)
    - Vector → vector copy
    - Automatically selects between `VECTOR_BROADCAST` and `VECTOR_COPY` based on `SrcType`

- `VECTOR_COPY<T, LMUL, VecType>`
  - Vector-to-vector copy:
    - Copies all elements from `src` to a new vector
    - Supported element types `T`:
      - Floating-point: `float`, `double`
      - Signed integers: `int8_t`, `int16_t`, `int32_t`, `int64_t`

- `VECTOR_BROADCAST<T, LMUL>`
  - Scalar-to-vector broadcast:
    - Fills all elements of the vector with the same scalar value
    - Supported element types `T`:
      - Floating-point: `float`, `double`
      - Signed integers: `int8_t`, `int16_t`, `int32_t`, `int64_t`

- `VECTOR_EXTRACT_SCALAR<T, LMUL, VecType>`
  - Vector-to-scalar extraction:
    - Returns the first element of a vector register
    - Supported element types `T`:
      - Floating-point: `float`, `double`
      - Signed integers: `int8_t`, `int16_t`, `int32_t`, `int64_t`

- `VECTOR_SPLAT<T, LMUL>`
  - Scalar-to-vector “splat”:
    - Sets only the first element to the scalar; remaining lanes are unspecified
    - Mainly useful for mask/length handling and advanced patterns

---

## Vector Min/Max Instructions

Wrappers:

- `VECTOR_MAX<T, LMUL, VecType, Op2Type>`
  - Unified max wrapper (covers):
    - Vector–vector max
    - Vector–scalar max
    - Supported element types `T`:
      - Floating-point: `float`, `double`
      - Signed integers: `int8_t`, `int16_t`, `int32_t`, `int64_t`

- `VECTOR_MIN<T, LMUL, VecType, Op2Type>`
  - Unified min wrapper (covers):
    - Vector–vector min
    - Vector–scalar min
    - Supported element types `T`:
      - Floating-point: `float`, `double`
      - Signed integers: `int8_t`, `int16_t`, `int32_t`, `int64_t`

- `VECTOR_MAX_MASKED<T, LMUL, MaskType, VecType, Op2Type>`
  - Unified masked max wrapper (covers):
    - Masked vector–vector max
    - Masked vector–scalar max
    - Same element type coverage as `VECTOR_MAX`

- (Internal helpers used by the above, typically not called directly):
  - `VECTOR_MAX_VV`, `VECTOR_MAX_VX`
  - `VECTOR_MAX_VV_MASKED`, `VECTOR_MAX_VX_MASKED`
  - `VECTOR_MIN_VV`, `VECTOR_MIN_VX`

---

## Vector Mask Logical Operations

Wrappers:

- `MASK_AND<LMUL, MaskType>`
  - Mask–mask bitwise AND for generic element widths:
    - Uses LMUL-only dispatch to select correct mask type width

- `MASK_AND_E32<T, LMUL, MaskType>`
  - Mask–mask bitwise AND specialized for element width 32:
    - Corrects mask type vs. LMUL mapping for 32‑bit data

- `MASK_OR<LMUL, MaskType>`
  - Mask–mask bitwise OR for generic element widths

- `MASK_OR_E32<T, LMUL, MaskType>`
  - Mask–mask bitwise OR specialized for element width 32

- `MASK_XOR<LMUL, MaskType>`
  - Mask–mask bitwise XOR for generic element widths

---

## Vector Fused Multiply-Accumulate / Multiply-Add Instructions

Wrappers:

- `VECTOR_FMACC<T, LMUL, SrcType, VD, VS2>`
  - Unified fused multiply‑accumulate wrapper (covers):
    - Vector–vector FMACC / MACC–style operations: `vd = vd + vs1 * vs2`
    - Vector–scalar FMACC / MACC–style operations: `vd = vd + scalar * vs2`
    - Internally dispatches between vector–vector and vector–scalar forms
  - Supported element types `T`:
    - Floating-point: `float`, `double`
    - Signed integers: `int8_t`, `int16_t`, `int32_t`, `int64_t`

- `VECTOR_FMADD_VF<T, LMUL, VS1, VD>`
  - Fused multiply‑add with explicit operand order:
    - `vd = vs1 * scalar + vd`
    - Intended for floating‑point and `_Float16` cases where operand order matters
  - Supported element types `T`:
    - Floating-point: `_Float16`, `float`, `double`

---

## Vector Integer and Floating-Point Comparison Instructions

Wrappers:

- `VECTOR_GT<T, LMUL, VecType>`
  - Vector–vector “greater than” comparison:
    - Returns a mask (`vbool*`) where `op1 > op2`
    - Supported element types `T`:
      - Floating-point: `_Float16`, `float`, `double`
      - Signed integers: `int8_t`, `int16_t`, `int32_t`, `int64_t`

- `VECTOR_GT_SCALAR<T, LMUL, VecType>`
  - Vector–scalar “greater than” comparison:
    - Mask where `op1 > scalar`

- `VECTOR_GE<T, LMUL, VecType>`
  - Vector–vector “greater or equal” comparison:
    - Mask where `op1 >= op2`

- `VECTOR_GE_SCALAR<T, LMUL, VecType>`
  - Vector–scalar “greater or equal” comparison:
    - Mask where `op1 >= scalar`

- `VECTOR_LT<T, LMUL, VecType>`
  - Vector–vector “less than” comparison:
    - Mask where `op1 < op2`
    - Supports floating‑point and signed integer `T`

- `VECTOR_LT_SCALAR<T, LMUL, VecType>`
  - Vector–scalar “less than” comparison:
    - Mask where `op1 < scalar`

- `VECTOR_EQ_SCALAR<T, LMUL, VecType>`
  - Vector–scalar equality comparison:
    - Mask where `op1 == scalar`
    - Supported element types `T`:
      - Unsigned integers: `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`
      - Signed integers: `int8_t`, `int16_t`, `int32_t`, `int64_t`

- `VECTOR_LTU_SCALAR<T, LMUL, VecType>`
  - Unsigned vector–scalar “less than” comparison:
    - Mask where `op1 < scalar`
    - Supported element types `T`:
      - Unsigned integers: `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`

---

## Vector Indexed Load (Gather) Instructions

Wrappers:

- `VECTOR_INDEXED_LOAD<T, LMUL, IdxType>`
  - Unordered indexed load (gather), non‑masked:
    - Gathers elements from `base[index[i]]` into a vector
    - Supported element types `T`:
      - Floating-point: `float`, `double`
      - Signed integers: `int32_t`
      - Unsigned integers: `uint32_t`

- `VECTOR_INDEXED_LOAD_MU<T, LMUL, MaskType, VecType, IdxType>`
  - Unordered indexed load (gather), masked undisturbed:
    - Uses `maskedoff` for inactive lanes, preserves them
    - Same element type coverage as `VECTOR_INDEXED_LOAD`, plus:
      - `int64_t`, `uint64_t` (64‑bit variants)

---

## Vector Population Count Instructions

Wrappers:

- `VECTOR_COUNT_POP<BoolType>`
  - Population count over a mask vector:
    - Returns the number of active lanes (`1` bits) in `mask`
    - Supported mask types:
      - `vbool1_t`, `vbool2_t`, `vbool4_t`, `vbool8_t`
      - `vbool16_t`, `vbool32_t`, `vbool64_t`

## Vector Index Generation Instructions

Wrappers:

- `VECTOR_VID<T, LMUL>`
  - Generates a vector of indices `[0, 1, 2, ...]` in the element type `T`
  - Supported element types `T`:
    - Unsigned integers: `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`

---

## Vector Floating-Point Reduction Sum Instructions

Wrappers:

- `VECTOR_VFREDSUM<T, LMUL, VecType, ScalarType>`
  - Reduces a floating-point vector to a scalar using unordered sum:
    - `result = sum(vector) + scalar`
  - Supported element types `T`:
    - Floating-point: `_Float16`, `float`, `double`

---

## Vector Load Instructions

Wrappers:

- `VECTOR_LOAD<T, LMUL>`
  - Unified contiguous load wrapper:
    - Loads a vector from `base[i]` (unit-stride)
    - Supported element types `T`:
      - Floating-point: `float`, `double`
      - Signed integers: `int8_t`, `int16_t`, `int32_t`, `int64_t`
      - Unsigned integers: `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`

- `VECTOR_STRIDED_LOAD<T, LMUL>`
  - Unified strided load wrapper:
    - Loads a vector with byte stride `stride` between elements
    - Supported element types `T`:
      - Floating-point: `float`, `double`
      - Signed integers: `int8_t`, `int16_t`, `int32_t`, `int64_t`

---

## Vector Store and Indexed Store Instructions

Wrappers:

- `VECTOR_STORE<T, LMUL, VecType>`
  - Unified contiguous store wrapper:
    - Stores a full vector `value` to `base[i]` (unit-stride)
    - Supported element types `T`:
      - Floating-point: `_Float16`, `float`, `double`
      - Signed integers: `int8_t`, `int16_t`, `int32_t`, `int64_t`
      - Unsigned integers: `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`

- `VECTOR_STRIDED_STORE<T, LMUL, VecType>`
  - Unified strided store wrapper:
    - Stores a vector with byte stride `stride` between elements
    - Supported element types `T`:
      - Floating-point: `float`, `double`
      - Signed integers: `int8_t`, `int16_t`, `int32_t`, `int64_t`
      - Unsigned integers: `uint16_t`, `uint32_t`, `uint64_t`

- `VECTOR_INDEXED_STORE<T, LMUL, VecType, IndexType>`
  - Unified indexed store (scatter) wrapper:
    - Stores `value[i]` into `base[ index[i] ]` using gather-style indices
    - Automatically dispatches to:
      - `VECTOR_INDEXED_STORE_8`   (8-bit indices)
      - `VECTOR_INDEXED_STORE_16`  (16-bit indices)
      - `VECTOR_INDEXED_STORE_32`  (32-bit indices)
      - `VECTOR_INDEXED_STORE_64`  (64-bit indices)
    - Supported element types `T`:
      - Floating-point: `_Float16`, `float`, `double`
      - Signed integers: `int8_t`, `int16_t`, `int32_t`, `int64_t`
      - Unsigned integers: `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`

---

## Vector Narrowing Shift-Right Instructions

Wrappers:

- `VECTOR_NARROW_SRA<T, LMUL, WideVecType, ShiftType>`
  - Unified *arithmetic* narrowing shift-right wrapper (signed):
    - Covers both:
      - Vector–vector narrowing SRA (`WideVec >> ShiftVec`)
      - Vector–scalar narrowing SRA (`WideVec >> scalar`)
    - Narrows from 2× element width to 1× element width
    - Supported signed integer types `T` (narrow result):
      - `int8_t`, `int16_t`, `int32_t`

- `VECTOR_NARROW_SRL<T, LMUL, WideVecType, ShiftType>`
  - Unified *logical* narrowing shift-right wrapper (unsigned):
    - Covers both:
      - Vector–vector narrowing SRL (`WideVec >> ShiftVec`)
      - Vector–scalar narrowing SRL (`WideVec >> scalar`)
    - Narrows from 2× element width to 1× element width
    - Supported unsigned integer types `T` (narrow result):
      - `uint8_t`, `uint16_t`, `uint32_t`

---

## Vector Slide Down Instructions

Wrappers:

- `VECTOR_SLIDEDOWN<T, LMUL, VecType>`
  - Slides vector elements down by `offset` positions:
    - `dst[i] = src[i + offset]` for active lanes
  - Supported element types `T`:
    - Floating-point: `_Float16`, `float`, `double`
    - Signed integers: `int8_t`, `int16_t`, `int32_t`, `int64_t`
    - Unsigned integers: `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`

---

## Vector Length Configuration Instructions

Wrappers:

- `SET_VECTOR_LENGTH<T, LMUL>`
  - Sets VL for a given element type and LMUL:
    - `vl = vsetvl_eXXmY(avl)`
  - Supported element types `T`:
    - Signed integers: `int8_t`, `int16_t`, `int32_t`, `int64_t`
    - Floating-point: `float`, `double`

- `SET_VECTOR_LENGTH_MAX<T, LMUL>`
  - Sets VL to the hardware maximum for a given element type and LMUL:
    - `vl = vsetvlmax_eXXmY()`
  - Supported element types `T`:
    - Floating-point: `_Float16`, `float`, `double`
    - Signed integers: `int8_t`, `int16_t`, `int32_t`, `int64_t`
    - Unsigned integers: `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`

---

## Vector Type Reinterpretation Instructions

Wrappers:

- `VECTOR_REINTERPRET<TFrom, TTo, LMUL, VecType>`
  - Reinterprets the bits of a vector between signed and unsigned integer types:
    - No change to the underlying bits, only the type/view
  - Supported conversions (`TFrom` → `TTo`):
    - `uint32_t`  ↔ `int32_t`
    - `uint64_t`  ↔ `int64_t`
    - `uint16_t`  ↔ `int16_t`
    - `uint8_t`   ↔ `int8_t`