# Benchmarks of Leaky ReLU Kernel

---

## Size = 1,024 (32 × 32)

### Input Configuration

| Parameter        | Value |
|------------------|-------|
| Tensor Shape     | 32 × 32 |
| Total Elements   | 1,024 |

### Performance Results

| Implementation | Cycles | Speedup |
|---------------|--------|---------|
| Scalar        | 19,388 | 1.00×   |
| Vector (M1)   | 1,378  | 14.07×  |
| Vector (M2)   | 986    | 19.66×  |
| Vector (M4)   | 740    | 26.20×  |
| Vector (M8)   | 629    | 30.82×  |

---

## Size = 16,384 (128 × 128)

### Input Configuration

| Parameter        | Value |
|------------------|-------|
| Tensor Shape     | 128 × 128 |
| Total Elements   | 16,384 |

### Performance Results

| Implementation | Cycles  | Speedup |
|---------------|---------|---------|
| Scalar        | 307,938 | 1.00×   |
| Vector (M1)   | 20,759  | 14.83×  |
| Vector (M2)   | 14,486  | 21.26×  |
| Vector (M4)   | 10,340  | 29.78×  |
| Vector (M8)   | 8,609   | 35.77×  |

---

## Size = 65,536 (256 × 256)

### Input Configuration

| Parameter        | Value |
|------------------|-------|
| Tensor Shape     | 256 × 256 |
| Total Elements   | 65,536 |

### Performance Results

| Implementation | Cycles   | Speedup |
|---------------|----------|---------|
| Scalar        | 1,230,050 | 1.00×   |
| Vector (M1)   | 82,775   | 14.86×  |
| Vector (M2)   | 57,686   | 21.32×  |
| Vector (M4)   | 41,060   | 29.96×  |
| Vector (M8)   | 34,145   | 36.02×  |
