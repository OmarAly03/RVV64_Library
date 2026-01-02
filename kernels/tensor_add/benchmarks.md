# Benchmarks of Tensor Add Kernel

---

## Size = 1,024

### Input Configuration

| Parameter        | Value |
|------------------|-------|
| Total Elements   | 1,024 |

### Performance Results

| Implementation | Cycles | Speedup |
|---------------|--------|---------|
| Scalar        | 15,938 | 1.00×   |
| Vector (M1)   | 1,646  | 9.68×   |
| Vector (M2)   | 1,131  | 14.09×  |
| Vector (M4)   | 989    | 16.12×  |
| Vector (M8)   | 889    | 17.93×  |

---

## Size = 16,384

### Input Configuration

| Parameter        | Value |
|------------------|-------|
| Total Elements   | 16,384 |

### Performance Results

| Implementation | Cycles  | Speedup |
|---------------|---------|---------|
| Scalar        | 265,236 | 1.00×   |
| Vector (M1)   | 25,166  | 10.54×  |
| Vector (M2)   | 17,451  | 15.20×  |
| Vector (M4)   | 14,909  | 17.79×  |
| Vector (M8)   | 13,609  | 19.49×  |

---

## Size = 65,536

### Input Configuration

| Parameter        | Value |
|------------------|-------|
| Total Elements   | 65,536 |

### Performance Results

| Implementation | Cycles    | Speedup |
|---------------|-----------|---------|
| Scalar        | 1,062,966 | 1.00×   |
| Vector (M1)   | 100,430   | 10.58×  |
| Vector (M2)   | 69,675    | 15.26×  |
| Vector (M4)   | 59,453    | 17.88×  |
| Vector (M8)   | 54,313    | 19.57×  |
