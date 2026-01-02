# Benchmarks of ReLU Kernel

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
| Scalar        | 12,023 | 1.00×   |
| Vector (M1)   | 1,284  | 9.36×   |
| Vector (M2)   | 909    | 13.23×  |
| Vector (M4)   | 729    | 16.49×  |
| Vector (M8)   | 649    | 18.53×  |

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
| Scalar        | 190,583 | 1.00×   |
| Vector (M1)   | 19,044  | 10.01×  |
| Vector (M2)   | 13,389  | 14.23×  |
| Vector (M4)   | 10,809  | 17.63×  |
| Vector (M8)   | 9,529   | 20.00×  |

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
| Scalar        | 761,975  | 1.00×   |
| Vector (M1)   | 75,876   | 10.04×  |
| Vector (M2)   | 53,325   | 14.29×  |
| Vector (M4)   | 43,065   | 17.69×  |
| Vector (M8)   | 37,945   | 20.08×  |
