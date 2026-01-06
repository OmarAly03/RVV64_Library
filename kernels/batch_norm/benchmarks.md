# Benchmarks of Batch Normalization (BatchNorm) Kernel

---

## Shape = [1, 6, 14] | Epsilon = 1e-5

### Input Configuration

| Parameter        | Value |
|------------------|-------|
| Input Shape      | [1, 6, 14] |
| Epsilon          | 0.00001 |
| Total Elements   | 84 |

### Performance Results

| Implementation | Cycles | Speedup |
|----------------|--------|---------|
| Scalar         | 1,714  | 1.00×   |
| Vector (M1)    | 330    | 5.19×   |
| Vector (M2)    | 282    | 6.08×   |
| Vector (M4)    | 277    | 6.19×   |
| Vector (M8)    | 250    | 6.86×   |

---

## Shape = [1, 64, 8] | Epsilon = 1e-5

### Input Configuration

| Parameter        | Value |
|------------------|-------|
| Input Shape      | [1, 64, 8] |
| Epsilon          | 0.00001 |
| Total Elements   | 512 |

### Performance Results

| Implementation | Cycles | Speedup |
|----------------|--------|---------|
| Scalar         | 8,876  | 1.00×   |
| Vector (M1)    | 957    | 9.27×   |
| Vector (M2)    | 699    | 12.70×  |
| Vector (M4)    | 596    | 14.89×  |
| Vector (M8)    | 509    | 17.44×  |

---

## Shape = [1, 128, 32] | Epsilon = 1e-5

### Input Configuration

| Parameter        | Value |
|------------------|-------|
| Input Shape      | [1, 128, 32] |
| Epsilon          | 0.00001 |
| Total Elements   | 4,096 |

### Performance Results

| Implementation | Cycles | Speedup |
|----------------|--------|---------|
| Scalar         | 68,914 | 1.00×   |
| Vector (M1)    | 6,353  | 10.85×  |
| Vector (M2)    | 4,226  | 16.31×  |
| Vector (M4)    | 3,249  | 21.21×  |
| Vector (M8)    | 2,755  | 25.01×  |
