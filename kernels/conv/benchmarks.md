# Benchmarks of Convolution (Conv) Kernel

---

## Input = [1, 4, 4] | Filters = 1 | Kernel = 3 × 3 | Stride = 1 | Padding = 0

### Input Configuration

| Parameter        | Value |
|------------------|-------|
| Input Shape      | [1, 4, 4] |
| Number of Filters| 1 |
| Kernel Size      | 3 × 3 |
| Stride           | 1 |
| Padding          | 0 |
| Output Elements  | 4 |

### Performance Results

| Implementation           | Cycles | Speedup |
|--------------------------|--------|---------|
| Scalar                   | 1,875  | 1.00×   |
| Vector (M1)              | 2,290  | 0.82×   |
| Vector (M2)              | 2,211  | 0.85×   |
| Vector (M4)              | 2,222  | 0.84×   |
| Vector (M8)              | 2,218  | 0.85×   |
| IM2COL + GEMM (M8)       | 3,014  | 0.64×   |

---

## Input = [1, 8, 8] | Filters = 2 | Kernel = 3 × 3 | Stride = 1 | Padding = 0

### Input Configuration

| Parameter        | Value |
|------------------|-------|
| Input Shape      | [1, 8, 8] |
| Number of Filters| 2 |
| Kernel Size      | 3 × 3 |
| Stride           | 1 |
| Padding          | 0 |
| Output Elements  | 72 |

### Performance Results

| Implementation           | Cycles | Speedup |
|--------------------------|--------|---------|
| Scalar                   | 20,625 | 1.00×   |
| Vector (M1)              | 11,628 | 1.77×   |
| Vector (M2)              | 11,645 | 1.77×   |
| Vector (M4)              | 11,575 | 1.78×   |
| Vector (M8)              | 11,564 | 1.78×   |
| IM2COL + GEMM (M8)       | 5,629  | 3.67×   |

---

## Input = [1, 32, 32] | Filters = 6 | Kernel = 5 × 5 | Stride = 1 | Padding = 0

### Input Configuration

| Parameter        | Value |
|------------------|-------|
| Input Shape      | [1, 32, 32] |
| Number of Filters| 6 |
| Kernel Size      | 5 × 5 |
| Stride           | 1 |
| Padding          | 0 |
| Output Elements  | 4,704 |

### Performance Results

| Implementation           | Cycles    | Speedup |
|--------------------------|-----------|---------|
| Scalar                   | 2,969,298 | 1.00×   |
| Vector (M1)              | 553,675   | 5.36×   |
| Vector (M2)              | 555,793   | 5.34×   |
| Vector (M4)              | 554,216   | 5.36×   |
| Vector (M8)              | 554,234   | 5.36×   |
| IM2COL + GEMM (M8)       | 134,698   | 22.04×  |

---

## Input = [1, 32, 32] | Filters = 32 | Kernel = 5 × 5 | Stride = 1 | Padding = 0

### Input Configuration

| Parameter        | Value |
|------------------|-------|
| Input Shape      | [1, 32, 32] |
| Number of Filters| 32 |
| Kernel Size      | 5 × 5 |
| Stride           | 1 |
| Padding          | 0 |
| Output Elements  | 25,088 |

### Performance Results

| Implementation           | Cycles     | Speedup |
|--------------------------|------------|---------|
| Scalar                   | 15,826,364 | 1.00×   |
| Vector (M1)              | 823,869    | 19.21×  |
| Vector (M2)              | 827,691    | 19.12×  |
| Vector (M4)              | 825,432    | 19.17×  |
| Vector (M8)              | 825,358    | 19.18×  |
| IM2COL + GEMM (M8)       | 534,683    | 29.60×  |
