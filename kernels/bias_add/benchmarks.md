# Benchmarks of Bias Add Kernel

---

## Size = 1 × 8 × 32 × 32

### Input Tensor Configuration

| Parameter              | Value |
|------------------------|-------|
| Batch Size             | 1     |
| Number of Channels     | 8     |
| Width                  | 32    |
| Height                 | 32    |
| Tensor Shape (N×C×W×H) | 1 × 8 × 32 × 32 |

### Performance Results

| Implementation | Cycles  | Speedup |
|---------------|---------|---------|
| Scalar        | 103,759 | 1.00×   |
| Vector (M1)   | 10,496  | 9.89×   |
| Vector (M2)   | 6,779   | 15.31×  |
| Vector (M4)   | 5,537   | 18.74×  |
| Vector (M8)   | 4,931   | 21.04×  |

---

## Size = 1 × 8 × 64 × 64

### Input Tensor Configuration

| Parameter              | Value |
|------------------------|-------|
| Batch Size             | 1     |
| Number of Channels     | 8     |
| Width                  | 64    |
| Height                 | 64    |
| Tensor Shape (N×C×W×H) | 1 × 8 × 64 × 64 |

### Performance Results

| Implementation | Cycles  | Speedup |
|---------------|---------|---------|
| Scalar        | 414,096 | 1.00×   |
| Vector (M1)   | 41,216  | 10.05×  |
| Vector (M2)   | 26,363  | 15.71×  |
| Vector (M4)   | 21,473  | 19.28×  |
| Vector (M8)   | 19,043  | 21.75×  |

