# Benchmarks of MaxPool Kernel

---

## Shape = [1, 16, 16] | Pool = 2 × 2 | Stride = 1

### Input Configuration

| Parameter        | Value |
|------------------|-------|
| Input Shape      | [1, 16, 16] |
| Pool Size        | 2 × 2 |
| Stride           | 1 |
| Output Elements  | 225 |

### Performance Results

| Implementation              | Cycles | Speedup |
|-----------------------------|--------|---------|
| Scalar                      | 17,721 | 1.00×   |
| Vector (M1)                 | 2,319  | 7.64×   |
| Vector (M2)                 | 2,305  | 7.69×   |
| Vector (M4)                 | 2,326  | 7.62×   |
| Vector (M8)                 | 2,286  | 7.75×   |
| Vector (M1) Tiled           | 2,645  | 6.70×   |
| Vector (M2) Tiled           | 2,656  | 6.67×   |
| Vector (M4) Tiled           | 2,670  | 6.64×   |
| Vector (M8) Tiled           | 2,653  | 6.68×   |

---

## Shape = [1, 16, 16] | Pool = 2 × 2 | Stride = 2

### Input Configuration

| Parameter        | Value |
|------------------|-------|
| Input Shape      | [1, 16, 16] |
| Pool Size        | 2 × 2 |
| Stride           | 2 |
| Output Elements  | 64 |

### Performance Results

| Implementation              | Cycles | Speedup |
|-----------------------------|--------|---------|
| Scalar                      | 5,895  | 1.00×   |
| Vector (M1)                 | 1,848  | 3.19×   |
| Vector (M2)                 | 1,833  | 3.22×   |
| Vector (M4)                 | 1,848  | 3.19×   |
| Vector (M8)                 | 1,836  | 3.21×   |
| Vector (M1) Tiled           | 2,044  | 2.88×   |
| Vector (M2) Tiled           | 2,069  | 2.85×   |
| Vector (M4) Tiled           | 2,027  | 2.91×   |
| Vector (M8) Tiled           | 2,070  | 2.85×   |

---

## Shape = [1, 64, 64] | Pool = 2 × 2 | Stride = 2

### Input Configuration

| Parameter        | Value |
|------------------|-------|
| Input Shape      | [1, 64, 64] |
| Pool Size        | 2 × 2 |
| Stride           | 2 |
| Output Elements  | 1,024 |

### Performance Results

| Implementation              | Cycles | Speedup |
|-----------------------------|--------|---------|
| Scalar                      | 82,213 | 1.00×   |
| Vector (M1)                 | 12,284 | 6.69×   |
| Vector (M2)                 | 12,264 | 6.70×   |
| Vector (M4)                 | 12,295 | 6.69×   |
| Vector (M8)                 | 12,254 | 6.71×   |
| Vector (M1) Tiled           | 12,970 | 6.34×   |
| Vector (M2) Tiled           | 12,963 | 6.34×   |
| Vector (M4) Tiled           | 12,976 | 6.34×   |
| Vector (M8) Tiled           | 12,940 | 6.35×   |

---

## Shape = [1, 64, 64] | Pool = 2 × 2 | Stride = 1

### Input Configuration

| Parameter        | Value |
|------------------|-------|
| Input Shape      | [1, 64, 64] |
| Pool Size        | 2 × 2 |
| Stride           | 1 |
| Output Elements  | 3,969 |

### Performance Results

| Implementation              | Cycles  | Speedup |
|-----------------------------|---------|---------|
| Scalar                      | 295,391 | 1.00×   |
| Vector (M1)                 | 17,660  | 16.73×  |
| Vector (M2)                 | 12,601  | 23.44×  |
| Vector (M4)                 | 12,598  | 23.45×  |
| Vector (M8)                 | 12,578  | 23.48×  |
| Vector (M1) Tiled           | 20,895  | 14.14×  |
| Vector (M2) Tiled           | 14,840  | 19.91×  |
| Vector (M4) Tiled           | 14,822  | 19.93×  |
| Vector (M8) Tiled           | 14,845  | 19.90×  |
