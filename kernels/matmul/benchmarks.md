# Benchmarks of Matrix Multiplication (MatMul) Kernel

---

## MATMUL [4 × 4] × [4 × 4]

### Input Configuration

| Parameter            | Value |
|---------------------|-------|
| Matrix A Shape      | 4 × 4 |
| Matrix B Shape      | 4 × 4 |
| Output Matrix Shape | 4 × 4 |

### Performance Results

| Implementation              | Cycles | Speedup |
|-----------------------------|--------|---------|
| Scalar                      | 972    | 1.00×   |
| Vector (M1)                 | 641    | 1.52×   |
| Vector (M1) unrolled        | 175    | 5.55×   |
| Vector (M2)                 | 553    | 1.76×   |
| Vector (M2) unrolled        | 477    | 2.04×   |
| Vector (M4)                 | 596    | 1.63×   |
| Vector (M4) unrolled        | 519    | 1.87×   |
| Vector (M8)                 | 546    | 1.78×   |
| Vector (M8) unrolled        | 438    | 2.22×   |

---

## MATMUL [16 × 16] × [16 × 16]

### Input Configuration

| Parameter            | Value |
|---------------------|-------|
| Matrix A Shape      | 16 × 16 |
| Matrix B Shape      | 16 × 16 |
| Output Matrix Shape | 16 × 16 |

### Performance Results

| Implementation              | Cycles | Speedup |
|-----------------------------|--------|---------|
| Scalar                      | 33,802 | 1.00×   |
| Vector (M1)                 | 7,066  | 4.78×   |
| Vector (M1) unrolled        | 2,532  | 13.35×  |
| Vector (M2)                 | 7,040  | 4.80×   |
| Vector (M2) unrolled        | 3,335  | 10.14×  |
| Vector (M4)                 | 7,035  | 4.80×   |
| Vector (M4) unrolled        | 3,388  | 9.98×   |
| Vector (M8)                 | 7,020  | 4.82×   |
| Vector (M8) unrolled        | 4,285  | 7.89×   |

---

## MATMUL [32 × 32] × [32 × 32]

### Input Configuration

| Parameter            | Value |
|---------------------|-------|
| Matrix A Shape      | 32 × 32 |
| Matrix B Shape      | 32 × 32 |
| Output Matrix Shape | 32 × 32 |

### Performance Results

| Implementation              | Cycles  | Speedup |
|-----------------------------|---------|---------|
| Scalar                      | 249,114 | 1.00×   |
| Vector (M1)                 | 30,840  | 8.08×   |
| Vector (M1) unrolled        | 9,873   | 25.23×  |
| Vector (M2)                 | 30,804  | 8.09×   |
| Vector (M2) unrolled        | 12,958  | 19.22×  |
| Vector (M4)                 | 30,803  | 8.09×   |
| Vector (M4) unrolled        | 13,009  | 19.15×  |
| Vector (M8)                 | 30,797  | 8.09×   |
| Vector (M8) unrolled        | 19,159  | 13.00×  |

---

## MATMUL [64 × 64] × [64 × 64]

### Input Configuration

| Parameter            | Value |
|---------------------|-------|
| Matrix A Shape      | 64 × 64 |
| Matrix B Shape      | 64 × 64 |
| Output Matrix Shape | 64 × 64 |

### Performance Results

| Implementation              | Cycles    | Speedup |
|-----------------------------|-----------|---------|
| Scalar                      | 4,808,029 | 1.00×   |
| Vector (M1)                 | 241,574   | 19.90×  |
| Vector (M1) unrolled        | 71,208    | 67.52×  |
| Vector (M2)                 | 153,792   | 31.26×  |
| Vector (M2) unrolled        | 68,417    | 70.27×  |
| Vector (M4)                 | 153,809   | 31.25×  |
| Vector (M4) unrolled        | 68,448    | 70.24×  |
| Vector (M8)                 | 153,803   | 31.26×  |
| Vector (M8) unrolled        | 98,498    | 48.81×  |
