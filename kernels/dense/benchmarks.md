# Benchmarks of Dense (Fully Connected) Layer

---

## In = 64 | Out = 64

### Input Configuration

| Parameter     | Value |
|--------------|-------|
| Input Size   | 64 |
| Output Size  | 64 |
| Total MACs   | 4,096 |

### Performance Results

| Implementation | Cycles | Speedup |
|----------------|--------|---------|
| Scalar         | 37,076 | 1.00×   |
| Vector (M1)    | 11,261 | 3.29×   |
| Vector (M2)    | 10,236 | 3.62×   |
| Vector (M4)    | 10,224 | 3.63×   |
| Vector (M8)    | 10,225 | 3.63×   |

---

## In = 128 | Out = 128

### Input Configuration

| Parameter     | Value |
|--------------|-------|
| Input Size   | 128 |
| Output Size  | 128 |
| Total MACs   | 16,384 |

### Performance Results

| Implementation | Cycles | Speedup |
|----------------|--------|---------|
| Scalar         | 146,576 | 1.00×   |
| Vector (M1)    | 44,730  | 3.28×   |
| Vector (M2)    | 40,607  | 3.61×   |
| Vector (M4)    | 38,150  | 3.84×   |
| Vector (M8)    | 38,153  | 3.84×   |

---

## In = 256 | Out = 256

### Input Configuration

| Parameter     | Value |
|--------------|-------|
| Input Size   | 256 |
| Output Size  | 256 |
| Total MACs   | 65,536 |

### Performance Results

| Implementation | Cycles | Speedup |
|----------------|--------|---------|
| Scalar         | 589,635 | 1.00×   |
| Vector (M1)    | 178,470 | 3.30×   |
| Vector (M2)    | 162,021 | 3.64×   |
| Vector (M4)    | 152,253 | 3.87×   |
| Vector (M8)    | 147,133 | 4.01×   |
