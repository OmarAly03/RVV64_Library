## Benchmarks

This document summarizes the **highest speedups** obtained by the RVV implementations over their scalar baselines for each kernel category.  
All results were measured on the **ARA** RISC‑V vector accelerator (`VLEN = 1024 bits`, `4` vector lanes).  

Each row links to a more detailed benchmark log for that kernel, including all tested shapes and implementation variants.

---

### 1. Compute-Intensive Linear Operators

| Kernel   | Called Name             | Best Implementation              | Speedup  | Full Results |
|----------|-------------------------|----------------------------------|----------|--------------|
| `matmul` | Matrix Multiplication   | Vector (M2) unrolled (64×64)    | **70.27×** | [MatMul Benchmarks](./kernels/matmul/benchmarks.md) |
| `dense`  | Fully Connected Layer   | Vector (M8) (256×256)           | **4.01×**  | [Dense Benchmarks](./kernels/dense/benchmarks.md)   |

---

### 2. Pointwise Activation & Arithmetic Kernels

| Kernel        | Called Name                   | Best Implementation                 | Speedup   | Full Results                    |
|---------------|-------------------------------|-------------------------------------|-----------|---------------------------------|
| `relu`        | Rectified Linear Unit         | Vector (M8) (256×256)              | **20.08×** | [ReLU Benchmarks](./kernels/relu/benchmarks.md)           |
| `leaky_relu`  | Leaky Rectified Linear Unit   | Vector (M8) (256×256)              | **36.02×** | [Leaky ReLU Benchmarks](./kernels/leaky_relu/benchmarks.md) |
| `bias_add`    | Bias Addition                 | Vector (M8) (1×8×64×64)            | **21.75×** | [Bias Add Benchmarks](./kernels/bias_add/benchmarks.md)     |
| `tensor_add`  | Element-wise Tensor Addition  | Vector (M8) (65,536)               | **19.57×** | [Tensor Add Benchmarks](./kernels/tensor_add/benchmarks.md) |

---

### 3. Statistical & Normalization Kernels

| Kernel        | Called Name         | Best Implementation        | Speedup   | Full Results                               |
|---------------|---------------------|----------------------------|-----------|--------------------------------------------|
| `batch_norm`  | Batch Normalization | Vector (M8) ([1,128,32])   | **25.01×** | [Batch Norm Benchmarks](./kernels/batch_norm/benchmarks.md) |

---

### 4. Spatial Reduction Kernels

| Kernel   | Called Name  | Best Implementation                         | Speedup   | Full Results                         |
|----------|--------------|---------------------------------------------|-----------|--------------------------------------|
| `maxpool`| Max Pooling  | Vector (M8) (64×64, stride 1)              | **23.48×** | [Maxpool Benchmarks](./kernels/maxpool/benchmarks.md) |
