# RVV64 Library

The **RVV64 Library** is a modular collection of **RISC-V Vector (RVV) kernel implementations** and utilities for accelerating common deep learning and tensor operations using the **RVV (RISC-V Vector) extension**.  
It includes both **scalar** (reference) and **vectorized** implementations, as well as scripts to generate, test, and validate results using **ONNX models**.

---

## 🧠 Overview

This repository serves as a foundation for developing and benchmarking vectorized operators under the **RVV64** architecture.  
Each operator (e.g., `softmax`, `matmul`, `relu`, etc.) has:
- Scalar and vector RVV C++ implementations  
- Matching ONNX model generation scripts  
- Python tools for generating inputs and validating outputs  
- Makefiles for building and running experiments

The project also includes a full example model (**LeNet-5**) built from the available kernels to demonstrate end-to-end inference with RVV acceleration.

---

## 📁 Repository Structure

```
RVV64_Library/
│
├── lib/                      # Core vector intrinsic wrappers and utilities
│   ├── rvv_arithmetic.hpp
│   ├── rvv_vector_load.hpp
│   ├── rvv_vector_store.hpp
│   └── ...
│
├── kernels/                  # Individual kernel implementations
│   ├── bias_add/
│   ├── dense/
│   ├── matmul/
│   ├── relu/
│   ├── softmax/
│   └── tensor_add/
│
├── conv_transpose/           # Transposed convolution (deconvolution) operator
├── maxpool/                  # Max pooling operator
├── scatter_elements/         # Scatter elements operator
├── nms/                      # Non-Maximum Suppression operator
│
├── models/
│   └── lenet-5/              # Complete LeNet-5 implementation using RVV kernels
│
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## ⚙️ Installation

### 1. Prerequisites

You’ll need:
- **RISC-V GCC toolchain** supporting RVV (e.g., `riscv64-unknown-elf-g++`)
- **Python 3.10+**
- **ONNX** and related packages (`onnxruntime`, `numpy`, etc.)
- **CMake** and **Make**

Install the Python requirements:
```bash
pip install -r requirements.txt
```

---

## 🧩 Building and Running Kernels

Each kernel directory (e.g., `kernels/softmax`, `kernels/matmul`, etc.) contains:
- `main.py` – generates ONNX model and binary test data  
- `Makefile` – builds and runs the scalar and vector versions  
- `run_<kernel>.cpp` – entry point for running the kernel  
- `src/` – contains both RVV and scalar C++ implementations  

---

## 🧪 Example Model: LeNet-5

The `models/lenet-5/` directory demonstrates how to compose the RVV kernels into a complete neural network model.  
It includes:
- Layer parameters in `model_parameters/`
- Extracted sample images and binaries
- A `Makefile` to build and run the full model on an RVV simulator or target

Example:
```bash
cd models/lenet-5
make
make run
```

---

## 🧰 Core Library (`lib/`)

The `lib` folder contains reusable C++ headers wrapping **RVV intrinsics** for:
- Arithmetic operations (`rvv_arithmetic.hpp`)
- Vector loads/stores (`rvv_vector_load.hpp`, `rvv_vector_store.hpp`)
- Bitwise and mask operations (`rvv_mask_ops.hpp`, `rvv_bitwise.hpp`)
- MACC and min/max functions (`rvv_macc.hpp`, `rvv_minmax.hpp`)
- Vector length setup (`rvv_setVectorLength.hpp`)

These headers provide consistent, reusable building blocks for writing new kernels efficiently.

---

## 🧾 Results and Evaluation

Each operator directory includes a `results.md` file summarizing:
- Execution time (scalar vs RVV)
- Vector configurations (e.g., `e32m1`, `e32m8`)
- Observed performance gain or memory bandwidth efficiency
