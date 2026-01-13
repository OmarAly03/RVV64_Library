# RVV64_Library

A collection of **RISC-V Vector (RVV)** accelerated kernels and utilities for high-performance computing and machine learning workloads.  
This repository implements a variety of computational kernels using RISC-V Vector instructions and organizes them into a reusable library — suitable for benchmarking, integration, and further development.

---

## 📁 Repository Structure

The project is organized into the following top-level directories:
```
RVV64_Library​
├── kernels
├── lib
├── libso
├── models
├── pyv
├── README.md
├── requirements.txt
└── scripts
```

---

### 📦 `kernels/`

Contains the **core RVV-optimized kernel implementations** and their tests.  
Each kernel corresponds to a key numerical primitive (e.g., `matmul`, `conv`, `batch_norm`, etc.) used in deep learning and scientific computing workloads.  
These are often written in C/C++ with RVV intrinsics or assembly for maximum performance.

---

### 📦 `lib/`

This directory contains **low-level RVV vector APIs** implemented as C++ headers.  
These files wrap RVV instructions into reusable building blocks:

- Vector loads & stores
- Mask operations
- Reductions
- Indexed loads
- Narrowing & reinterpretation
- Multiply-accumulate
- Vector length control

These are used internally by all kernels and models to keep the RVV code clean, portable, and maintainable.

---

### 📦 `libso/`

Contains files to build the **shared (dynamic) library** version of the RVV kernels.  
Use this when you want to link dynamically against the RVV64 library from other applications.

---

### 📦 `models/`

Includes sample or reference **data/model files** used for validating kernels or running example benchmarks.

---

### 📦 `pyv/`

Python bindings and utilities for the library.  
This section provides a bridge to use RVV kernels from Python — ideal for quick experimentation and scripting.

---

### 📄 `requirements.txt`

Python package dependencies used by the Python tools / benchmarks under `pyv/` or `scripts/`.  
Install them before running Python-based utilities:

```bash
pip install -r requirements.txt

## Kernel Categorization

### 1. Compute-Intensive Linear Operators
- Matrix Multiplication (`matmul`)
- Fully Connected Layer (`dense`)
- Convolution (`conv`)
- Transposed Convolution (`conv_transpose`)

### 2. Pointwise Activation & Arithmetic Kernels
- Rectified Linear Unit (`relu`)
- Leaky Rectified Linear Unit (`leaky_relu`)
- Bias Addition (`bias_add`)
- Element-wise Tensor Addition (`tensor_add`)

### 3. Statistical & Normalization Kernels
- Batch Normalization (`batch_norm`)
- Softmax Probability Function (`softmax`)

### 4. Spatial Reduction Kernels
- Max Pooling (`maxpool`)

### 5. Tensor Indexing & Data Movement (ONNX-style)
- Tensor Gather (`gather`)
- Indexed Element Gathering (`gather_elements`)
- Indexed Element Scattering (`scatter_elements`)

### 6. Post-Processing & Decision Kernels
- Non-Maximum Suppression (`nms`)

---

## Benchmarks

### 1. Compute-Intensive Linear Operators

| Kernel | Called Name | Best Implementation | Speedup |
|--------|-------------|---------------------|---------|
| `matmul` | Matrix Multiplication | Vector (M2) unrolled (64×64) | **70.27×** |
| `dense` | Fully Connected Layer | Vector (M8) (256×256) | **4.01×** |

---

### 2. Pointwise Activation & Arithmetic Kernels

| Kernel | Called Name | Best Implementation | Speedup |
|--------|-------------|---------------------|---------|
| `relu` | Rectified Linear Unit | Vector (M8) (256×256) | **20.08×** |
| `leaky_relu` | Leaky Rectified Linear Unit | Vector (M8) (256×256) | **36.02×** |
| `bias_add` | Bias Addition | Vector (M8) (1×8×64×64) | **21.75×** |
| `tensor_add` | Element-wise Tensor Addition | Vector (M8) (65,536) | **19.57×** |

---

### 3. Statistical & Normalization Kernels

| Kernel | Called Name | Best Implementation | Speedup |
|--------|-------------|---------------------|---------|
| `batch_norm` | Batch Normalization | Vector (M8) ([1,128,32]) | **25.01×** |

---

### 4. Spatial Reduction Kernels

| Kernel | Called Name | Best Implementation | Speedup |
|--------|-------------|---------------------|---------|
| `maxpool` | Max Pooling | Vector (M8) (64×64, stride 1) | **23.48×** |

---

## Image Installation
- Download the Ubuntu 24.04.3 LTS (Noble Numbat) RISC-V for QEMU from [HERE](https://canonical-ubuntu-hardware-support.readthedocs-hosted.com/boards/how-to/qemu-riscv/)
- Install the prerequisites
	```bash
	sudo apt update
	sudo apt install opensbi qemu-system-riscv64 qemu-efi-riscv64 u-boot-qemu
	```
- Unpack the image:
	```bash
	xz -dk ubuntu-*-preinstalled-server-riscv64.img.xz
	```

- To run the image with qemu-system
	```bash
	qemu-system-riscv64 \
	 -machine virt \
	 -cpu rv64,v=true \
	 -m <RAMSIZE> \
	 -smp <NO.CPU_CORES> \
	 -nographic \
	 -kernel /usr/lib/u-boot/qemu-riscv64_smode/uboot.elf \
	 -device virtio-net-device,netdev=net0 \
	 -netdev user,id=net0 \
	 -drive file=<PATH_TO_UBUNTU_IMG>,format=raw,if=virtio
	 ```
