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
```

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

## 🚀 Performance Benchmarks

Our RVV-optimized kernels deliver significant performance improvements over scalar implementations.  
The highest speedups obtained (RVV vs. scalar) are:

| Kernel              | Speedup  |
|---------------------|----------|
| Matrix Multiplication | **70.27×** |
| Leaky ReLU            | **36.02×** |
| Batch Normalization   | **25.01×** |

These measurements were obtained on a **soft-core RISC-V vector processor** implemented by the **pulp-platform**:

-  The PULP Ara is a 64-bit Vector Unit, compatible with the RISC-V Vector Extension Version 1.0, working as a coprocessor to CORE-V's CVA6 core  
- Repository: [pulp-platform/ara](https://github.com/pulp-platform/ara)
- Configuration used for our benchmarks:
  - **Vector length (VLEN): 1024 bits**
  - **Number of vector lanes: 4**

> [!TIP]
> For the complete results across all kernels and more detailed benchmark methodology, see **[BENCHMARKS.md](./BENCHMARKS.md)**.

---

## 🖥️ RISC-V Ubuntu Image Setup (QEMU System Mode)

If you want to work with the `pyv/` Python wrappers, you'll need a Linux system capable of running a Python interpreter on RISC-V architecture. When actual RISC-V hardware isn't available, we recommend using an Ubuntu image specifically built for RISC-V on QEMU.

### Prerequisites Installation

First, install the required packages:
```bash
sudo apt update
sudo apt install opensbi qemu-system-riscv64 qemu-efi-riscv64 u-boot-qemu
```

### Download and Prepare the Ubuntu Image

1. Download **Ubuntu 24.04.3 LTS (Noble Numbat) RISC-V** for QEMU from the [official documentation](https://canonical-ubuntu-hardware-support.readthedocs-hosted.com/boards/how-to/qemu-riscv/)

2. Extract the image:
	```bash
	xz -dk ubuntu-*-preinstalled-server-riscv64.img.xz
	```

### Running the Image

Use the following command template:
```bash
qemu-system-riscv64 \
	 -machine virt \
	 -cpu rv64,v=true \
	 -m <RAM_SIZE> \
	 -smp <CPU_CORES> \
	 -nographic \
	 -kernel /usr/lib/u-boot/qemu-riscv64_smode/uboot.elf \
	 -device virtio-net-device,netdev=net0 \
	 -netdev user,id=net0 \
	 -drive file=<PATH_TO_UBUNTU_IMG>,format=raw,if=virtio
```

**Example configuration:**
```bash
qemu-system-riscv64 \
	 -machine virt \
	 -cpu rv64,v=true \
	 -m 4G \
	 -smp 4 \
	 -nographic \
	 -kernel /usr/lib/u-boot/qemu-riscv64_smode/uboot.elf \
	 -device virtio-net-device,netdev=net0 \
	 -netdev user,id=net0 \
	 -drive file=./ubuntu-24.04-preinstalled-server-riscv64.img,format=raw,if=virtio
```

> [!NOTE]
> The `-cpu rv64,v=true` flag enables RISC-V Vector Extension support, which is essential for running the RVV kernels.

---

