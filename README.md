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
