# RISC-V Vector (RVV) Library – Matrix Multiplication & ReLU

This repository contains RISC-V Vector implementations of common computational kernels with functional verification using ONNX models. The goal is to validate different RVV implementations by comparing them against ONNX golden reference outputs.

## Kernels

### 1. Matrix Multiplication (`matmul/`)

**ONNX model:** `matrix_multiply.onnx`

**Implementations:**
- Python scalar (`matmult.py`)
- Python (NumPy)
- C scalar (`matmul_scalar`)
- **RISC-V Vector intrinsics:**
  - `matmul_e32m1` (LMUL=1)
  - `matmul_e32m2` (LMUL=2)
  - `matmul_e32m4` (LMUL=4)
  - `matmul_e32m8` (LMUL=8)

**Features:**
- Supports arbitrary matrix sizes (M×K @ K×N → M×N)
- Vectorized across output matrix columns
- Fused multiply-accumulate (FMA) operations

### 2. ReLU Activation (`relu/`)

**ONNX model:** `relu.onnx`

**Implementations:**
- Python scalar (`relu_scalar.py`)
- Python (NumPy maximum)
- C scalar (`relu_scalar`)
- **RISC-V Vector intrinsics:**
  - `relu_e32m1` (LMUL=1)
  - `relu_e32m2` (LMUL=2) 
  - `relu_e32m4` (LMUL=4)
  - `relu_e32m8` (LMUL=8)

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```
## Usage

### Build and Run Tests

**Matrix Multiplication:**
```bash
cd matmul/
make clean
make       
make run SIZE=512    
```

**ReLU Activation:**
```bash
cd relu/
make clean
make         
make run SIZE=4096    
```

### Python-only Testing (x86)

```bash
python3 matmul/main.py
python3 relu/main.py
```

## Performance Metrics

Both kernels report:
- **SNR (Signal-to-Noise Ratio):** Higher is better (∞ = perfect match)
- **Max Absolute Error:** Lower is better (0 = perfect match)
