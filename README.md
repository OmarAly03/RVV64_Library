# RISC-V Vector (RVV) Library – Matrix Multiplication & ReLU

This branch contains RISC-V Vector implementations of common computational kernels with functional verification using ONNX models. The goal is to validate different RVV implementations by comparing them against ONNX golden reference outputs.

## Kernels

### 1. Matrix Multiplication (`matmul/`)

**ONNX model:** `matrix_multiply.onnx`

**Implementations:**
- Python scalar (`matmult.py`)
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

**Parameters:**
- `SIZE`: The size of the matrices.

### 2. ReLU Activation (`relu/`)

**ONNX model:** `relu.onnx`

**Implementations:**
- Python scalar (`relu_scalar.py`)
- C scalar (`relu_scalar`)
- **RISC-V Vector intrinsics:**
  - `relu_e32m1` (LMUL=1)
  - `relu_e32m2` (LMUL=2) 
  - `relu_e32m4` (LMUL=4)
  - `relu_e32m8` (LMUL=8)

**Parameters:**
- `SIZE`: The size of the input tensor.

### 3. Non-Maximum Suppression (`nms/`)

**ONNX model:** `nms.onnx`

**Implementations:**
- Python reference (`nms_utils.py`)
- C scalar (`nms_scalar`)
- **RISC-V Vector intrinsics:**
  - `nms_e32m1` (LMUL=1)
  - `nms_e32m2` (LMUL=2)
  - `nms_e32m4` (LMUL=4)
  - `nms_e32m8` (LMUL=8)

**Features:**
- Implements the ONNX `NonMaxSuppression` operator.
- Supports multi-batch and multi-class inputs.
- Filters boxes by score and IoU thresholds.
- Limits the number of output boxes per class.

**Parameters:**
- `BATCHES`: The number of batches.
- `CLASSES`: The number of classes.
- `SPATIAL`: The spatial dimension (number of boxes).
- `MAX_BOXES`: The maximum number of output boxes per class.
- `IOU_THRESH`: The IOU threshold.
- `SCORE_THRESH`: The score threshold.

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

**Non-Maximum Suppression:**
```bash
cd nms/
make clean
make
make run SPATIAL=4096 BATCHES=2 CLASSES=3
```

### Python-only Testing (x86)

```bash
# Matmul: can be run with no arguments (default size), one argument (SIZE), or three arguments (M, N, K)
python3 matmul/main.py
python3 matmul/main.py <SIZE>
python3 matmul/main.py <M> <N> <K>

# ReLU: can be run with no arguments (default size) or one argument (SIZE)
python3 relu/main.py
python3 relu/main.py <SIZE>

# NMS: can be run with no arguments (default values) or six arguments
python3 nms/main.py
python3 nms/main.py <BATCHES> <CLASSES> <SPATIAL> <MAX_BOXES> <IOU_THRESH> <SCORE_THRESH>
```

## Performance Metrics

All kernels report:
- **SNR (Signal-to-Noise Ratio):** Higher is better (∞ = perfect match)
- **Max Absolute Error:** Lower is better (0 = perfect match)
