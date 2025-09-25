# ONNX Testing Kernels – Matrix Multiplication & ReLU

This branch contains test kernels for functional verification using ONNX models.
The goal is to validate different implementations of matrix multiplication and ReLU by comparing them against the ONNX golden reference output.

## Kernels

### 1. Matrix Multiplication (`matmul/`)

**ONNX model:** `matrix_multiply.onnx`

**Implementations:**
- Python scalar (`matmult.py`)
- Python vectorized (NumPy)
- C scalar (`matmul.c` with wrapper `matmul_wrapper.py`)

**Metrics:**
- SNR (Signal-to-Noise Ratio)
- Max Absolute Difference

Results are documented in `matmul/x86/results.md`

### 2. ReLU (`relu/`)

**ONNX model:** `relu.onnx`

**Implementations:**
- Python scalar (`relu_scalar.py`)
- C scalar (`relu.c` with wrapper `relu_wrapper.py`)

**Metrics:**
- SNR (Signal-to-Noise Ratio)
- Max Absolute Difference

Results are documented in `relu/x86/results.md`

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Run the test runners:

```bash
python3 matmul/x86/main.py
python3 relu/x86/main.py
```
