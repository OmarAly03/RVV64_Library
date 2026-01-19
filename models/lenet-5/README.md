# LeNet-5 Inference with RVV Vectorized Kernels

This project implements the classic **LeNet-5** convolutional neural network architecture for digit recognition. It features high-performance inference using **RISC-V Vector (RVV 1.0) kernels** and provides two ways to interact with the model: a native **C++ implementation** and a **Python wrapper (pyv)** that utilizes the vectorized C kernels for backend computations.

## 🏗 Project Structure

The repository is organized into a core C++ inference engine and a Python-based visualization/testing suite:

```text
.
├── C/                       # C++ Implementation
│   ├── include/             # Headers (Architecture & RVV Kernel defs)
│   │   ├── config.hpp
│   │   ├── defs.hpp
│   │   └── lenet5.hpp
│   ├── src/                 # Source files
│   │   ├── kernels.cpp      # RVV 1.0 Vectorized Kernels
│   │   └── lenet5.cpp       # LeNet-5 Pipeline Logic
│   └── main.cpp             # C++ Entry Point
├── py/                      
│   └── main.py              # Python Implementation using RVV kernels (via pyv)
├── model_parameters/        # Extracted .bin weights and biases
├── images/                  # Source PNG images for testing
├── image_binaries/          # Preprocessed binary images for C++ input
├── onnx_model/              # Original ONNX model and graph
├── Makefile                 # Unified build and run system
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- **RISC-V Toolchain**: Required for compiling the C++ code with RVV support.
- **QEMU (User mode)**: To run the RVV C++ binaries on non-RISC-V hardware.
- **QEMU (System mode)**: To emulate a full RISC-V OS environment for running the Python implementation with PIL, numpy, matplotlib, and the RVV C++ binaries.
- **Python 3.x (On the RISC-V Emulated System)**: With numpy, matplotlib and PIL/Pillow for the Python suite.

### 🛠 Makefile Actions

The Makefile at the root simplifies the compilation and execution flow.

| Command | Action |
|---------|--------|
| `make run_c` | Runs the C++ prediction using a binary image. |
| `make run_py` | Runs the Python prediction (using vectorized wrappers). |
| `make clean` | Removes compiled binaries and temporary objects. |

## 🧠 Implementation Details

### RVV Vectorized Kernels

The heart of the project lies in `C/src/kernels.cpp`. Unlike standard scalar implementations, these kernels use RISC-V Vector instructions to process multiple data points in parallel, significantly accelerating:

- **Conv2D**: Spatial convolution with optimized channel loops.
- **MaxPool**: Vectorized window comparisons.
- **Dense (FC)**: Matrix-vector multiplication using vector accumulation.
- **ReLU/Softmax**: Element-wise vector activations.

### Python Wrapper (pyv)

The Python implementation in `py/main.py` serves as a high-level interface. It performs image preprocessing (normalization, resizing) and then calls the vectorized kernels via a shared library wrapper. This allows for rapid prototyping and visualization while maintaining the performance of C++/RVV.

## 📊 Performance & Output

### C++ Inference Output

To run the C++ inference:

```bash
make run DIGIT=6
```

C++ Output:

```plaintext
Running LeNet-5 on QEMU...
Loading weights...
All 12 weights/biases loaded.
Loading image: ./image_binaries/6.bin

Prediction: 6
Expected: 6
Result: CORRECT
```

### Python Inference Output

To run the Python visualization:

```bash
make run_py DIGIT=6
```


#### Python Visualization Examples

Here are examples of the Python wrapper predictions for different MNIST digits:

![Prediction for digit 0](images/lenet5_prediction_0.png)

![Prediction for digit 3](images/lenet5_prediction_3.png)

![Prediction for digit 4](images/lenet5_prediction_4.png)

![Prediction for digit 6](images/lenet5_prediction_6.png)

![Prediction for digit 8](images/lenet5_prediction_8.png)

Python Output:

```plaintext
Loading LeNet-5 parameters from /mnt/host/models/lenet-5/model_parameters...
Using input file: /mnt/host/models/lenet-5/images/6.png
The model predicted: 6
```

## 🛠 Model Weights

The weights are stored as raw IEEE 754 floating-point binaries in the `model_parameters/` folder. These were extracted from the original ONNX model to ensure compatibility with the custom C++ inference engine without needing a heavy runtime like ONNXRuntime.
