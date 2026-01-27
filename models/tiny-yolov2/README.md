# Tiny-YOLOv2 Inference with RVV Vectorized Kernels

This project implements **Tiny-YOLOv2** end‑to‑end object detection using custom **RISC‑V Vector (RVV 1.0) kernels**.  
It mirrors a standard Tiny‑YOLOv2 ONNX model and runs:

* A native **C++ RVV implementation** for detection on RISC‑V (via QEMU).

The goal is to validate and demonstrate RVV‑accelerated convolutional workloads on a realistic detection model.

---

## 🏗 Project Structure

The Tiny‑YOLOv2 project is organized around a C++ inference pipeline, model parameter extraction, and visualization utilities:

```text
models/tiny-yolov2
├── main.cpp                     # C++ entry point (runs Tiny-YOLOv2 on an image)
├── Makefile                     # Build + run helpers (C++ + analysis)
├── README.md                    # This documentation
├── TinyYolov2.png               # Architecture / model diagram (reference)
├── visualize_results.py         # Python script to overlay and visualize detections
│
├── images/                      # Input test images (host format, e.g., JPG)
│   ├── cat.jpg
│   ├── man.jpg
│   └── ...
│
├── image_binaries/              # Preprocessed binary images for C++ input
│   └── .gitkeep
│
├── include/
│   ├── kernels.hpp              # Declarations of RVV/scalar kernels
│   ├── model.hpp                # High-level model driver
│   └── yolo_model.hpp           # Tiny-YOLOv2 network definition
│
├── src/
│   ├── kernels.cpp              # RVV/scalar kernel implementations
│   ├── model.cpp                # Orchestration: weights, network, outputs
│   ├── extract_image.py         # Convert images/* -> image_binaries/*.bin
│   └── extract_weights.py       # Extract .bin weights from tinyyolov2.onnx
│
├── model_parameters/            # Extracted weights and biases (.bin files)
│
├── onnx_model/
│   └── tinyyolov2.onnx          # Reference ONNX Tiny-YOLOv2 model
│
├── output_files/                # Raw C++ outputs (feature maps, predictions)
│
└── output_images/               # Visualized detection overlays (PNG/JPG)

```

---

## Getting Started

### Prerequisites

On the host (x86 or other non‑RISC‑V machine):

1. **RISC-V Toolchain**: To compile C++ with RVV support.
* `riscv64-unknown-linux-gnu-g++` with `-march=rv64gcv -mabi=lp64d`


2. **QEMU (user mode)**: To run the statically‑linked RVV binary.
* `qemu-riscv64 -cpu rv64,v=true`


3. **Python 3.x**: For preprocessing and visualization (`numpy`, `Pillow`).

> [!NOTE]
> You do not need a full RISC‑V Linux system image; Tiny‑YOLOv2 is run via user‑mode QEMU on a statically‑linked binary.

### 🛠 Makefile Actions

The Makefile orchestrates the typical workflow:

| Command | Action |
| --- | --- |
| `make` | Build the C++ Tiny‑YOLOv2 binary with RVV support. |
| `make run IMG=<name>` | Run C++ inference under QEMU on `images/<name>.jpg`. |
| `make extract_weights` | Run `src/extract_weights.py` to populate `model_parameters/`. |
| `make extract_images` | Convert `images/*.jpg` to `image_binaries/*.bin`. |
| `make clean` | Remove compiled binaries and temporary build objects. |

---

## Model Assets

### ONNX Model

The reference Tiny‑YOLOv2 network drives the layout of `model_parameters/*.bin` and the C++ network wiring.

### Weights & Biases

Populated by `src/extract_weights.py`, these are stored as **little‑endian IEEE‑754 floats (float32)**.

### Test Images

`src/extract_image.py` processes JPEGs into C++-friendly binaries:

1. **Resize/Letterbox** to resolution.
2. **Convert** to RGB float32.
3. **Normalize**

---

## Implementation Details

### Tiny‑YOLOv2 Architecture

The C++ implementation reproduces:

* **Convolution**: Stride and padding matching ONNX.
* **Nonlinearities**: LeakyReLU.
* **Max Pooling**.
* **Final YOLO head**:  tensor decoding.

### RVV Usage

All heavy operations use RVV intrinsics wrapped by generic helpers in the `/lib` directory:

* **Vectorized convolutions**: Inner products over channels and kernels.
* **Vectorized activation**: Batched LeakyReLU.
* **Memory operations**: Vectorized loads/stores and slides.

---

## Visualization and Outputs

`visualize_results.py` reconstructs bounding boxes, applies NMS (Non-maximum suppression), and draws labels.

**Example workflow:**

```bash
# Run C++ inference on cat.jpg
make run IMG=cat
```

---

## Typical Workflow

```bash
cd models/tiny-yolov2

make extract_weights
make extract_images
make
make run IMG=cat
# Result saved to output_images/output_detected_cat.jpg

```

---
